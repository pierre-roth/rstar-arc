import logging
import os
import random
import sys
from dataclasses import asdict

import torch
from accelerate import Accelerator
from datasets import DatasetDict, load_dataset
from torch.optim import AdamW
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_scheduler,
    set_seed,
)

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from constants import (
    LOCAL_SCRATCH_PATH,
    NET_SCRATCH_PATH,
    SFT_IN_BETWEEN_PROMPT,
    SFT_SYSTEM_PROMPT,
    SPECIAL_TOKENS,
)
from utils import setup_logging
from train_utils import renormalize_task_weights, WeightedCollator
from rstar_deepthink import Config
from rstar_deepthink.arc_task import ARCTask
from rstar_deepthink.arc_task.task_utils import task_to_prompt

logger = logging.getLogger(__name__)


# -------------------------------------------------------------
# preprocessing
# -------------------------------------------------------------

def preprocess(batch):
    """Tokenize solutions. Only tokens after the prompt contribute to the loss."""
    prompts: list[str] = []
    full_texts: list[str] = []
    for task_json, solution in zip(batch["task_json"], batch["solution"]):
        prompt_text = (
                SFT_SYSTEM_PROMPT
                + task_to_prompt(ARCTask.from_dict(task_json))
                + SFT_IN_BETWEEN_PROMPT
        )
        prompts.append(prompt_text)
        full_texts.append(prompt_text + solution + tok.eos_token)

    model_inputs = tok(
        full_texts,
        max_length=config.max_seq_len,
        truncation=True,
        padding=False,
        return_attention_mask=True,
    )

    labels: list[list[int]] = []
    for i in range(len(full_texts)):
        prompt_len = len(tok(prompts[i], add_special_tokens=False).input_ids)
        input_ids = model_inputs["input_ids"][i]
        lbl = list(input_ids)
        for j in range(prompt_len):
            if j < len(lbl):
                lbl[j] = -100
        labels.append(lbl)

    model_inputs["labels"] = labels
    model_inputs["weight"] = [float(w) for w in batch["weight"]]
    return model_inputs


# -------------------------------------------------------------
# helper functions
# -------------------------------------------------------------

def compute_loss(logits: torch.Tensor, labels: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
    per_token = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
    per_token = per_token.view(shift_labels.size())
    active = shift_labels != -100
    per_seq = (per_token * active).sum(dim=1) / active.sum(dim=1).clamp_min(1)
    weight = weights.to(per_seq.device)
    return (per_seq * weight).sum() / weight.sum().clamp_min(1e-8)


def evaluate(model, dataloader, accelerator: Accelerator, prefix: str, step: int) -> float:
    model.eval()
    all_losses = []
    all_weights = []
    for batch in dataloader:
        with torch.no_grad():
            outputs = model(**batch)
        
        # Calculate per-sequence loss, but don't average it yet
        logits = outputs.logits
        labels = batch["labels"]
        weights = batch["weight"]

        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
        per_token = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        per_token = per_token.view(shift_labels.size())
        
        active = shift_labels != -100
        per_seq_loss = (per_token * active).sum(dim=1) / active.sum(dim=1).clamp_min(1)

        # Gather per-sequence losses and weights from all GPUs
        gathered_losses = accelerator.gather_for_metrics(per_seq_loss)
        gathered_weights = accelerator.gather_for_metrics(weights)
        
        all_losses.append(gathered_losses)
        all_weights.append(gathered_weights)

    # Now, calculate the final weighted average loss on the main process
    total_loss = 0.0
    if accelerator.is_main_process:
        loss_tensor = torch.cat(all_losses)
        weight_tensor = torch.cat(all_weights)
        
        # Ensure tensors are on the same device for the final calculation
        weight_tensor = weight_tensor.to(loss_tensor.device)

        total_loss_tensor = (loss_tensor * weight_tensor).sum() / weight_tensor.sum().clamp_min(1e-8)
        total_loss = total_loss_tensor.item()
        
        if config.report_to == "wandb":
            accelerator.log({f"{prefix}/loss": total_loss}, step=step)
    
    model.train()
    # Return the computed loss if on the main process, otherwise a placeholder
    return total_loss


# -------------------------------------------------------------
# main setup
# -------------------------------------------------------------

config = Config()
set_seed(config.seed or 42)
setup_logging(config.numeric_log_level)

TRAIN_PATH = os.path.join(NET_SCRATCH_PATH, "sft_data", f"round_{config.round_number}", config.training_dataset_name)
VAL_PATH = os.path.join(NET_SCRATCH_PATH, "sft_data", f"round_{config.round_number}", config.validation_dataset_name)

run_name = f"policy-ft-{config.policy_model.split('/')[-1]}"
OUT_DIR = os.path.join(NET_SCRATCH_PATH, "models", "fine_tuned", "policy", run_name)
BEST_MODEL_DIR = os.path.join(OUT_DIR, "best_model")
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(BEST_MODEL_DIR, exist_ok=True)

accelerator = Accelerator(log_with="wandb" if config.report_to == "wandb" else None)
if accelerator.is_main_process and config.report_to == "wandb":
    accelerator.init_trackers(config.wandb_project, config=asdict(config))

logger.info(f"Loading tokenizer and model: {config.policy_model}")

tok = AutoTokenizer.from_pretrained(config.policy_model, trust_remote_code=True)
tok.pad_token = tok.pad_token or tok.eos_token
tok.pad_token_id = tok.pad_token_id or tok.eos_token_id
tok.padding_side = "right"
added_tokens = tok.add_special_tokens({"additional_special_tokens": SPECIAL_TOKENS})

model = AutoModelForCausalLM.from_pretrained(
    config.policy_model,
    torch_dtype=torch.bfloat16 if config.use_bf16 else torch.float16,
    trust_remote_code=True,
    attn_implementation=config.attn_implementation,
)
if added_tokens > 0:
    model.resize_token_embeddings(len(tok))
model.config.use_cache = False
if config.gradient_checkpointing:
    model.gradient_checkpointing_enable()
model.enable_input_require_grads()

logger.info("Loading training dataset …")
raw_dataset = load_dataset(
    "json",
    data_files={"train": TRAIN_PATH},
    cache_dir=os.path.join(LOCAL_SCRATCH_PATH, ".cache/huggingface/datasets"),
)
num_proc = max(1, config.cpus - 1)


def _within_max_len(example):
    text = (
            SFT_SYSTEM_PROMPT
            + task_to_prompt(ARCTask.from_dict(example["task_json"]))
            + SFT_IN_BETWEEN_PROMPT
            + example["solution"]
    )
    return len(tok.encode(text)) <= config.max_seq_len


orig_train_len = len(raw_dataset["train"])
raw_dataset["train"] = raw_dataset["train"].filter(_within_max_len, num_proc=num_proc)

rng = random.Random(config.seed or 42)
task_to_indices: dict[str, list[int]] = {}
for idx, ex in enumerate(raw_dataset["train"]):
    task_to_indices.setdefault(ex["task_name"], []).append(idx)

val_task_indices: list[int] = []
if config.task_validation_fraction > 0:
    all_tasks = list(task_to_indices.keys())
    rng.shuffle(all_tasks)
    val_task_count = int(len(all_tasks) * config.task_validation_fraction)
    val_tasks = set(all_tasks[:val_task_count])
else:
    val_tasks = set()

for task, indices in task_to_indices.items():
    idxs = list(indices)
    rng.shuffle(idxs)
    if task in val_tasks:
        val_task_indices.extend(idxs)

val_example_indices: list[int] = []
train_final_indices: list[int] = []
for task, indices in task_to_indices.items():
    if task in val_tasks:
        continue
    idxs = list(indices)
    rng.shuffle(idxs)
    if config.example_validation_num > 0 and len(idxs) >= config.example_validation_threshold:
        n_take = min(config.example_validation_num, len(idxs) // 2)
        val_example_indices.extend(idxs[:n_take])
        train_final_indices.extend(idxs[n_take:])
    else:
        train_final_indices.extend(idxs)

raw_val = load_dataset(
    "json",
    data_files={"validation": VAL_PATH},
    cache_dir=os.path.join(LOCAL_SCRATCH_PATH, ".cache/huggingface/datasets"),
)
orig_val_len = len(raw_val["validation"])
raw_val["validation"] = raw_val["validation"].filter(_within_max_len, num_proc=num_proc)

train_ds = renormalize_task_weights(raw_dataset["train"].select(train_final_indices))
val_task_ds = renormalize_task_weights(raw_dataset["train"].select(val_task_indices)) if val_task_indices else None
val_example_ds = renormalize_task_weights(
    raw_dataset["train"].select(val_example_indices)) if val_example_indices else None
val_val_ds = renormalize_task_weights(raw_val["validation"]) if len(raw_val["validation"]) > 0 else None

dataset = DatasetDict({"train": train_ds})
if val_task_ds is not None:
    dataset["val_task"] = val_task_ds
if val_example_ds is not None:
    dataset["val_example"] = val_example_ds
if val_val_ds is not None:
    dataset["val_val"] = val_val_ds

dataset["train"] = dataset["train"].shuffle(seed=config.seed or 42)

logger.info("Tokenizing dataset …")
tokenized_datasets = DatasetDict()
for split, ds in dataset.items():
    if len(ds) == 0:
        tokenized_datasets[split] = ds
        continue
    tokenized_datasets[split] = ds.map(
        preprocess,
        batched=True,
        remove_columns=[c for c in dataset["train"].column_names if c != "weight"],
        num_proc=num_proc,
    )

collator = WeightedCollator(tokenizer=tok)
train_loader = torch.utils.data.DataLoader(
    tokenized_datasets["train"],
    batch_size=config.per_device_train_batch_size,
    shuffle=True,
    num_workers=4,
    collate_fn=collator,
)

val_loaders = {}
for name in ("val_task", "val_example", "val_val"):
    ds = tokenized_datasets.get(name)
    if ds is not None:
        val_loaders[name] = torch.utils.data.DataLoader(
            ds,
            batch_size=config.per_device_eval_batch_size,
            shuffle=False,
            num_workers=4,
            collate_fn=collator,
        )

optimizer = AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
steps_per_epoch = len(train_loader) // config.gradient_accumulation_steps + int(
    len(train_loader) % config.gradient_accumulation_steps != 0)
max_train_steps = steps_per_epoch * config.num_train_epochs
lr_scheduler = get_scheduler(
    config.lr_scheduler_type,
    optimizer=optimizer,
    num_warmup_steps=int(config.warmup_ratio * max_train_steps),
    num_training_steps=max_train_steps,
)

model, optimizer, train_loader, lr_scheduler, *val_loader_list = accelerator.prepare(
    model,
    optimizer,
    train_loader,
    lr_scheduler,
    *val_loaders.values(),
)

# mapping after accelerator.prepare
prepared_val_loaders = dict(zip(val_loaders.keys(), val_loader_list))

global_step = 0
best_eval_loss = float('inf')
best_step = 0
logger.info("Starting training …")

for epoch in range(config.num_train_epochs):
    for step, batch in enumerate(train_loader):
        with accelerator.accumulate(model):
            outputs = model(**batch)
            loss = compute_loss(outputs.logits, batch["labels"], batch["weight"])
            accelerator.backward(loss)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

        if accelerator.is_main_process and global_step % config.logging_steps == 0:
            accelerator.log({"train/loss": loss.item(), "lr": lr_scheduler.get_last_lr()[0]}, step=global_step)

        if global_step % config.eval_steps == 0 and global_step > 0:
            for name, loader in prepared_val_loaders.items():
                eval_loss = evaluate(model, loader, accelerator, name, global_step)
                if name == "val_val":
                    accelerator.log({"eval_loss": eval_loss}, step=global_step)
                    if accelerator.is_main_process and eval_loss < best_eval_loss:
                        best_eval_loss = eval_loss
                        best_step = global_step
                        logger.info(f"New best val_val loss: {best_eval_loss:.4f} at step {best_step}. Saving model to {BEST_MODEL_DIR}")
                        accelerator.wait_for_everyone()
                        unwrapped = accelerator.unwrap_model(model)
                        unwrapped.save_pretrained(BEST_MODEL_DIR, safe_serialization=True)
                        tok.save_pretrained(BEST_MODEL_DIR)

        if global_step % config.save_steps == 0 and accelerator.is_main_process:
            accelerator.wait_for_everyone()
            unwrapped = accelerator.unwrap_model(model)
            unwrapped.save_pretrained(OUT_DIR, safe_serialization=True)
            tok.save_pretrained(OUT_DIR)

        global_step += 1
        if global_step >= max_train_steps:
            break
    if global_step >= max_train_steps:
        break

accelerator.wait_for_everyone()
if accelerator.is_main_process:
    unwrapped = accelerator.unwrap_model(model)
    unwrapped.save_pretrained(OUT_DIR, safe_serialization=True)
    tok.save_pretrained(OUT_DIR)
    logger.info(f"Training finished. Best val_val loss: {best_eval_loss:.4f} at step {best_step}")

for name, loader in prepared_val_loaders.items():
    eval_loss = evaluate(model, loader, accelerator, name, global_step)
    if name == "val_val" and accelerator.is_main_process:
        logger.info(f"Final val_loss: {eval_loss:.4f}")

if config.report_to == "wandb" and accelerator.is_main_process:
    accelerator.end_training()
