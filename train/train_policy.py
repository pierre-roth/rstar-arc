"""
LoRA-based supervised fine-tuning for the *policy* model that writes ARC code.

Everything is driven by rstar_deepthink.Config:
  • model + dataset paths
  • LoRA hyper-params
  • batch-size, LR, logging, etc.
Multi-GPU training is handled via Accelerate.

Launch with:
  # 4 GPUs with torchrun
  torchrun --nproc_per_node 4 train/train_policy.py --config-file configs/train_policy.yaml
or
  # via Accelerate
  accelerate launch train/train_policy.py --config-file configs/train_policy.yaml
"""

from __future__ import annotations

import logging
import os
import random
import sys
from dataclasses import asdict

import torch
import wandb
from datasets import DatasetDict
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    set_seed,
)

# ------------------- project imports -------------------
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from constants import (
    SFT_SYSTEM_PROMPT,
    SFT_IN_BETWEEN_PROMPT,
    LOCAL_SCRATCH_PATH,
    NET_SCRATCH_PATH,
    CODE_END,
    SPECIAL_TOKENS
)
from utils import setup_logging
from train_utils import (
    maybe_peft_wrap,
    renormalize_task_weights,
    WeightedCollator,
)
from rstar_deepthink import Config
from rstar_deepthink.arc_task import ARCTask
from rstar_deepthink.arc_task.task_utils import task_to_prompt

logger = logging.getLogger(__name__)

# ------------------- config -------------------
config = Config()
set_seed(config.seed or 42)
setup_logging(config.numeric_log_level)
logger.info(f"Using model: {config.policy_model}")

TRAIN_PATH = os.path.join(NET_SCRATCH_PATH, "sft_data", f"round_{config.round_number}", config.training_dataset_name)
VAL_PATH = os.path.join(NET_SCRATCH_PATH, "sft_data", f"round_{config.round_number}", config.validation_dataset_name)

if not config.full_finetune:
    dir_name = (
        f"ft-{config.policy_model.split('/')[-1]}-"
        f"{config.max_seq_len}-{config.learning_rate}-"
        f"{config.lora_rank}-{config.lora_alpha}-"
        f"{config.lora_dropout}-"
        f"{config.weight_decay}-"
        f"{config.train_on_prompts}"
    )
else:
    dir_name = (
        f"ft-{config.policy_model.split('/')[-1]}-"
        f"{config.max_seq_len}-{config.learning_rate}-"
        f"{config.weight_decay}-"
        f"{config.train_on_prompts}"
    )

# The directory where checkpoints/adapters are stored – unchanged semantics.
OUT_DIR = os.path.join(NET_SCRATCH_PATH, "models", "fine_tuned", "policy", dir_name)

# A more explicit run name for logs & experiment tracking.  This change is purely
# cosmetic and does **not** influence training behaviour.
RUN_NAME = f"policy-{dir_name}"
os.makedirs(OUT_DIR, exist_ok=True)

# ------------------- tokenizer -------------------
tok = AutoTokenizer.from_pretrained(config.policy_model, trust_remote_code=True)
tok.pad_token = tok.pad_token or tok.eos_token
tok.pad_token_id = tok.pad_token_id or tok.eos_token_id
tok.padding_side = "right"

added_tokens = tok.add_special_tokens({"additional_special_tokens": SPECIAL_TOKENS})
if added_tokens > 0:
    logger.info(f"Added {added_tokens} special tokens to tokenizer")

# Generation should stop on either regular EOS or CODE_END
EOS_TOKEN_IDS = [tok.eos_token_id, tok.convert_tokens_to_ids(CODE_END)]

# ------------------- model -------------------
model = AutoModelForCausalLM.from_pretrained(
    config.policy_model,
    torch_dtype=torch.bfloat16 if config.use_bf16 else torch.float16,
    trust_remote_code=True,
    attn_implementation=config.attn_implementation,
)

# Resize embeddings if we added new tokens
if added_tokens > 0:
    model.resize_token_embeddings(len(tok))

# ensure pad_token_id is set on the model to avoid fallback warning during generation
if model.config.pad_token_id is None:
    model.config.pad_token_id = tok.pad_token_id

# Multi-GPU training is handled via Accelerate launch
model.config.use_cache = False
model.gradient_checkpointing_enable()  # save memory

model.enable_input_require_grads()

# Wrap model with LoRA adapters or full-model fine-tuning as configured
model = maybe_peft_wrap(model, config)

# log trainable params
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total = sum(p.numel() for p in model.parameters())
logger.info(
    f"Trainable params: {trainable / 1e6:.1f} M / {total / 1e6:.1f} M ({100 * trainable / total:.2f}%)"
)


# ------------------- preprocessing -------------------
def preprocess_prompt_and_completion(batch):
    texts = [
        SFT_SYSTEM_PROMPT +
        task_to_prompt(ARCTask.from_dict(j)) +
        SFT_IN_BETWEEN_PROMPT + sol + tok.eos_token
        for j, sol in zip(batch["task_json"], batch["solution"])
    ]
    # ↓ removed `padding="max_length"` to enable dynamic padding
    model_inp = tok(
        texts,
        max_length=config.max_seq_len,
        truncation=True,
        padding=False,
        return_attention_mask=True
    )

    model_inp["labels"] = model_inp["input_ids"].copy()
    model_inp["weight"] = [float(w) for w in batch["weight"]]
    return model_inp


def preprocess_for_completion_only(batch):
    """
    Preprocesses the batch to train the model only on completing the solution, given the prompt.
    The prompt tokens in the labels will be masked.
    """
    prompts = []
    full_texts = []

    for task_json, solution in zip(batch["task_json"], batch["solution"]):
        prompt_text = (
                SFT_SYSTEM_PROMPT +
                task_to_prompt(ARCTask.from_dict(task_json)) +
                SFT_IN_BETWEEN_PROMPT
        )
        prompts.append(prompt_text)
        full_texts.append(prompt_text + solution + tok.eos_token)

    # ↓ removed `padding="max_length"` to enable dynamic padding
    model_inputs = tok(
        full_texts,
        max_length=config.max_seq_len,
        truncation=True,
        padding=False,
        return_attention_mask=True
    )

    labels = []
    for i in range(len(full_texts)):
        prompt_token_ids = tok(prompts[i], add_special_tokens=False).input_ids
        prompt_length = len(prompt_token_ids)

        current_input_ids = model_inputs["input_ids"][i]
        current_labels = list(current_input_ids)

        for j in range(len(current_labels)):
            if j < prompt_length:
                current_labels[j] = -100

        labels.append(current_labels)

    model_inputs["labels"] = labels
    model_inputs["weight"] = [float(w) for w in batch["weight"]]
    return model_inputs


if config.train_on_prompts:
    preprocess = preprocess_prompt_and_completion
else:
    preprocess = preprocess_for_completion_only

logger.info("Loading SFT training dataset …")
raw_dataset = load_dataset(
    "json",
    data_files={"train": TRAIN_PATH},
    cache_dir=os.path.join(LOCAL_SCRATCH_PATH, ".cache/huggingface/datasets"),
)
logger.info(f"Loaded {len(raw_dataset['train'])} examples from {TRAIN_PATH}")


def _within_max_len(example):
    text = (
            SFT_SYSTEM_PROMPT
            + task_to_prompt(ARCTask.from_dict(example["task_json"]))
            + SFT_IN_BETWEEN_PROMPT
            + example["solution"]
            + tok.eos_token
    )
    return len(tok.encode(text)) <= config.max_seq_len


orig_train_len = len(raw_dataset["train"])
raw_dataset["train"] = raw_dataset["train"].filter(
    _within_max_len,
    num_proc=max(1, config.cpus - 1 if config.cpus > 1 else config.cpus),
)
logger.info(
    f"Filtered training dataset to {len(raw_dataset['train'])}/{orig_train_len} examples with max_seq_len={config.max_seq_len}"
)

# Split into training and validation sets
if config.task_validation_fraction > 0 or config.example_validation_fraction > 0:
    rng = random.Random(config.seed or 42)
    task_to_indices: dict[str, list[int]] = {}
    for idx, ex in enumerate(raw_dataset["train"]):
        task_to_indices.setdefault(ex["task_name"], []).append(idx)

    all_tasks = list(task_to_indices.keys())
    rng.shuffle(all_tasks)
    val_task_count = int(len(all_tasks) * config.task_validation_fraction)
    val_tasks = set(all_tasks[:val_task_count])

    train_indices: list[int] = []
    val_indices: list[int] = []
    for task, indices in task_to_indices.items():
        indices = list(indices)
        rng.shuffle(indices)
        if task in val_tasks:
            val_indices.extend(indices)
        else:
            n_val = int(len(indices) * config.example_validation_fraction)
            val_indices.extend(indices[:n_val])
            train_indices.extend(indices[n_val:])

    logger.info(
        f"Split {len(all_tasks)} tasks with {len(val_tasks)} held-out tasks and {len(val_indices)} validation examples "
        f"(task fraction={config.task_validation_fraction}, example fraction={config.example_validation_fraction})"
    )

    train_ds = renormalize_task_weights(raw_dataset["train"].select(train_indices))
    val_ds = renormalize_task_weights(raw_dataset["train"].select(val_indices))
    dataset = DatasetDict({"train": train_ds, "validation": val_ds})
else:
    # Use static validation dataset as validation split, no test evaluation
    logger.info(
        f"No validation split configured: using static validation dataset from {VAL_PATH} and skipping test evaluation")
    train_ds = renormalize_task_weights(raw_dataset["train"])
    raw_val = load_dataset(
        "json",
        data_files={"validation": VAL_PATH},
        cache_dir=os.path.join(LOCAL_SCRATCH_PATH, ".cache/huggingface/datasets"),
    )

    orig_val_len = len(raw_val["validation"])
    raw_val["validation"] = raw_val["validation"].filter(
        _within_max_len,
        num_proc=max(1, config.cpus - 1 if config.cpus > 1 else config.cpus),
    )
    logger.info(
        f"Filtered validation dataset to {len(raw_val['validation'])}/{orig_val_len} examples with max_seq_len={config.max_seq_len}")

    val_ds = renormalize_task_weights(raw_val["validation"])
    dataset = DatasetDict({"train": train_ds, "validation": val_ds})

dataset["train"] = dataset["train"].shuffle(seed=config.seed or 42)

# Tokenize datasets
tokenized_datasets = dataset.map(
    preprocess,
    batched=True,
    remove_columns=[c for c in dataset["train"].column_names if c != "weight"],
    num_proc=max(1, config.cpus - 1 if config.cpus > 1 else config.cpus),
)
logger.info("Tokenization complete.")
# Count unique tasks per split (weight sum = number of tasks)
train_weight_sum = sum(tokenized_datasets["train"]["weight"])
val_weight_sum = sum(tokenized_datasets["validation"]["weight"])
train_task_count = int(round(train_weight_sum))
val_task_count = int(round(val_weight_sum))
logger.info(f"Number of unique tasks — train: {train_task_count}, validation: {val_task_count}")


# ------------------- trainer subclass -------------------
class WeightedTrainer(Trainer):
    def compute_loss(
            self,
            model,
            inputs,
            return_outputs: bool = False,
            num_items_in_batch=None,
    ):
        """
        Compute weighted loss for a batch based on per-example weights.
        Overrides Trainer.compute_loss to apply per-sequence weighting.
        """
        weights = inputs.pop("weight")
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        loss_fct = torch.nn.CrossEntropyLoss(
            reduction="none", label_smoothing=self.args.label_smoothing_factor
        )
        per_token = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
        ).view(shift_labels.size())
        active = shift_labels != -100
        per_seq = (per_token * active).sum(dim=1) / active.sum(dim=1).clamp_min(1)
        weight = weights.to(per_seq.device)
        weighted_loss = (per_seq * weight).sum() / weight.sum().clamp_min(1e-8)
        if return_outputs:
            return weighted_loss, outputs
        return weighted_loss

    def compute_weighted_loss(self, dataset):
        """
        Compute weighted average loss over the entire dataset, normalized by total task weight.
        """
        loader = self.get_eval_dataloader(dataset)
        all_losses: list[torch.Tensor] = []
        all_weights: list[torch.Tensor] = []
        for inputs in loader:
            w = inputs.pop("weight")
            with torch.no_grad():
                outputs = self.model(**inputs)
            logits, labels = outputs.logits, inputs["labels"]
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = torch.nn.CrossEntropyLoss(
                reduction="none", label_smoothing=self.args.label_smoothing_factor
            )
            per_token = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
            ).view(shift_labels.size())
            active = shift_labels != -100
            per_seq = (per_token * active).sum(dim=1) / active.sum(dim=1).clamp_min(1)
            all_losses.append(per_seq)
            all_weights.append(w.to(per_seq.device))
        all_losses = torch.cat(all_losses)
        all_weights = torch.cat(all_weights)
        denom = all_weights.sum().clamp_min(1e-8)
        return (all_losses * all_weights).sum() / denom

    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval", **kwargs):
        """Override evaluate to include qualitative code generation logs and exact loss normalization."""
        # Run standard evaluation (e.g., logging generations)
        metrics = super().evaluate(
            eval_dataset=eval_dataset,
            ignore_keys=ignore_keys,
            metric_key_prefix=metric_key_prefix,
            **kwargs,
        )
        # Recompute exact loss per task for eval/test to normalize by total tasks
        if metric_key_prefix in ("eval", "test"):
            ds = eval_dataset if eval_dataset is not None else self.eval_dataset
            metrics[f"{metric_key_prefix}_loss"] = self.compute_weighted_loss(ds).item()

        return metrics


# ------------------- training args -------------------
args = TrainingArguments(
    output_dir=OUT_DIR,
    per_device_train_batch_size=config.per_device_train_batch_size,
    per_device_eval_batch_size=config.per_device_eval_batch_size,
    gradient_accumulation_steps=config.gradient_accumulation_steps,
    learning_rate=config.learning_rate,
    num_train_epochs=config.num_train_epochs,
    warmup_ratio=config.warmup_ratio,
    lr_scheduler_type=config.lr_scheduler_type,
    logging_steps=config.logging_steps,
    eval_strategy="steps",
    eval_steps=config.eval_steps,
    save_strategy="steps",
    save_steps=config.save_steps,
    save_total_limit=config.save_total_limit,
    bf16=config.use_bf16,
    fp16=not config.use_bf16,
    gradient_checkpointing=config.gradient_checkpointing,
    run_name=RUN_NAME,
    report_to=config.report_to,
    remove_unused_columns=False,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    weight_decay=config.weight_decay,
    max_grad_norm=config.max_grad_norm,
    label_smoothing_factor=config.label_smoothing_factor,
)

# ------------------- wandb (lightweight) -------------------
if config.report_to == "wandb":
    os.environ["WANDB_SILENT"] = "true"
    os.environ["WANDB_CONSOLE"] = "off"
    wandb.init(
        project=config.wandb_project,
        entity=config.wandb_entity,
        name=args.run_name,
        config={
            "lr": args.learning_rate,
            "batch_size": args.per_device_train_batch_size * args.gradient_accumulation_steps,
            "epochs": args.num_train_epochs,
            "lora_r": config.lora_rank,
            "lora_alpha": config.lora_alpha,
            "train_samples": len(tokenized_datasets["train"]),
            "val_samples": len(tokenized_datasets["validation"]),
            "max_seq_len": config.max_seq_len,
            "train_tasks": train_task_count,
            "val_tasks": val_task_count,
        },
    )

    # Log full configuration as a W&B Table to preserve parameters in a structured format
    cfg = asdict(config)
    cfg_table = wandb.Table(columns=["parameter", "value"])
    for key, val in cfg.items():
        cfg_table.add_data(key, str(val))
    wandb.log({"Config": cfg_table})

# ------------------- train / eval -------------------
trainer = WeightedTrainer(
    model=model,
    args=args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    processing_class=tok,
    data_collator=WeightedCollator(tokenizer=tok),
)

logger.info("⇢ Starting training …")
trainer.train()
trainer.save_model(OUT_DIR)  # saves LoRA adapter only
tok.save_pretrained(OUT_DIR)
mets = trainer.evaluate()
trainer.save_metrics("eval", mets)
logger.info(f"✓ Finished — final loss {mets.get('eval_loss', 0.0):.4f}")

# Finalize wandb run after all evaluations
if config.report_to == "wandb":
    wandb.finish()
