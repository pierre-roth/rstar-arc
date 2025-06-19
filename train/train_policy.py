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

import json
import logging
import math
import os
import random
import re
import sys
from dataclasses import asdict
from typing import Any

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
    CODE_PREFIX,
    STEP_END,
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
from rstar_deepthink.tools.python_tool import remove_markers, run_examples
from train.data_utils import get_code_length

logger = logging.getLogger(__name__)

# ------------------- config -------------------
config = Config()
set_seed(config.seed or 42)
setup_logging(config.numeric_log_level)
logger.info(f"Using model: {config.policy_model}")

TRAIN_PATH = os.path.join(NET_SCRATCH_PATH, "sft_data", f"round_{config.round_number}", "policy_dataset_training.jsonl")
VAL_PATH = os.path.join(NET_SCRATCH_PATH, "sft_data", f"round_{config.round_number}", "policy_dataset_validation.jsonl")

if not config.full_finetune:
    dir_name = (
        f"ft-{config.policy_model.split('/')[-1]}-"
        f"{config.max_seq_len}-{config.learning_rate}-"
        f"{config.lora_rank}-{config.lora_alpha}-"
        f"{config.lora_dropout}-"
        f"{config.weight_decay}-{config.curriculum_learning}-"
        f"{config.train_on_prompts}"
    )
else:
    dir_name = (
        f"ft-{config.policy_model.split('/')[-1]}-"
        f"{config.max_seq_len}-{config.learning_rate}-"
        f"{config.weight_decay}-{config.curriculum_learning}-"
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
)

# Resize embeddings if we added new tokens
if added_tokens > 0:
    model.resize_token_embeddings(len(tok))

# ensure pad_token_id is set on the model to avoid fallback warning during generation
if model.config.pad_token_id is None:
    model.config.pad_token_id = model.config.eos_token_id
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
    num_proc=max(1, os.cpu_count() // 2 if os.cpu_count() else 1),
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
        num_proc=max(1, os.cpu_count() // 2 if os.cpu_count() else 1),
    )
    logger.info(
        f"Filtered validation dataset to {len(raw_val['validation'])}/{orig_val_len} examples with max_seq_len={config.max_seq_len}")

    val_ds = renormalize_task_weights(raw_val["validation"])
    dataset = DatasetDict({"train": train_ds, "validation": val_ds})

# Shuffle or sort training split
if config.curriculum_learning:
    logger.info("Curriculum learning enabled: sorting training examples by code length")
    dataset["train"] = dataset["train"].map(lambda ex: {"code_length": get_code_length(ex["solution"])})
    dataset["train"] = dataset["train"].sort("code_length")
    dataset["train"] = dataset["train"].remove_columns(["code_length"])
else:
    dataset["train"] = dataset["train"].shuffle(seed=config.seed or 42)

# Tokenize datasets
tokenized_datasets = dataset.map(
    preprocess,
    batched=True,
    remove_columns=[c for c in dataset["train"].column_names if c != "weight"],
    num_proc=max(1, os.cpu_count() // 2 if os.cpu_count() else 1),
)
logger.info("Tokenization complete.")
# Count unique tasks per split (weight sum = number of tasks)
train_weight_sum = sum(tokenized_datasets["train"]["weight"])
val_weight_sum = sum(tokenized_datasets["validation"]["weight"])
train_task_count = int(round(train_weight_sum))
val_task_count = int(round(val_weight_sum))
logger.info(f"Number of unique tasks — train: {train_task_count}, validation: {val_task_count}")

# ------------------- qualitative logging helper -------------------
_QUAL_TABLE_COLUMNS = [
    "split",
    "task_name",
    "temperature",
    "pass@k",
    "pass_rate",
    "generations",
]

_eval_counter = 0


def _log_task_generations(
        trainer: Any,
        split: str,
        indices: list[int],
        ds: Any,
        summary: wandb.Table,
) -> None:
    """Generate code for sampled tasks, log qualitative results and per-token perplexity."""
    for idx in indices:
        row = ds[idx]
        task = ARCTask.from_dict(row["task_json"])
        prompt_body = task_to_prompt(task)
        prompt = SFT_SYSTEM_PROMPT + prompt_body + SFT_IN_BETWEEN_PROMPT + CODE_PREFIX
        for temp in config.eval_temperatures:
            gen_entries: list[dict[str, Any]] = []
            pass_count = 0
            for _ in range(config.pass_k):
                inputs = tok(prompt, return_tensors="pt")
                inputs = {k: v.to(trainer.model.device) for k, v in inputs.items()}
                gen_ids = trainer.model.generate(
                    **inputs,
                    do_sample=True,
                    temperature=temp,
                    top_p=config.top_p,
                    top_k=config.top_k if config.top_k > 0 else None,
                    max_new_tokens=config.max_tokens,
                    repetition_penalty=config.repetition_penalty,
                    eos_token_id=EOS_TOKEN_IDS,
                )
                gen_text = tok.decode(gen_ids[0], skip_special_tokens=False)
                raw_steps = re.split(f"{STEP_END}", gen_text)
                num_steps = len(raw_steps)
                format_adherence = CODE_END in gen_text and num_steps >= config.min_steps_for_format_adherence
                prefix_errors: list[bool] = []
                for k_i in range(1, num_steps + 1):
                    code_str = remove_markers("".join(raw_steps[:k_i]))
                    err, _, _ = run_examples(task, code_str)
                    prefix_errors.append(err)
                err_full, passed_train_full, results_full = run_examples(
                    task, remove_markers(gen_text)
                )
                n_train = len(task.training_examples)
                test_results = results_full[n_train:]
                expected_test = [ex.output_grid.grid for ex in task.test_examples]
                passed_test = (
                        len(test_results) == len(expected_test)
                        and all(tr == et for tr, et in zip(test_results, expected_test))
                )
                if not format_adherence:
                    category = "formatting"
                elif any(prefix_errors):
                    category = "runtime"
                elif not (passed_train_full and passed_test):
                    category = "semantics"
                else:
                    category = "none"
                gen_entries.append({
                    "num_steps": num_steps,
                    "passed_train": passed_train_full,
                    "passed_test": passed_test,
                    "error": err_full,
                    "category": category,
                    "generation": gen_text,
                })
                if passed_train_full and passed_test:
                    pass_count += 1

            pass_rate = pass_count / config.pass_k if config.pass_k > 0 else 0.0
            pass_at_k = pass_count > 0

            try:
                gen_entries_json = json.dumps(gen_entries, indent=2)  # indent for readability if viewed raw
            except TypeError as e:
                logger.error(f"Could not serialize gen_entries to JSON for task {row.get('task_name', None)}: {e}")
                gen_entries_json = str(gen_entries)  # Fallback to plain string representation

            summary.add_data(
                split,
                row.get("task_name", None),
                temp,
                pass_at_k,
                pass_rate,
                gen_entries_json,
            )
            # Per-token perplexity analysis
            if config.perplexity_window_size is not None:
                with torch.no_grad():
                    seq = gen_ids[0]
                    input_len = inputs["input_ids"].shape[1]
                    outputs = trainer.model(seq.unsqueeze(0)).logits
                    shift_logits = outputs[..., :-1, :].contiguous()
                    shift_labels = seq[1:].contiguous()
                    loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
                    per_token_loss = loss_fct(
                        shift_logits.view(-1, shift_logits.size(-1)),
                        shift_labels.view(-1),
                    ).view(shift_labels.size())
                    gen_loss = per_token_loss[input_len - 1:]
                    perp = torch.exp(gen_loss).cpu().tolist()
                if config.perplexity_window_size and config.perplexity_window_size > 1:
                    window = config.perplexity_window_size
                    smoothed = [
                        sum(perp[max(0, i - window + 1): i + 1]) / min(i + 1, window)
                        for i in range(len(perp))
                    ]
                else:
                    smoothed = perp
                perp_table = wandb.Table(columns=["token_index", "perplexity", "windowed_perplexity"])
                for i, (p, w) in enumerate(zip(perp, smoothed)):
                    perp_table.add_data(i, p, w)
                wandb.log(
                    {f"perplexity/{row.get('task_name', None)}/{temp}": perp_table},
                    step=trainer.state.global_step,
                )


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
            metrics[f"{metric_key_prefix}_perplexity"] = math.exp(metrics[f"{metric_key_prefix}_loss"])

        if config.report_to == "wandb" and metric_key_prefix == "eval":
            try:
                global _eval_counter
                _eval_counter += 1
                table_name = f"evaluation_{_eval_counter}"
                rng = random.Random(config.seed or 42)

                ds_train = dataset["train"]
                total_train = len(ds_train)
                num_train = min(config.num_training_samples, total_train)
                name_to_train: dict[str, list[int]] = {}
                for i, row in enumerate(ds_train):
                    name_to_train.setdefault(row.get("task_name"), []).append(i)
                unique_train = list(name_to_train.keys())
                rng.shuffle(unique_train)
                train_indices = [rng.choice(name_to_train[name]) for name in unique_train[:num_train]]

                ds_val = dataset["validation"]
                total_val = len(ds_val)
                num_val = min(config.num_validation_samples, total_val)
                name_to_val: dict[str, list[int]] = {}
                for i, row in enumerate(ds_val):
                    name_to_val.setdefault(row.get("task_name"), []).append(i)
                unique_val = list(name_to_val.keys())
                rng.shuffle(unique_val)
                val_indices = [rng.choice(name_to_val[name]) for name in unique_val[:num_val]]

                summary = wandb.Table(columns=_QUAL_TABLE_COLUMNS)
                _log_task_generations(self, "train", train_indices, ds_train, summary)
                _log_task_generations(self, "validation", val_indices, ds_val, summary)
                wandb.log({table_name: summary}, step=self.state.global_step)
            except Exception as e:
                logger.warning(f"Qualitative eval logging failed: {e}")
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
metrics = trainer.evaluate()
trainer.save_metrics("eval", metrics)
logger.info(
    f"✓ Finished — final loss {metrics.get('eval_loss', 0.0):.4f}, perplexity {metrics.get('eval_perplexity', 0.0):.4f}")

if config.report_to == "wandb":
    wandb.log({
        "final_eval/loss": metrics.get("eval_loss"),
        "final_eval/perplexity": metrics.get("eval_perplexity"),
    })

# ------------------- final test evaluation -------------------
if config.task_validation_fraction > 0 or config.example_validation_fraction > 0:
    logger.info("Loading SFT test dataset …")
    raw_test = load_dataset(
        "json",
        data_files={"test": VAL_PATH},
        cache_dir=os.path.join(LOCAL_SCRATCH_PATH, ".cache/huggingface/datasets"),
    )
    logger.info(f"Loaded {len(raw_test['test'])} examples for test evaluation from {VAL_PATH}")

    orig_test_len = len(raw_test["test"])
    raw_test["test"] = raw_test["test"].filter(
        _within_max_len,
        num_proc=max(1, os.cpu_count() // 2 if os.cpu_count() else 1),
    )
    logger.info(
        f"Filtered test dataset to {len(raw_test['test'])}/{orig_test_len} examples with max_seq_len={config.max_seq_len}")

    # Tokenize test dataset
    tokenized_test = raw_test["test"].map(
        preprocess,
        batched=True,
        remove_columns=[c for c in raw_test["test"].column_names if c != "weight"],
        num_proc=max(1, os.cpu_count() // 2 if os.cpu_count() else 1),
    )
    logger.info("Running final evaluation on test set …")
    # Count unique tasks in test split (weight sum = number of tasks)
    test_task_count = int(sum(tokenized_test["weight"]))
    logger.info(f"Number of unique tasks in test set: {test_task_count}")
    # Use test prefix for metrics
    test_metrics = trainer.evaluate(eval_dataset=tokenized_test, metric_key_prefix="test")
    trainer.save_metrics("test", test_metrics)
    logger.info(
        f"Test set evaluation complete — loss {test_metrics.get('test_loss', 0.0):.4f}, perplexity {test_metrics.get('test_perplexity', 0.0):.4f}")
    if config.report_to == "wandb":
        # Log quantitative test loss, perplexity and task count
        wandb.log({
            "test/loss": test_metrics.get("test_loss"),
            "test/perplexity": test_metrics.get("test_perplexity"),
            "test_tasks": test_task_count,
        })
    # Qualitative evaluation on test set: sample and log code generation and correctness
    try:
        ds_test = raw_test["test"]
        total_test = len(ds_test)
        num_test = min(config.num_validation_samples, total_test)
        rng_test = random.Random(config.seed or 42)
        test_indices = rng_test.sample(range(total_test), num_test)
        table_test = wandb.Table(columns=_QUAL_TABLE_COLUMNS)
        _log_task_generations(trainer, "test", test_indices, ds_test, table_test)
        wandb.log({"qualitative_test": table_test})
    except Exception as e:
        logger.warning(f"Qualitative test evaluation failed: {e}")
else:
    logger.info("Skipping final test evaluation because no validation split was configured")
# Finalize wandb run after all evaluations
if config.report_to == "wandb":
    wandb.finish()
