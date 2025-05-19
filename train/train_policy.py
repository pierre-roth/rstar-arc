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
import re
import sys
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
    DataCollatorForLanguageModeling,
    set_seed
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
)
from utils import setup_logging
from train_utils import maybe_peft_wrap
from rstar_deepthink import Config
from rstar_deepthink.arc_task import ARCTask
from rstar_deepthink.arc_task.task_utils import task_to_prompt
from rstar_deepthink.tools.python_tool import remove_markers, run_examples

logger = logging.getLogger(__name__)

# ------------------- config -------------------
config = Config()
set_seed(config.seed or 42)
setup_logging(config.numeric_log_level)
logger.info("Using model: %s", config.policy_model)

TRAIN_PATH = os.path.join(NET_SCRATCH_PATH, "sft_data", f"round_{config.round_number}", "policy_dataset_training.jsonl")
VAL_PATH = os.path.join(NET_SCRATCH_PATH, "sft_data", f"round_{config.round_number}", "policy_dataset_validation.jsonl")

if not config.full_finetune:
    dir_name = (
        f"ft-{config.policy_model.split('/')[-1]}-"
        f"{config.max_seq_len}-{config.learning_rate}-"
        f"{config.lora_rank}-{config.lora_alpha}"
    )
else:
    dir_name = (
        f"ft-{config.policy_model.split('/')[-1]}-"
        f"{config.max_seq_len}-{config.learning_rate}"
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

# ------------------- model -------------------
model = AutoModelForCausalLM.from_pretrained(
    config.policy_model,
    torch_dtype=torch.bfloat16 if config.use_bf16 else torch.float16,
    trust_remote_code=True,
)
# Multi-GPU training is handled via Accelerate launch
model.config.use_cache = False
model.gradient_checkpointing_enable()  # save memory

model.enable_input_require_grads()

# Wrap model with LoRA adapters or full-model fine-tuning as configured
model = maybe_peft_wrap(model, config)

# log trainable params
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total = sum(p.numel() for p in model.parameters())
logger.info("Trainable params: %.1f M / %.1f M (%.2f%%)", trainable / 1e6, total / 1e6, 100 * trainable / total)


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
        padding=False
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


logger.info("Loading SFT training dataset …")
raw_dataset = load_dataset(
    "json",
    data_files={"train": TRAIN_PATH},
    cache_dir=os.path.join(LOCAL_SCRATCH_PATH, ".cache/huggingface/datasets"),
)
logger.info(f"Loaded {len(raw_dataset['train'])} examples from {TRAIN_PATH}")

# Split into training and validation sets
if config.validation_fraction > 0:
    # Split by task names to create a validation set
    task_names = list({ex["task_name"] for ex in raw_dataset["train"]})
    rng = random.Random(config.seed or 42)
    rng.shuffle(task_names)
    val_count = int(len(task_names) * config.validation_fraction)
    val_tasks = set(task_names[:val_count])
    train_tasks = set(task_names[val_count:])
    logger.info(
        f"Splitting {len(task_names)} tasks into {len(train_tasks)} train and {len(val_tasks)} validation tasks "
        f"(validation fraction={config.validation_fraction})"
    )
    # Filter examples by task membership
    train_ds = raw_dataset["train"].filter(lambda ex: ex["task_name"] in train_tasks)
    val_ds = raw_dataset["train"].filter(lambda ex: ex["task_name"] in val_tasks)
    dataset = DatasetDict({"train": train_ds, "validation": val_ds})
else:
    # Use static validation dataset as validation split, no test evaluation
    logger.info(
        "validation_fraction is 0: using static validation dataset from %s and skipping test evaluation",
        VAL_PATH,
    )
    # Training set is full training dataset
    train_ds = raw_dataset["train"]
    # Load static validation dataset
    raw_val = load_dataset(
        "json",
        data_files={"validation": VAL_PATH},
        cache_dir=os.path.join(LOCAL_SCRATCH_PATH, ".cache/huggingface/datasets"),
    )
    val_ds = raw_val["validation"]
    dataset = DatasetDict({"train": train_ds, "validation": val_ds})

# Shuffle training split
dataset["train"] = dataset["train"].shuffle(seed=config.seed or 42)

# Tokenize datasets
tokenized_datasets = dataset.map(
    preprocess_for_completion_only,
    batched=True,
    remove_columns=[c for c in dataset["train"].column_names if c != "weight"],
    num_proc=max(1, os.cpu_count() // 2 if os.cpu_count() else 1),
)
logger.info("Tokenization complete.")


# ------------------- data collator -------------------
class WeightedCollator(DataCollatorForLanguageModeling):
    def __call__(self, ex, **k) -> dict[str, Any]:
        w = torch.tensor([e["weight"] for e in ex], dtype=torch.float32)
        ex_wo = [{k2: v for k2, v in e.items() if k2 != "weight"} for e in ex]
        batch = super().__call__(ex_wo)  # mlm=False inherited
        batch["weight"] = w
        return batch


# ------------------- trainer subclass -------------------
class WeightedTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **_):
        w = inputs.pop("weight")  # (B,)
        outputs = model(**inputs)
        logits, labels = outputs.logits, inputs["labels"]

        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
        per_token = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
        ).view(shift_labels.size())

        active = shift_labels != -100
        per_seq = (per_token * active).sum(dim=1) / active.sum(dim=1).clamp_min(1)
        loss = (per_seq * w.to(per_seq.device)).mean()

        return (loss, outputs) if return_outputs else loss

    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval", **kwargs):
        """Override evaluate to include qualitative code generation logs."""
        # Run standard evaluation
        metrics = super().evaluate(
            eval_dataset=eval_dataset,
            ignore_keys=ignore_keys,
            metric_key_prefix=metric_key_prefix,
            **kwargs,
        )
        if config.report_to == "wandb" and metric_key_prefix == "eval":
            # Comprehensive evaluation: sample validation and optional training prompts, generate multiple variants,
            # execute step prefixes, check full solution correctness, classify errors, and log to WandB with split info.
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

            summary = wandb.Table(
                columns=[
                    "split",
                    "task_name",
                    "temperature",
                    "num_steps",
                    "error_flag",
                    "passed_train",
                    "passed_test",
                    "error_category",
                    "prompt_and_generation",
                ]
            )

            def _log(split: str, indices: list[int], ds: Any):
                for idx in indices:
                    row = ds[idx]
                    task = ARCTask.from_dict(row["task_json"])
                    prompt_body = task_to_prompt(task)
                    prompt = SFT_SYSTEM_PROMPT + prompt_body + SFT_IN_BETWEEN_PROMPT + CODE_PREFIX
                    for temp in config.eval_temperatures:
                        inputs = tok(prompt, return_tensors="pt")
                        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
                        gen_ids = self.model.generate(
                            **inputs,
                            do_sample=True,
                            temperature=temp,
                            top_p=config.top_p,
                            top_k=config.top_k if config.top_k > 0 else None,
                            max_new_tokens=config.max_tokens,
                        )
                        gen_text = tok.decode(gen_ids[0], skip_special_tokens=True)
                        raw_steps = re.split(f"{STEP_END}", gen_text)
                        num_steps = len(raw_steps)
                        format_adherence = num_steps >= config.min_steps_for_format_adherence
                        prefix_errors = []
                        for k in range(1, num_steps + 1):
                            code_str = remove_markers("".join(raw_steps[:k]))
                            err, _, _ = run_examples(task, code_str)
                            prefix_errors.append(err)
                        err_full, passed_train_full, results_full = run_examples(task, remove_markers(gen_text))
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
                        summary.add_data(
                            split,
                            row.get("task_name", None),
                            temp,
                            num_steps,
                            err_full,
                            passed_train_full,
                            passed_test,
                            category,
                            gen_text,
                        )
                        # Per-token perplexity analysis
                        if config.perplexity_window_size is not None:
                            with torch.no_grad():
                                seq = gen_ids[0]
                                input_len = inputs["input_ids"].shape[1]
                                outputs = self.model(seq.unsqueeze(0)).logits
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
                                    sum(perp[max(0, i - window + 1): i + 1]) / (min(i + 1, window))
                                    for i in range(len(perp))
                                ]
                            else:
                                smoothed = perp
                            perp_table = wandb.Table(
                                columns=["token_index", "perplexity", "windowed_perplexity"]
                            )
                            for i, (p, w) in enumerate(zip(perp, smoothed)):
                                perp_table.add_data(i, p, w)
                            wandb.log(
                                {f"perplexity/{row.get('task_name', None)}/{temp}": perp_table},
                                step=self.state.global_step,
                            )
            if config.num_training_samples > 0:
                _log("train", train_indices, ds_train)
            if config.num_validation_samples > 0:
                _log("validation", val_indices, ds_val)
            wandb.log({"eval_summary": summary}, step=self.state.global_step)
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
    weight_decay=config.weight_decay
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
        },
    )

# ------------------- train / eval -------------------
trainer = WeightedTrainer(
    model=model,
    args=args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    processing_class=tok,
    data_collator=WeightedCollator(tokenizer=tok, mlm=False),
)

logger.info("⇢ Starting training …")
trainer.train()
trainer.save_model(OUT_DIR)  # saves LoRA adapter only
metrics = trainer.evaluate()
trainer.save_metrics("eval", metrics)
logger.info("✓ Finished — final loss %.4f", metrics.get("eval_loss", 0.0))

if config.report_to == "wandb":
    wandb.log({"final_eval/loss": metrics.get("eval_loss")})

if config.validation_fraction > 0:
    # ------------------- final test evaluation -------------------
    logger.info("Loading SFT test dataset …")
    raw_test = load_dataset(
        "json",
        data_files={"test": VAL_PATH},
        cache_dir=os.path.join(LOCAL_SCRATCH_PATH, ".cache/huggingface/datasets"),
    )
    logger.info(f"Loaded {len(raw_test['test'])} examples for test evaluation from {VAL_PATH}")
    # Tokenize test dataset
    tokenized_test = raw_test["test"].map(
        preprocess_for_completion_only,
        batched=True,
        remove_columns=[c for c in raw_test["test"].column_names if c != "weight"],
        num_proc=max(1, os.cpu_count() // 2 if os.cpu_count() else 1),
    )
    logger.info("Running final evaluation on test set …")
    # Use test prefix for metrics
    test_metrics = trainer.evaluate(eval_dataset=tokenized_test, metric_key_prefix="test")
    trainer.save_metrics("test", test_metrics)
    logger.info("Test set evaluation complete — loss %.4f", test_metrics.get("test_loss", 0.0))
    if config.report_to == "wandb":
        # Log quantitative test loss
        wandb.log({"test/loss": test_metrics.get("test_loss")})
        # Qualitative evaluation on test set: generate sample solutions
        try:
            # Sample a few test examples
            ds_test = raw_test["test"]  # original test dataset
            total_test = len(ds_test)
            num_samples_test = min(3, total_test)
            rng_test = random.Random(config.seed or 42)
            test_indices = rng_test.sample(range(total_test), num_samples_test)
            table_test = wandb.Table(columns=["task_name", "prompt_and_generation"])
            for tidx in test_indices:
                row_test = ds_test[tidx]
                task_obj = ARCTask.from_dict(row_test["task_json"])
                prompt_body = task_to_prompt(task_obj)
                # Construct full prompt: system + task + between + code prefix
                prompt = SFT_SYSTEM_PROMPT + prompt_body + SFT_IN_BETWEEN_PROMPT + CODE_PREFIX
                inputs = tok(prompt, return_tensors="pt")
                inputs = {k: v.to(model.device) for k, v in inputs.items()}
                gen_ids = model.generate(
                    **inputs,
                    do_sample=True,
                    temperature=config.policy_temperature,
                    top_p=config.top_p,
                    top_k=config.top_k if config.top_k > 0 else None,
                    max_new_tokens=config.max_tokens,
                )
                gen_text = tok.decode(gen_ids[0], skip_special_tokens=True)
                table_test.add_data(row_test.get("task_name", None), gen_text)
            wandb.log({"qualitative_test": table_test}, commit=False)
        except Exception as e:
            logger.warning(f"Qualitative test evaluation failed: {e}")
else:
    logger.info("Skipping final test evaluation because validation_fraction is 0")
# Finalize wandb run after all evaluations
if config.report_to == "wandb":
    wandb.finish()
