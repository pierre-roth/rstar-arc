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
import sys
from typing import Any

import torch
import wandb
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    set_seed
)

# ─────────────────── project imports ───────────────────
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from constants import (
    SFT_SYSTEM_PROMPT, SFT_IN_BETWEEN_PROMPT, LOCAL_SCRATCH_PATH, NET_SCRATCH_PATH
)
from utils import setup_logging
from rstar_deepthink import Config
from rstar_deepthink.arc_task import ARCTask
from rstar_deepthink.arc_task.task_utils import task_to_prompt

logger = logging.getLogger(__name__)

# ─────────────────── config ───────────────────
config = Config()
set_seed(config.seed or 42)
setup_logging(config.numeric_log_level)
logger.info("Using model: %s", config.policy_model)

TRAIN_PATH = os.path.join(NET_SCRATCH_PATH, "sft_data", f"round_{config.round_number}", "policy_dataset_training.jsonl")
VAL_PATH = os.path.join(NET_SCRATCH_PATH, "sft_data", f"round_{config.round_number}", "policy_dataset_validation.jsonl")
OUT_DIR = os.path.join(NET_SCRATCH_PATH, "models", "fine_tuned", "policy",
                       f"ft-{config.policy_model.split('/')[-1]}-{config.max_seq_len}-{config.learning_rate}-{config.lora_rank}-{config.lora_alpha}")
os.makedirs(OUT_DIR, exist_ok=True)

# ─────────────────── tokenizer ───────────────────
tok = AutoTokenizer.from_pretrained(config.policy_model, trust_remote_code=True)
tok.pad_token = tok.pad_token or tok.eos_token
tok.padding_side = "right"

# ─────────────────── LoRA config ───────────────────
lora_config = LoraConfig(
    r=config.lora_rank,
    lora_alpha=config.lora_alpha,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    rank_pattern={
        r"(q_proj|k_proj|v_proj|o_proj)$": config.lora_rank // 2,
        r"(gate_proj|up_proj|down_proj)$": config.lora_rank,
    },
    alpha_pattern={
        r"(q_proj|k_proj|v_proj|o_proj)$": config.lora_alpha // 2,
        r"(gate_proj|up_proj|down_proj)$": config.lora_alpha,
    },
    lora_dropout=config.lora_dropout,
    bias="none",
    task_type="CAUSAL_LM",
)

logger.info("LoRA: r=%d α=%d dropout=%.2f", lora_config.r, lora_config.lora_alpha, lora_config.lora_dropout)

# ─────────────────── model ───────────────────
model = AutoModelForCausalLM.from_pretrained(
    config.policy_model,
    torch_dtype=torch.bfloat16 if config.use_bf16 else torch.float16,
    trust_remote_code=True,
)
# Multi-GPU training is handled via Accelerate launch
model.config.use_cache = False
model.gradient_checkpointing_enable()  # save memory

model.enable_input_require_grads()

model = get_peft_model(model, lora_config)  # add adapters

# log trainable params
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total = sum(p.numel() for p in model.parameters())
logger.info("Trainable params: %.1f M / %.1f M (%.2f%%)", trainable / 1e6, total / 1e6, 100 * trainable / total)


# ─────────────────── preprocessing ───────────────────
def preprocess(batch):
    texts = [
        SFT_SYSTEM_PROMPT +
        task_to_prompt(ARCTask.from_dict(j)) +
        SFT_IN_BETWEEN_PROMPT + sol + tok.eos_token
        for j, sol in zip(batch["task_json"], batch["solution"])
    ]
    model_inp = tok(
        texts, max_length=config.max_seq_len,
        truncation=True, padding="max_length"
    )
    model_inp["labels"] = model_inp["input_ids"].copy()
    model_inp["weight"] = [float(w) for w in batch["weight"]]
    return model_inp


logger.info("Loading SFT dataset …")
dataset = load_dataset(
    "json",
    data_files={"train": TRAIN_PATH, "validation": VAL_PATH},
    cache_dir=os.path.join(LOCAL_SCRATCH_PATH, ".cache/huggingface/datasets"),
)
dataset["train"] = dataset["train"].shuffle(seed=42)
tok_ds = dataset.map(
    preprocess, batched=True,
    remove_columns=[c for c in dataset["train"].column_names if c != "weight"],
    num_proc=max(1, os.cpu_count() // 2),
)
logger.info("Dataset ready.")


# ─────────────────── data collator ───────────────────
class WeightedCollator(DataCollatorForLanguageModeling):
    def __call__(self, ex, **k) -> dict[str, Any]:
        w = torch.tensor([e["weight"] for e in ex], dtype=torch.float32)
        ex_wo = [{k2: v for k2, v in e.items() if k2 != "weight"} for e in ex]
        batch = super().__call__(ex_wo)  # mlm=False inherited
        batch["weight"] = w
        return batch


# ─────────────────── trainer subclass ───────────────────
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


# ─────────────────── training args ───────────────────
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
    run_name=f"ft-policy-lora-{config.reward_model.split('/')[-1]}-{config.max_seq_len}-{config.learning_rate}-{config.lora_rank}-{config.lora_alpha}",
    report_to=config.report_to,
    remove_unused_columns=False,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False
)

# ─────────────────── wandb (lightweight) ───────────────────
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
            "train_samples": len(tok_ds["train"]),
            "val_samples": len(tok_ds["validation"]),
            "max_seq_len": config.max_seq_len,
        },
    )

# ─────────────────── train / eval ───────────────────
trainer = WeightedTrainer(
    model=model,
    args=args,
    train_dataset=tok_ds["train"],
    eval_dataset=tok_ds["validation"],
    tokenizer=tok,
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
    wandb.finish()
