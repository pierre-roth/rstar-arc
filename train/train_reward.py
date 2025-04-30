"""
Train a reward model on positive–negative preference pairs.

Everything is driven by rstar_deepthink.Config:
  • model and dataset paths
  • LoRA hyper-params
  • batch-size, LR, logging, etc.
Multi-GPU training is handled via Accelerate.

Launch with:
  # 4 GPUs with torchrun
  torchrun --nproc_per_node 4 train/train_reward.py --config-file configs/train_reward.yaml
or
  # via Accelerate
  accelerate launch train/train_reward.py --config-file configs/train_reward.yaml
"""

from __future__ import annotations

import logging
import os
import sys
from functools import partial
from typing import Any, Sequence

import torch
import torch.nn.functional as f
import wandb
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    set_seed,
)

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from constants import NET_SCRATCH_PATH, SFT_SYSTEM_PROMPT, SFT_IN_BETWEEN_PROMPT
from rstar_deepthink import Config
from utils import setup_logging
from rstar_deepthink.llms.reward import RewardModelModule
from rstar_deepthink.arc_task import ARCTask
from rstar_deepthink.arc_task.task_utils import task_to_prompt

logger = logging.getLogger(__name__)

config = Config()
set_seed(config.seed or 42)
setup_logging(config.numeric_log_level)

reward_output_dir = os.path.join(
    NET_SCRATCH_PATH,
    "models",
    "fine_tuned",
    "reward",
    f"ft-{config.reward_model.split('/')[-1]}-{config.max_seq_len}-{config.learning_rate}-{config.lora_rank}-{config.lora_alpha}"
)
os.makedirs(reward_output_dir, exist_ok=True)

# Hardware overview
num_gpus = torch.cuda.device_count()
logger.info("GPUs visible: %d", num_gpus)

# Tokenizer
tok = AutoTokenizer.from_pretrained(config.reward_model, trust_remote_code=True)
tok.pad_token = tok.pad_token or tok.eos_token
tok.padding_side = "left"

# Base model + value head
dtype = torch.bfloat16 if config.use_bf16 else torch.float16
base = AutoModelForCausalLM.from_pretrained(
    config.reward_model,
    torch_dtype=dtype,
    trust_remote_code=True,
)

base.config.use_cache = False  # disable cache for gradient checkpointing

base.gradient_checkpointing_enable()  # enable gradient checkpointing for memory efficiency

base.enable_input_require_grads()  # enable input gradients for LoRA

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

base = get_peft_model(base, lora_config)

model = RewardModelModule(base, dropout=config.reward_value_head_dropout)

# get rid of sliding-window SPDA warning
# model.backbone.config.attn_config['attn_impl'] = "flash"


def preprocess_batch(
        ex: dict[str, Sequence[Any]], *, max_len: int
) -> dict[str, Sequence[Any]]:
    """
    Tokenise a batch of preference pairs.

    *No* EOS token is appended: prefix + completion is *not* a full convo.
    """
    out: dict[str, list] = {
        "chosen_input_ids": [], "chosen_attention_mask": [],
        "rejected_input_ids": [], "rejected_attention_mask": [],
        "weight": [],
    }

    for j, p, ch, rj, w in zip(
            ex["task_json"], ex["prefix"], ex["chosen"], ex["rejected"], ex["weight"]
    ):
        chosen = tok(SFT_SYSTEM_PROMPT +
                     task_to_prompt(ARCTask.from_dict(j)) +
                     SFT_IN_BETWEEN_PROMPT + p + ch, max_length=max_len, truncation=True,
                     padding=False, add_special_tokens=False)
        rejected = tok(SFT_SYSTEM_PROMPT +
                       task_to_prompt(ARCTask.from_dict(j)) +
                       SFT_IN_BETWEEN_PROMPT + p + rj, max_length=max_len, truncation=True,
                       padding=False, add_special_tokens=False)

        out["chosen_input_ids"].append(chosen["input_ids"])
        out["chosen_attention_mask"].append(chosen["attention_mask"])
        out["rejected_input_ids"].append(rejected["input_ids"])
        out["rejected_attention_mask"].append(rejected["attention_mask"])
        out["weight"].append(float(w))  # keep as float
    return out


class PairwiseCollator(DataCollatorWithPadding):
    """
    Pads chosen & rejected separately, concatenates them, **and** returns
    a `weight` vector of shape (batch,).
    """

    def __call__(self, feats: Sequence[dict[str, Any]]) -> dict[str, torch.Tensor]:
        chosen, rejected, weights = [], [], []
        for f in feats:
            chosen.append(
                {"input_ids": f["chosen_input_ids"],
                 "attention_mask": f["chosen_attention_mask"]}
            )
            rejected.append(
                {"input_ids": f["rejected_input_ids"],
                 "attention_mask": f["rejected_attention_mask"]}
            )
            weights.append(f["weight"])

        # Pad chosen and rejected separately and collate into model inputs
        batch = self.tokenizer.pad(
            chosen + rejected,
            return_tensors="pt",
            padding=self.padding
        )
        # sample-wise weights for pairwise loss (length B)
        batch["weight"] = torch.tensor(weights, dtype=torch.float32)
        # add dummy labels to ensure Trainer runs prediction and compute_metrics
        # labels will be ignored by our custom compute_loss
        batch["labels"] = batch["input_ids"].clone()
        return dict(batch)


class PairwiseTrainer(Trainer):
    """-log σ(r₊ − r₋) * sample_weight"""

    def compute_loss(self, model, inputs, return_outputs: bool = False, **_) -> torch.Tensor | tuple:
        """
        Pair-wise preference loss

        For every batch we receive the *chosen* sequence followed by the *rejected*
        sequence.  The loss is

            L = − log σ(r_chosen − r_rejected) · weight

        where σ is the logistic sigmoid and *weight* is a scalar attached to every
        pair.  When `return_outputs=True` we hand back the concatenated rewards
        (first all chosen, then all rejected) so that `Trainer` can feed them to
        `compute_metrics`.
        """
        # ------------------------------------------------------------------ data
        weights = inputs.pop("weight")  # shape (B,)
        input_ids = inputs["input_ids"]  # shape (2B, L)
        attention_mask = inputs["attention_mask"]  # shape (2B, L)
        B = input_ids.size(0) // 2

        # ------------------------------------------------------------ forward pass
        rewards = model(input_ids=input_ids,
                        attention_mask=attention_mask)  # shape (2B,)
        chosen, rejected = rewards[:B], rewards[B:]  # each (B,)

        # ------------------------------------------------------------------ loss
        loss_per_pair = -f.logsigmoid(chosen - rejected)  # (B,)
        loss = (loss_per_pair * weights.to(loss_per_pair.device)).sum() / weights.sum()

        # ----------------------------------------------------------- bookkeeping
        if return_outputs:
            # Trainer expects a *single* tensor for predictions, so we stack
            # them in the same order they appeared in the batch.
            preds = torch.cat([chosen.detach(), rejected.detach()], dim=0)
            return loss, preds

        return loss


def compute_accuracy(eval_preds):
    r = torch.tensor(eval_preds.predictions)  # ensure tensor
    b = r.size(0) // 2
    acc = (r[:b] > r[b:]).float().mean().item()  # cast to float first
    return {"accuracy": acc}


# Load and tokenise dataset
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total = sum(p.numel() for p in model.parameters())
logger.info(
    f"Trainable params: {trainable / 1e6:.1f} M / {total / 1e6:.1f} M ({trainable / total * 100:.2f}%)"
)

TRAIN_PATH = os.path.join(NET_SCRATCH_PATH, "sft_data", f"round_{config.round_number}", "reward_dataset_training.jsonl")
VAL_PATH = os.path.join(NET_SCRATCH_PATH, "sft_data", f"round_{config.round_number}", "reward_dataset_validation.jsonl")
logger.info(f"Loading preference pairs from {TRAIN_PATH} and {VAL_PATH}")
raw = load_dataset("json", data_files={"train": TRAIN_PATH, "validation": VAL_PATH})

remove_cols = [c for c in ("prefix", "chosen", "rejected") if c in raw["train"].column_names]
tok_ds = raw.map(
    partial(preprocess_batch, max_len=config.max_seq_len),
    batched=True,
    remove_columns=remove_cols,
    num_proc=max(1, os.cpu_count() // 2),
)
logger.info(f"Dataset ready – {len(tok_ds['train'])} train / {len(tok_ds['validation'])} validation examples")

# TrainingArguments
args = TrainingArguments(
    output_dir=reward_output_dir,
    # batching
    per_device_train_batch_size=config.per_device_train_batch_size,
    per_device_eval_batch_size=config.per_device_eval_batch_size,
    gradient_accumulation_steps=config.gradient_accumulation_steps,
    # optimisation
    num_train_epochs=config.num_train_epochs,
    learning_rate=config.learning_rate,
    lr_scheduler_type=config.lr_scheduler_type,
    warmup_ratio=config.warmup_ratio,
    # logging / eval / save
    logging_steps=config.logging_steps,
    eval_strategy="steps",
    eval_steps=config.eval_steps,
    save_strategy="steps",
    save_steps=config.save_steps,
    save_total_limit=config.save_total_limit,
    # precision
    bf16=config.use_bf16,
    fp16=not config.use_bf16,
    # distributed: handled via Accelerate launch
    # misc
    seed=config.seed,
    run_name=f"ft-reward-lora-{config.reward_model.split('/')[-1]}-{config.max_seq_len}-{config.learning_rate}-{config.lora_rank}-{config.lora_alpha}",
    report_to=config.report_to,
    remove_unused_columns=False,
    load_best_model_at_end=True,
    metric_for_best_model="eval_accuracy",
    greater_is_better=True,
)  # end of TrainingArguments
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
        }
    )

# Train
trainer = PairwiseTrainer(
    model=model,
    args=args,
    train_dataset=tok_ds["train"],
    eval_dataset=tok_ds["validation"],
    processing_class=tok,
    data_collator=PairwiseCollator(tokenizer=tok, padding="longest"),
    compute_metrics=compute_accuracy,
)

logger.info(f"Starting training; evaluation every {config.eval_steps} steps")
train_out = trainer.train()
trainer.save_metrics("train", train_out.metrics)
if config.report_to == "wandb":
    # Log final training metrics
    wandb.log({f"train/{k}": v for k, v in train_out.metrics.items()})

logger.info(f"Saving LoRA adapter and value head to {reward_output_dir}")
# Save LoRA adapter from the backbone (PeftModel) and the reward head
model.backbone.save_pretrained(reward_output_dir)

# Save the value head separately
os.makedirs(reward_output_dir, exist_ok=True)
torch.save(model.v_head.state_dict(), os.path.join(reward_output_dir, "v_head.bin"))

# Also save tokenizer config for scoring convenience
model.tokenizer.save_pretrained(reward_output_dir)

eval_metrics = trainer.evaluate(eval_dataset=tok_ds["validation"])
trainer.save_metrics("eval", eval_metrics)
if config.report_to == "wandb":
    # Log final evaluation metrics
    wandb.log({"final_eval/accuracy": eval_metrics.get("eval_accuracy")})
    wandb.finish()

logger.info(f"Done – final accuracy {eval_metrics.get('eval_accuracy', 0.0):.4f}")
