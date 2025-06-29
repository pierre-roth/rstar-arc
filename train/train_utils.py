"""
Utility functions for training scripts (policy & reward).
"""
import logging
from collections import defaultdict

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from transformers import PreTrainedTokenizerBase


def maybe_peft_wrap(model, config):
    """
    Wrap the given model with LoRA adapters unless full_finetune is enabled.

    Args:
        model: a HuggingFace PreTrainedModel to (optionally) wrap
        config: rstar_deepthink.Config instance with LoRA settings

    Returns:
        model: if full_finetune is True, the original model; otherwise a PeftModel with LoRA adapters.
    """
    logger = logging.getLogger(__name__)
    # If user requests full-model fine-tuning, skip LoRA
    if config.full_finetune:
        logger.info("Full-model fine-tuning enabled: skipping LoRA adapters")
        return model

    # Build LoRA configuration
    lora_cfg = LoraConfig(
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
    logger.info(
        "LoRA adapters enabled: r=%d, α=%d, dropout=%.2f",
        lora_cfg.r, lora_cfg.lora_alpha, lora_cfg.lora_dropout
    )
    # Wrap model with PEFT LoRA adapters
    return get_peft_model(model, lora_cfg)


def renormalize_task_weights(ds: Dataset) -> Dataset:
    """Ensure per-task weights in the dataset sum to 1."""
    totals = defaultdict(float)
    for row in ds:
        totals[row["task_name"]] += float(row.get("weight", 0.0))

    def _scale(ex, totals=totals):
        total = totals.get(ex["task_name"], 1.0)
        return {"weight": float(ex["weight"]) / total if total else 0.0}

    return ds.map(_scale)


class WeightedCollator:
    """Pad dynamically and preserve per-example weights."""

    def __init__(self, tokenizer: PreTrainedTokenizerBase, max_len: int):
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __call__(self, features: list[dict]):
        weights = torch.tensor([f.pop("weight") for f in features], dtype=torch.float32)

        # remove labels before tokenizer padding - HF tokenizer can't pad them directly
        labels = [f.pop("labels") for f in features]

        # pad input IDs/attention masks to the longest sequence in the batch

        batch = self.tokenizer.pad(
            features,
            padding=True,
            return_tensors="pt",
        )

        max_len = batch["input_ids"].shape[1]
        padded_labels = torch.full((len(labels), max_len), self.tokenizer.pad_token_id, dtype=torch.long)
        for i, lab in enumerate(labels):
            length = min(len(lab), max_len)
            padded_labels[i, :length] = torch.tensor(lab[:length], dtype=torch.long)

        padded_labels[padded_labels == self.tokenizer.pad_token_id] = -100

        batch["labels"] = padded_labels
        batch["weight"] = weights
        return batch
