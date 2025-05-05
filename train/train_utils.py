"""
Utility functions for training scripts (policy & reward).
"""
import logging

from peft import LoraConfig, get_peft_model


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
    if getattr(config, "full_finetune", False):
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
