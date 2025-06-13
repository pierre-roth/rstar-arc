from __future__ import annotations

import logging
import os
import sys
from typing import Dict, List

import torch
import wandb
from dataclasses import asdict
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
    set_seed,
)

# Project imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from constants import (
    LOCAL_SCRATCH_PATH,
    NET_SCRATCH_PATH,
    SFT_IN_BETWEEN_PROMPT,
    SFT_SYSTEM_PROMPT,
    SPECIAL_TOKENS
)
from rstar_deepthink import Config
from rstar_deepthink.arc_task import ARCTask
from rstar_deepthink.arc_task.task_utils import task_to_prompt
from train_utils import maybe_peft_wrap
from utils import setup_logging

logger = logging.getLogger(__name__)


class WeightedTrainer(Trainer):
    """Trainer with support for weighted loss computation."""

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """Compute weighted loss for examples."""
        # Extract weights if present
        weights = inputs.pop("weight")

        # Forward pass
        outputs = model(**inputs)

        # Compute weighted loss
        logits = outputs.logits
        labels = inputs.get("labels")

        # Shift for autoregressive loss
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        # Compute per-token loss with label smoothing if configured
        loss_fct = torch.nn.CrossEntropyLoss(
            reduction='none',
            label_smoothing=self.args.label_smoothing_factor
        )
        per_token_loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1)
        ).view(shift_labels.size())

        # Mask padding tokens and compute per-example loss
        mask = shift_labels != -100
        per_example_loss = (per_token_loss * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)

        # Apply weights
        weighted_loss = (per_example_loss * weights).sum() / weights.sum().clamp(min=1e-8)
        loss = weighted_loss

        return (loss, outputs) if return_outputs else loss


class WeightedDataCollator(DataCollatorForLanguageModeling):
    """Data collator that preserves weights while handling padding."""

    def __call__(self, features, return_tensors=None) -> Dict[str, torch.Tensor]:
        # Extract and temporarily remove weights
        weights = [f.pop("weight") for f in features]

        # Use parent class to handle padding and create batch
        batch = super().__call__(features, return_tensors=return_tensors)

        batch["weight"] = torch.tensor(weights, dtype=torch.float32)

        return batch


def preprocess_dataset(examples: Dict[str, List], tokenizer, max_length: int):
    """Preprocess examples for training on completions only."""
    formatted_texts = []
    prompts = []

    # Format each example and store prompts separately
    for task_json, solution in zip(examples["task_json"], examples["solution"]):
        task = ARCTask.from_dict(task_json)
        prompt = SFT_SYSTEM_PROMPT + task_to_prompt(task) + SFT_IN_BETWEEN_PROMPT
        prompts.append(prompt)
        formatted_texts.append(prompt + solution + tokenizer.eos_token)

    # Tokenize all examples
    model_inputs = tokenizer(
        formatted_texts,
        max_length=max_length,
        truncation=True,
        padding=False,  # Let collator handle padding
        return_tensors=None,
    )

    # Create labels with prompt tokens masked
    model_inputs["labels"] = []

    for i, prompt in enumerate(prompts):
        # Tokenize prompt to get its length
        prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        prompt_length = len(prompt_ids)

        # Create labels with prompt masked
        labels = model_inputs["input_ids"][i].copy()
        labels[:prompt_length] = [-100] * prompt_length
        model_inputs["labels"].append(labels)

    model_inputs["weight"] = examples["weight"]

    return model_inputs


def main():
    # Configuration
    config = Config()
    set_seed(config.seed or 42)
    setup_logging(config.numeric_log_level)

    # Paths
    train_path = os.path.join(
        NET_SCRATCH_PATH, "sft_data", f"round_{config.round_number}", "policy_dataset_training.jsonl"
    )
    val_path = os.path.join(
        NET_SCRATCH_PATH, "sft_data", f"round_{config.round_number}", "policy_dataset_validation.jsonl"
    )

    model_name = config.policy_model.split('/')[-1]
    run_name = f"policy-ft-{model_name}-lr{config.learning_rate}-wd{config.weight_decay}"
    output_dir = os.path.join(NET_SCRATCH_PATH, "models", "fine_tuned", "policy", run_name)
    os.makedirs(output_dir, exist_ok=True)

    if config.report_to == "wandb":
        # Convert Config object to dict if it has a method, or pass attributes manually
        config_dict = asdict(config)
        wandb.init(
            project=config.wandb_project,  # Specify project name
            entity=config.wandb_entity,
            name=run_name,
            config=config_dict,  # Log your config
            dir=output_dir  # Optional: specify wandb dir
        )

    logger.info(f"Model: {config.policy_model}")
    logger.info(f"Output directory: {output_dir}")

    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.policy_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    added_tokens = tokenizer.add_special_tokens({"additional_special_tokens": SPECIAL_TOKENS})
    if added_tokens > 0:
        logger.info(f"Added {added_tokens} special tokens to tokenizer")

    # Initialize model
    model = AutoModelForCausalLM.from_pretrained(
        config.policy_model,
        torch_dtype=torch.bfloat16 if config.use_bf16 else torch.float16,
        trust_remote_code=True,
        use_cache=False,  # Disable cache for training
    )

    if added_tokens > 0:
        model.resize_token_embeddings(len(tokenizer))

    # Set pad_token_id to avoid warnings
    if model.config.pad_token_id is None:
        model.config.pad_token_id = tokenizer.pad_token_id

    # Enable gradient checkpointing
    if config.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    # Apply LoRA if configured
    model = maybe_peft_wrap(model, config)

    # Log model parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(
        f"Trainable parameters: {trainable_params:,} / {total_params:,} "
        f"({100 * trainable_params / total_params:.2f}%)"
    )

    # Load datasets
    logger.info("Loading datasets...")
    dataset = load_dataset(
        "json",
        data_files={"train": train_path, "validation": val_path},
        cache_dir=os.path.join(LOCAL_SCRATCH_PATH, ".cache/huggingface/datasets"),
    )

    # Shuffle training data
    dataset["train"] = dataset["train"].shuffle(seed=config.seed or 42)

    logger.info(f"Train examples: {len(dataset['train'])}")
    logger.info(f"Validation examples: {len(dataset['validation'])}")

    # Preprocess datasets
    logger.info("Preprocessing datasets...")
    preprocess_fn = lambda examples: preprocess_dataset(examples, tokenizer, config.max_seq_len)

    # Keep weight column if present
    remove_columns = [col for col in dataset["train"].column_names if col != "weight"]

    tokenized_datasets = dataset.map(
        preprocess_fn,
        batched=True,
        remove_columns=remove_columns,
        num_proc=4,
        desc="Tokenizing",
    )

    # Create appropriate data collator
    data_collator = WeightedDataCollator(
        tokenizer=tokenizer,
        mlm=False,
        pad_to_multiple_of=8,  # Efficient padding for mixed precision
    )
    trainer_class = WeightedTrainer

    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        run_name=run_name,

        # Training hyperparameters
        num_train_epochs=config.num_train_epochs,
        per_device_train_batch_size=config.per_device_train_batch_size,
        per_device_eval_batch_size=config.per_device_eval_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,

        # Optimizer settings
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        warmup_ratio=config.warmup_ratio,
        lr_scheduler_type=config.lr_scheduler_type,
        max_grad_norm=config.max_grad_norm,

        # Precision
        bf16=config.use_bf16,
        fp16=not config.use_bf16,

        # Logging and evaluation
        logging_steps=config.logging_steps,
        eval_strategy="steps",
        eval_steps=config.eval_steps,

        # Checkpointing
        save_strategy="steps",
        save_steps=config.save_steps,
        save_total_limit=config.save_total_limit,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",

        # Other settings
        report_to=config.report_to if config.report_to else "none",
        push_to_hub=False,
        remove_unused_columns=False,
        label_smoothing_factor=config.label_smoothing_factor,
    )

    # Initialize trainer
    trainer = trainer_class(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    # Train
    logger.info("Starting training...")
    train_result = trainer.train()

    # Save final model
    trainer.save_model()
    tokenizer.save_pretrained(output_dir)
    trainer.save_state()

    # Evaluate
    logger.info("Running final evaluation...")
    eval_metrics = trainer.evaluate()
    trainer.save_metrics("eval", eval_metrics)

    logger.info("Training completed!")

    # Log to wandb if configured
    if config.report_to == "wandb":
        wandb.finish()


if __name__ == "__main__":
    main()
