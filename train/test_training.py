import argparse
import logging
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    set_seed,
)

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse the command-line arguments."""
    parser = argparse.ArgumentParser(description="A simple single-GPU finetuning script using HF Trainer.")
    parser.add_argument("--model_name_or_path", type=str, default="Qwen/Qwen2-0.5B",
                        help="HF Hub id of the base model.")
    parser.add_argument("--dataset_name", type=str, default="tatsu-lab/alpaca", help="HF Hub dataset repo id.")
    parser.add_argument("--output_dir", type=str, default="./alpaca-finetuned-model",
                        help="Where to store the final model.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_length", type=int, default=16384)  # Reduced for feasibility on consumer GPUs
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--bf16", action="store_true", help="Enable bfloat16 training.")
    return parser.parse_args()


def format_prompt(example: dict) -> str:
    """Convert a single Alpaca record into an instruction-style prompt."""
    instruction = example["instruction"]
    output = example["output"]
    # Handle examples with and without an 'input' field
    if example.get("input"):
        input_text = example["input"]
        return (
            "Below is an instruction that describes a task, paired with an input"
            " that provides further context. Write a response that appropriately"
            f" completes the request.\n\n### Instruction:\n{instruction}\n\n"
            f"### Input:\n{input_text}\n\n### Response:\n{output}"
        )
    return (
        "Below is an instruction that describes a task. Write a response that"
        f" appropriately completes the request.\n\n### Instruction:\n{instruction}"
        f"\n\n### Response:\n{output}"
    )


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    # --- 1. Load Model and Tokenizer ---
    logger.info(f"Loading model: {args.model_name_or_path}")
    # Using a smaller model for demonstration to make it runnable on more systems.
    # The original "Qwen/Qwen2.5-3B" would require substantial VRAM.
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=torch.bfloat16 if args.bf16 else torch.float32,
        trust_remote_code=True,
        use_cache=False,  # Important for training
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.pad_token_id

    # --- 2. Load and Process Dataset ---
    logger.info(f"Loading and processing dataset: {args.dataset_name}")
    dataset = load_dataset(args.dataset_name, split="train")

    def tokenize_function(examples):
        # Format prompts and then tokenize
        prompts = [format_prompt(ex) + tokenizer.eos_token for ex in examples]
        tokenized_output = tokenizer(
            prompts,
            truncation=True,
            padding="max_length",
            max_length=args.max_length,
        )
        # Set labels to be the same as input_ids for language modeling
        tokenized_output["labels"] = tokenized_output["input_ids"].copy()
        return tokenized_output

    # Using a smaller subset for a quick demonstration. Remove `.select(range(1000))` to run on the full dataset.
    processed_dataset = dataset.select(range(1000)).map(
        tokenize_function,
        batched=True,
        remove_columns=list(dataset.column_names),
        desc="Tokenizing dataset",
    )

    # --- 3. Configure Training ---
    logger.info("Setting up training arguments.")
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        lr_scheduler_type="cosine",
        warmup_ratio=args.warmup_ratio,
        logging_steps=10,
        save_strategy="epoch",
        bf16=args.bf16,
        gradient_checkpointing=True,  # Reduces memory usage
        report_to="none",  # Disable wandb/tensorboard for simplicity
    )

    # --- 4. Initialize Trainer and Start Training ---
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=processed_dataset,
        tokenizer=tokenizer,
    )

    logger.info("Starting training...")
    trainer.train()
    logger.info("Training finished.")

    # --- 5. Save the Final Model ---
    logger.info(f"Saving model to {args.output_dir}")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    logger.info("Script finished successfully.")


if __name__ == "__main__":
    main()
