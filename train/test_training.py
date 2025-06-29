# simple_train_wandb.py
#
# A simple script to demonstrate fine-tuning a Hugging Face LLM using the Accelerate library,
# with logging to Weights & Biases (wandb).
#
# To run this script, you first need to:
# 1. Install wandb: pip install wandb
# 2. Log in to your wandb account: wandb login
# 3. Configure Accelerate: accelerate config
#
# Then, you can launch the training job:
# > accelerate launch simple_train_wandb.py --<your_arguments>
#
# Example to run with gradient checkpointing:
# > accelerate launch simple_train_wandb.py \
#       --model_name_or_path "Qwen/Qwen2-1.5B-Instruct" \
#       --dataset_name "tatsu-lab/alpaca" \
#       --max_length 2048 \
#       --per_device_train_batch_size 1 \
#       --gradient_accumulation_steps 4 \
#       --learning_rate 2e-5 \
#       --num_train_epochs 1 \
#       --use_gradient_checkpointing

import argparse
import logging
import os

import torch
from accelerate import Accelerator
from accelerate.logging import get_logger
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
    set_seed,
)

# Set up logging
logger = get_logger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Simple LLM training script using Accelerate and W&B")
    parser.add_argument("--seed", type=int, default=42, help="A seed for reproducible training.")
    parser.add_argument("--model_name_or_path", type=str, default="Qwen/Qwen2.5-3B", help="Path to pretrained model or model identifier from huggingface.co/models.")
    parser.add_argument("--dataset_name", type=str, default="tatsu-lab/alpaca", help="The name of the dataset to use (via the datasets library).")
    parser.add_argument("--max_length", type=int, default=16384, help="The maximum total input sequence length after tokenization.")
    parser.add_argument("--per_device_train_batch_size", type=int, default=1, help="Batch size (per device) for the training dataloader.")
    parser.add_argument("--learning_rate", type=float, default=1e-5, help="Initial learning rate (after the potential warmup period).")
    parser.add_argument("--num_train_epochs", type=int, default=1, help="Total number of training epochs to perform.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=64, help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--num_warmup_steps", type=int, default=0, help="Number of steps for the warmup in the lr scheduler.")
    parser.add_argument("--use_gradient_checkpointing", action="store_true", default=True, help="Use gradient checkpointing to save memory.")
    parser.add_argument("--no_gradient_checkpointing", dest="use_gradient_checkpointing", action="store_false", help="Do not use gradient checkpointing.")

    args = parser.parse_args()
    return args


def create_prompt(example):
    """
    Formats a single data example into a standardized prompt format.
    This function is designed for the Alpaca dataset structure.
    """
    if example.get("input"):
        return (
            "Below is an instruction that describes a task, paired with an input that provides further context. "
            "Write a response that appropriately completes the request.\n\n"
            f"### Instruction:\n{example['instruction']}\n\n"
            f"### Input:\n{example['input']}\n\n"
            f"### Response:\n{example['output']}"
        )
    else:
        return (
            "Below is an instruction that describes a task. "
            "Write a response that appropriately completes the request.\n\n"
            f"### Instruction:\n{example['instruction']}\n\n"
            f"### Response:\n{example['output']}"
        )


def main():
    args = parse_args()

    # --- 1. Define Output Directory and Run Name ---
    # Use NET_SCRATCH_PATH environment variable if available, otherwise use a default.
    net_scratch_path = os.environ.get("NET_SCRATCH_PATH", "/tmp/llm_runs")
    run_name = f"test-policy-ft-{args.model_name_or_path.split('/')[-1]}"
    output_dir = os.path.join(net_scratch_path, "models", "fine_tuned", "policy", run_name)

    # --- 2. Initialize Accelerator with W&B logging ---
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision="bf16",  # Hardcoded to bf16
        log_with="wandb",
    )

    logger.info(f"Accelerator state: {accelerator.state}")
    logger.info(f"Output directory: {output_dir}")

    # --- 3. Set up reproducibility and logging ---
    set_seed(args.seed)
    if accelerator.is_main_process:
        os.makedirs(output_dir, exist_ok=True)
        # Initialize W&B tracker
        accelerator.init_trackers(
            project_name="simple-llm-finetuning",
            config=vars(args),
            init_kwargs={"wandb": {"name": run_name}}
        )

    # --- 4. Load Model and Tokenizer ---
    logger.info(f"Loading model: {args.model_name_or_path}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=True,  # Qwen models require this
        torch_dtype=torch.bfloat16,  # Load in bf16 directly
        use_cache=False if args.use_gradient_checkpointing else True,
    )

    if args.use_gradient_checkpointing:
        model.gradient_checkpointing_enable()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)

    if tokenizer.pad_token is None:
        logger.info("Tokenizer does not have a pad token, setting it to eos_token.")
        tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.pad_token_id

    # --- 5. Load and Preprocess Dataset ---
    logger.info(f"Loading dataset: {args.dataset_name}")
    raw_datasets = load_dataset(args.dataset_name)

    def tokenize_function(examples):
        full_prompts = [create_prompt(ex) + tokenizer.eos_token for ex in examples]
        tokenized_output = tokenizer(
            full_prompts,
            padding="max_length",
            truncation=True,
            max_length=args.max_length,
            return_tensors="pt"
        )
        tokenized_output["labels"] = tokenized_output["input_ids"].clone()
        return tokenized_output

    train_dataset = raw_datasets["train"]
    with accelerator.main_process_first():
        processed_dataset = train_dataset.map(
            tokenize_function,
            batched=True,
            batch_size=100,
            remove_columns=train_dataset.column_names,
            desc="Tokenizing dataset",
        )

    logger.info(f"Dataset processed. Number of examples: {len(processed_dataset)}")

    # --- 6. Create DataLoader, Optimizer, and Scheduler ---
    train_dataloader = DataLoader(
        processed_dataset,
        shuffle=True,
        batch_size=args.per_device_train_batch_size,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

    num_training_steps = len(train_dataloader) * args.num_train_epochs
    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps,
        num_training_steps=num_training_steps,
    )

    # --- 7. Prepare for training with Accelerate ---
    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler
    )

    # --- 8. Training Loop ---
    logger.info("***** Starting training *****")
    progress_bar = tqdm(range(num_training_steps), disable=not accelerator.is_local_main_process)
    completed_steps = 0

    model.train()
    for epoch in range(args.num_train_epochs):
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(model):
                outputs = model(**batch)
                loss = outputs.loss
                accelerator.backward(loss)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                progress_bar.update(1)
                completed_steps += 1

                # Log metrics to W&B
                # Gather loss across all processes for accurate logging
                avg_loss = accelerator.gather(loss.repeat(args.per_device_train_batch_size)).mean()
                accelerator.log({"loss": avg_loss.item()}, step=completed_steps)
                progress_bar.set_description(f"Epoch {epoch + 1} | Loss: {avg_loss.item():.4f}")

            if completed_steps >= num_training_steps:
                break

    # --- 9. Save the final model ---
    logger.info("***** Training finished, saving model *****")
    accelerator.end_training()  # Ends the W&B run
    accelerator.wait_for_everyone()
    unwrapped_model = accelerator.unwrap_model(model)
    unwrapped_model.save_pretrained(
        output_dir,
        is_main_process=accelerator.is_main_process,
        save_function=accelerator.save,
        state_dict=accelerator.get_state_dict(model),
    )
    if accelerator.is_main_process:
        tokenizer.save_pretrained(output_dir)
        logger.info(f"Model saved to {output_dir}")


if __name__ == "__main__":
    main()
