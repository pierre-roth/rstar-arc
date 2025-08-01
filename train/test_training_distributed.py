#!/usr/bin/env python

import argparse
import logging
import os
import warnings
from datetime import datetime
from pathlib import Path

import torch
from accelerate import Accelerator
from accelerate.logging import get_logger
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
    set_seed,
    logging as hf_logging
)


os.environ["NCCL_DEBUG"] = "WARN"
os.environ["WANDB_SILENT"] = "true"
os.environ["FLASH_ATTENTION_SKIP_INIT_WARNING"] = "1"
# os.environ["TORCHDYNAMO_CAPTURE_SCALAR_OUTPUTS"] = "1"

# -----------------------------------------------------------------------------#
# 1. Utilities
# -----------------------------------------------------------------------------#


NET_SCRATCH_PATH = f"/itet-stor/piroth/net_scratch"  # net-scratch directory
run_name = f"policy-ft-test"
# out_dir = os.path.join(NET_SCRATCH_PATH, "models", "fine_tuned", "policy", run_name)
out_dir = Path("/scratch") / "net_scratch" / "models" / "fine_tuned" / "policy" / run_name


def parse_args() -> argparse.Namespace:
    """Parse the command‑line arguments."""
    parser = argparse.ArgumentParser(
        description="Tiny finetuning script showcasing Accelerate."
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="Qwen/Qwen2.5-3B",
        help="HF Hub id or local path of the base model.",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="tatsu-lab/alpaca",
        help="HF Hub dataset repo id to finetune on.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=out_dir,
        help="Where to store the final checkpoint.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_length", type=int, default=16384)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=16)
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument("--num_warmup_steps", type=int, default=0)
    parser.add_argument(
        "--mixed_precision",
        choices=["no", "fp16", "bf16"],
        default="bf16",
        help="Precision for training.",
    )

    return parser.parse_args()


def _bytes(obj_size: int) -> float:
    """Convert bytes to megabytes."""
    return obj_size / 1024 ** 2


def parameters_memory(model: torch.nn.Module) -> int:
    """Return total parameter memory in bytes."""
    return sum(p.numel() * p.element_size() for p in model.parameters())


def optimizer_memory(optimizer: torch.optim.Optimizer) -> int:
    """Return the memory used by optimizer states in bytes."""
    total = 0
    for group in optimizer.param_groups:
        for p in group["params"]:
            state = optimizer.state.get(p, {})
            for val in state.values():
                if torch.is_tensor(val):
                    total += val.numel() * val.element_size()
    return total


def gradients_memory(model: torch.nn.Module) -> int:
    """Return the memory used by gradients in bytes."""
    total = 0
    for p in model.parameters():
        if p.grad is not None:
            total += p.grad.numel() * p.grad.element_size()
    return total


def format_prompt(example: dict) -> str:
    """Convert a single Alpaca record into an instruction‑style prompt."""
    instruction = example["instruction"]
    output = example["output"]
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


# -----------------------------------------------------------------------------#
# 2. Main
# -----------------------------------------------------------------------------#


def main() -> None:
    args = parse_args()

    # ------------- Logging & Reproducibility --------------------------------#
    logger = get_logger(__name__)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    set_seed(args.seed)

    # ------------- Accelerator & Weights & Biases ---------------------------#
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=(None if args.mixed_precision == "no" else args.mixed_precision),
        log_with="wandb",
    )

    if not accelerator.is_main_process:
        hf_logging.set_verbosity_error()
        warnings.filterwarnings("ignore")

    if accelerator.is_main_process:
        accelerator.init_trackers(
            project_name="simple-accelerate-demo",
            config=vars(args),
            init_kwargs={
                "wandb": {
                    "name": f"{Path(args.model_name_or_path).name}-"
                            f"{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                }
            },
        )

    # ------------- Model & Tokenizer ----------------------------------------#
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path, trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        attn_implementation="flash_attention_2",  # or "flash_attention_3" if you built FA-3
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        use_cache=False,
    )

    if not accelerator.is_main_process:
        warnings.resetwarnings()

    model.gradient_checkpointing_enable()

    if accelerator.is_main_process:
        logger.info(f"Gradient checkpointing enabled: {model.is_gradient_checkpointing}")

    model.config.pad_token_id = tokenizer.pad_token_id

    # ------------- Dataset ---------------------------------------------------#
    raw_datasets = load_dataset(args.dataset_name, split="train")
    if accelerator.is_main_process:
        logger.info(f"Loaded {len(raw_datasets):,} training examples.")

    def tokenize_fn(batch):
        prompts = [
            format_prompt(
                {
                    "instruction": instr,
                    "input": inp,
                    "output": out,
                }
            ) + tokenizer.eos_token
            for instr, inp, out in zip(
                batch["instruction"], batch["input"], batch["output"]
            )
        ]

        encoded = tokenizer(
            prompts,
            padding="max_length",
            truncation=True,
            max_length=args.max_length,
        )
        encoded["labels"] = encoded["input_ids"].copy()
        return encoded

    with accelerator.main_process_first():
        processed_dataset = raw_datasets.map(
            tokenize_fn,
            batched=True,
            remove_columns=list(raw_datasets.column_names),
            num_proc=18,
            desc="Tokenising",
        )
        processed_dataset.set_format(
            type="torch", columns=["input_ids", "attention_mask", "labels"]
        )

    train_dataloader = DataLoader(
        processed_dataset,
        batch_size=args.per_device_train_batch_size,
        shuffle=True,
    )

    # ------------- Optimiser & Scheduler ------------------------------------#
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    total_update_steps = (
            len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs
    )
    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer, args.num_warmup_steps, total_update_steps
    )

    # ------------- Prepare for Accelerate -----------------------------------#
    (
        model,
        optimizer,
        train_dataloader,
        lr_scheduler,
    ) = accelerator.prepare(model, optimizer, train_dataloader, lr_scheduler)

    # ------------- Memory Snapshot ------------------------------------------#
    baseline_mem = 0
    current_baseline_mem = 0
    if accelerator.device.type == "cuda" and accelerator.is_main_process:
        logger.info("Taking a VRAM snapshot after setup...")
        baseline_mem = torch.cuda.memory_allocated()
        current_baseline_mem = baseline_mem
        param_mem = parameters_memory(model)
        opt_mem = optimizer_memory(optimizer)  # zero before the first optimizer update
        overhead = max(baseline_mem - param_mem - opt_mem, 0)
        logger.info(f"Initial VRAM usage: {_bytes(baseline_mem):.2f} MB")
        logger.info(f" - Weights          : {_bytes(param_mem):.2f} MB")
        logger.info(f" - Optimizer states : {_bytes(opt_mem):.2f} MB")
        logger.info(f" - Additional overhead: {_bytes(overhead):.2f} MB")

        # Full memory summary
        for i in range(torch.cuda.device_count()):
            logger.info(f"--- VRAM summary for GPU {i} ---")
            logger.info(torch.cuda.memory_summary(device=i, abbreviated=False))

    # ------------- Training Loop --------------------------------------------#
    progress_bar = tqdm(
        range(total_update_steps),
        disable=not accelerator.is_local_main_process,
        position=0,
        leave=True,
    )
    model.train()
    global_step = 0

    for epoch in range(args.num_train_epochs):
        for step, batch in enumerate(train_dataloader):
            # --- Peak Memory Measurement (First Step) ---
            if global_step == 0 and step == 0 and accelerator.is_main_process and accelerator.device.type == "cuda":
                torch.cuda.reset_peak_memory_stats()

            with accelerator.accumulate(model):
                outputs = model(**batch)
                loss = outputs.loss

                if global_step == 0 and step == 0 and accelerator.is_main_process and accelerator.device.type == "cuda":
                    after_fwd = torch.cuda.memory_allocated()
                    activation_mem = max(after_fwd - current_baseline_mem, 0)
                    logger.info(f"Activation memory after forward: {_bytes(activation_mem):.2f} MB")

                accelerator.backward(loss)

                if global_step == 0 and step == 0 and accelerator.is_main_process and accelerator.device.type == "cuda":
                    grad_mem = gradients_memory(model)
                    after_bwd = torch.cuda.memory_allocated()
                    step_overhead = max(after_bwd - current_baseline_mem - grad_mem, 0)
                    logger.info(f"Gradient memory: {_bytes(grad_mem):.2f} MB")
                    logger.info(f"Additional overhead this step: {_bytes(step_overhead):.2f} MB")
                    logger.info(f"Total VRAM after backward: {_bytes(after_bwd):.2f} MB")
                    # Update baseline to include persistent gradient buffers
                    current_baseline_mem = after_bwd

                if accelerator.sync_gradients:
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()

                    # Log peak memory after the first full step
                    if global_step == 0 and step == 0 and accelerator.is_main_process and accelerator.device.type == "cuda":
                        peak_mem = torch.cuda.max_memory_allocated()
                        logger.info(f"Peak memory usage during first step: {_bytes(peak_mem):.2f} MB")

                    # Logging (only on main process)
                    avg_loss = accelerator.gather(loss.detach()).mean().item()
                    accelerator.log({"train/loss": avg_loss}, step=global_step)
                    progress_bar.set_description(f"Epoch {epoch + 1} | Loss {avg_loss:.4f}")
                    progress_bar.update(1)
                    global_step += 1

        accelerator.wait_for_everyone()

    # ------------- Save ------------------------------------------------------#
    accelerator.wait_for_everyone()
    unwrapped_model = accelerator.unwrap_model(model)

    if accelerator.is_main_process:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        unwrapped_model.save_pretrained(
            args.output_dir,
            state_dict=accelerator.get_state_dict(model),
            safe_serialization=True,
        )
        tokenizer.save_pretrained(args.output_dir)
        logger.info(f"Model & tokenizer saved to {args.output_dir.resolve()}")

    accelerator.end_training()


if __name__ == "__main__":
    main()
