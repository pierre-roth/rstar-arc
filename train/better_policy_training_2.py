import logging
import math
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from accelerate import Accelerator
from accelerate.utils import set_seed
from datasets import Dataset, load_dataset
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    get_scheduler,
)

# Add project root to path
project_root = Path(__file__).parent.parent.absolute()
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from constants import (
    LOCAL_SCRATCH_PATH,
    NET_SCRATCH_PATH,
    SFT_IN_BETWEEN_PROMPT,
    SFT_SYSTEM_PROMPT,
    SPECIAL_TOKENS,
)
from rstar_deepthink import Config
from rstar_deepthink.arc_task import ARCTask
from rstar_deepthink.arc_task.task_utils import task_to_prompt
from train_utils import renormalize_task_weights
from utils import setup_logging

logger = logging.getLogger(__name__)


@dataclass
class TrainingMetrics:
    """Container for training metrics."""
    best_eval_loss: float = float('inf')
    best_step: int = 0
    global_step: int = 0


class DataCollatorForSFT:
    """
    Efficient collator for supervised fine-tuning with proper tokenization and masking.
    """

    def __init__(self, tokenizer: PreTrainedTokenizerBase, max_len: int):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self._eos_token_id = tokenizer.eos_token_id

    def __call__(self, features: List[Dict]) -> Dict[str, torch.Tensor]:
        prompts = []
        completions = []
        weights_list = []

        for ex in features:
            task_json = ex["task_json"]
            solution = ex["solution"]
            prompt = (
                SFT_SYSTEM_PROMPT
                + task_to_prompt(ARCTask.from_dict(task_json))
                + SFT_IN_BETWEEN_PROMPT
            )
            prompts.append(prompt)
            completions.append(solution)
            weights_list.append(float(ex.get("weight", 1.0)))

        weights = torch.tensor(weights_list, dtype=torch.float32)

        prompt_tokens_batch = self.tokenizer(
            prompts, add_special_tokens=False, padding=False
        )["input_ids"]
        completion_tokens_batch = self.tokenizer(
            completions, add_special_tokens=False, padding=False
        )["input_ids"]

        all_input_ids = []
        all_labels = []

        for i in range(len(features)):
            prompt_tokens = prompt_tokens_batch[i]
            completion_tokens = completion_tokens_batch[i] + [self._eos_token_id]

            # Combine and truncate if necessary
            prompt_len = len(prompt_tokens)
            total_tokens = prompt_tokens + completion_tokens

            if len(total_tokens) > self.max_len:
                if prompt_len < self.max_len:
                    total_tokens = total_tokens[: self.max_len]
                    actual_prompt_len = prompt_len
                else:
                    total_tokens = prompt_tokens[: self.max_len]
                    actual_prompt_len = self.max_len
            else:
                actual_prompt_len = prompt_len

            labels = total_tokens[:]  # Copy all tokens first
            # Now mask the prompt portion
            labels[:actual_prompt_len] = [-100] * actual_prompt_len

            all_input_ids.append(total_tokens)
            all_labels.append(labels)

        # Pad sequences efficiently
        batch = self._pad_sequences(all_input_ids, self.tokenizer.pad_token_id)
        labels = self._pad_sequences(all_labels, -100)

        return {
            "input_ids": batch,
            "attention_mask": (batch != self.tokenizer.pad_token_id).long(),
            "labels": labels,
            "weight": weights,
        }

    @staticmethod
    def _pad_sequences(sequences: List[List[int]], pad_value: int) -> torch.Tensor:
        """Efficiently pad sequences to the same length."""
        max_len = max(len(seq) for seq in sequences)
        padded = torch.full((len(sequences), max_len), pad_value, dtype=torch.long)

        for i, seq in enumerate(sequences):
            padded[i, :len(seq)] = torch.tensor(seq, dtype=torch.long)

        return padded


class SFTTrainer:
    """Encapsulates the training logic for better organization and testing."""

    def __init__(
            self,
            config: Config,
            model: PreTrainedModel,
            tokenizer: PreTrainedTokenizerBase,
            accelerator: Accelerator,
            output_dir: Path,
    ):
        self.config = config
        self.model = model
        self.tokenizer = tokenizer
        self.accelerator = accelerator
        self.output_dir = output_dir
        self.best_model_dir = output_dir / "best_model"
        self.metrics = TrainingMetrics()

        # Create directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.best_model_dir.mkdir(parents=True, exist_ok=True)

    def compute_loss(
        self, logits: torch.Tensor, labels: torch.Tensor, weights: torch.Tensor, reduction: str = "mean"
    ) -> torch.Tensor:
        """Computes weighted cross-entropy loss. Can return total loss or per-sequence loss."""
        shift_logits = logits[..., :-1, :].contiguous().float()
        shift_labels = labels[..., 1:].contiguous()

        loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100, reduction="none")
        per_token_loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
        )
        per_token_loss = per_token_loss.view(shift_labels.size())

        active_mask = shift_labels != -100
        per_seq_loss = (per_token_loss * active_mask).sum(dim=1) / active_mask.sum(dim=1).clamp_min(1)

        if reduction == "none":
            return per_seq_loss

        # Default: compute weighted mean loss for the batch
        device_weights = weights.to(per_seq_loss.device)
        weighted_loss = (per_seq_loss * device_weights).sum() / device_weights.sum().clamp_min(1e-8)
        return weighted_loss

    def evaluate(self, dataloader: DataLoader, prefix: str) -> Optional[float]:
        """Evaluate model on a validation set with correct distributed averaging."""
        self.model.eval()
        all_per_seq_losses = []
        all_weights = []

        with torch.no_grad():
            for batch in dataloader:
                outputs = self.model(
                    input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]
                )

                # Get per-sequence losses, not the final batch average
                per_seq_loss = self.compute_loss(
                    outputs.logits, batch["labels"], batch["weight"], reduction="none"
                )

                # Gather per-sequence losses and their corresponding weights
                gathered_losses = self.accelerator.gather_for_metrics(per_seq_loss)
                gathered_weights = self.accelerator.gather_for_metrics(batch["weight"])

                all_per_seq_losses.append(gathered_losses)
                all_weights.append(gathered_weights)

        total_loss = None
        if self.accelerator.is_main_process:
            # Concatenate all gathered tensors
            loss_tensor = torch.cat(all_per_seq_losses)
            weight_tensor = torch.cat(all_weights).to(loss_tensor.device)

            # Perform the final, correctly weighted average
            total_loss_tensor = (loss_tensor * weight_tensor).sum() / weight_tensor.sum().clamp_min(1e-8)
            total_loss = total_loss_tensor.item()

            self.accelerator.log({f"{prefix}/loss": total_loss}, step=self.metrics.global_step)

        self.model.train()

        # Broadcast the final, correct loss to all processes
        total_loss_tensor = torch.tensor(
            total_loss if total_loss is not None else 0.0, device=self.accelerator.device
        )
        total_loss_tensor = self.accelerator.broadcast(total_loss_tensor, from_process=0)
        return total_loss_tensor.item()

    def save_checkpoint(self, is_best: bool = False, step: Optional[int] = None):
        """Save model checkpoint with proper synchronization and versioning."""
        if is_best:
            save_dir = self.best_model_dir
        elif step is not None:
            save_dir = self.output_dir / f"checkpoint-{step}"
        else:
            save_dir = self.output_dir

        save_dir.mkdir(parents=True, exist_ok=True)

        self.accelerator.wait_for_everyone()

        if self.accelerator.is_main_process:
            unwrapped_model = self.accelerator.unwrap_model(self.model)
            unwrapped_model.save_pretrained(
                save_dir,
                safe_serialization=True,
                save_function=self.accelerator.save
            )
            self.tokenizer.save_pretrained(save_dir)

            # Save training state
            state = {
                "global_step": self.metrics.global_step,
                "best_eval_loss": self.metrics.best_eval_loss,
                "best_step": self.metrics.best_step,
                "config": asdict(self.config),
            }
            torch.save(state, save_dir / "training_state.pt")

    def train(
            self,
            train_dataloader: DataLoader,
            val_dataloaders: Dict[str, DataLoader],
            optimizer: torch.optim.Optimizer,
            lr_scheduler: torch.optim.lr_scheduler._LRScheduler,
            max_train_steps: int,
    ):
        """Main training loop with proper step counting and gradient clipping."""
        self.model.train()

        for epoch in range(self.config.num_train_epochs):
            for step, batch in enumerate(train_dataloader):
                with self.accelerator.accumulate(self.model):
                    # Forward pass
                    outputs = self.model(
                        input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                    )

                    # Compute loss
                    loss = self.compute_loss(
                        outputs.logits, batch["labels"], batch["weight"]
                    )

                    # Backward pass
                    self.accelerator.backward(loss)

                    # Optimizer step (only when gradients are accumulated)
                    if self.accelerator.sync_gradients:
                        if self.config.max_grad_norm > 0:
                            self.accelerator.clip_grad_norm_(
                                self.model.parameters(),
                                self.config.max_grad_norm
                            )

                        optimizer.step()
                        lr_scheduler.step()
                        optimizer.zero_grad()

                # Only increment global_step after optimization
                if self.accelerator.sync_gradients:
                    self.metrics.global_step += 1

                    # Logging
                    if self.metrics.global_step % self.config.logging_steps == 0:
                        # Gather loss from all processes for more accurate logging
                        avg_loss = self.accelerator.gather_for_metrics(loss.detach()).mean()
                        self.accelerator.log({
                            "train/loss": avg_loss.item(),
                            "train/learning_rate": lr_scheduler.get_last_lr()[0],
                            "train/epoch": epoch,
                        }, step=self.metrics.global_step)

                    # Evaluation
                    if (self.metrics.global_step % self.config.eval_steps == 0 and
                            self.metrics.global_step > 0):

                        for name, val_loader in val_dataloaders.items():
                            eval_loss = self.evaluate(val_loader, name)

                            # Check for best model on primary validation set
                            if name == "val_val" and self.accelerator.is_main_process:
                                if eval_loss < self.metrics.best_eval_loss:
                                    self.metrics.best_eval_loss = eval_loss
                                    self.metrics.best_step = self.metrics.global_step

                                    # Log best metrics to wandb
                                    self.accelerator.log({
                                        "val_val/best_loss": self.metrics.best_eval_loss,
                                        "val_val/best_step": self.metrics.best_step,
                                    }, step=self.metrics.global_step)

                                    logger.info(
                                        f"New best {name} loss: {eval_loss:.4f} "
                                        f"at step {self.metrics.global_step}"
                                    )
                                    self.save_checkpoint(is_best=True)

                    # Regular checkpointing with step number
                    if self.metrics.global_step % self.config.save_steps == 0:
                        self.save_checkpoint(is_best=False, step=self.metrics.global_step)

                if self.metrics.global_step >= max_train_steps:
                    return


def create_validation_splits(
        dataset: Dataset,
        config: Config,
        seed: int = 42,
) -> Tuple[Dataset, Optional[Dataset], Optional[Dataset]]:
    """Create validation splits with clearer logic."""
    rng = random.Random(seed)

    # Group examples by task
    task_to_indices: Dict[str, List[int]] = {}
    for idx, ex in enumerate(dataset):
        task_to_indices.setdefault(ex["task_name"], []).append(idx)

    # Split by task
    val_task_indices = []
    if config.task_validation_fraction > 0:
        all_tasks = list(task_to_indices.keys())
        rng.shuffle(all_tasks)

        n_val_tasks = int(len(all_tasks) * config.task_validation_fraction)
        val_tasks = set(all_tasks[:n_val_tasks])

        for task in val_tasks:
            val_task_indices.extend(task_to_indices[task])
    else:
        val_tasks = set()

    # Split remaining tasks by example
    val_example_indices = []
    train_indices = []

    for task, indices in task_to_indices.items():
        if task in val_tasks:
            continue

        indices_copy = list(indices)
        rng.shuffle(indices_copy)

        # Take validation examples if task has enough samples
        if (config.example_validation_num > 0 and
                len(indices_copy) >= config.example_validation_threshold):

            n_val = min(config.example_validation_num, len(indices_copy) // 2)
            val_example_indices.extend(indices_copy[:n_val])
            train_indices.extend(indices_copy[n_val:])
        else:
            train_indices.extend(indices_copy)

    # Create dataset splits
    train_ds = dataset.select(train_indices)
    val_task_ds = dataset.select(val_task_indices) if val_task_indices else None
    val_example_ds = dataset.select(val_example_indices) if val_example_indices else None

    return train_ds, val_task_ds, val_example_ds


def main(config: Config):
    """Main training function with improved structure."""
    # Setup
    set_seed(config.seed or 42)
    setup_logging(config.numeric_log_level)

    # Paths
    train_path = Path(NET_SCRATCH_PATH) / "sft_data" / f"round_{config.round_number}" / config.training_dataset_name
    val_path = Path(NET_SCRATCH_PATH) / "sft_data" / f"round_{config.round_number}" / config.validation_dataset_name

    run_name = f"policy-ft-{Path(config.policy_model).name}"
    # output_dir = Path(NET_SCRATCH_PATH) / "models" / "fine_tuned" / "policy" / run_name
    output_dir = Path("/scratch") / "net_scratch" / "models" / "fine_tuned" / "policy" / run_name

    # Initialize accelerator
    accelerator_config = {
        "log_with": "wandb" if config.report_to == "wandb" else None,
        "mixed_precision": "bf16" if config.use_bf16 else "fp16",
        "gradient_accumulation_steps": config.gradient_accumulation_steps,
        "dynamo_backend": "no"
    }
    if config.torch_compile:
        # A common backend is "inductor", but you can choose others.
        # "reduce-overhead" is a good mode for inference-like workloads.
        accelerator_config["dynamo_backend"] = "reduce-overhead"

    accelerator = Accelerator(**accelerator_config)

    if accelerator.is_main_process and config.report_to == "wandb":
        accelerator.init_trackers(
            project_name=config.wandb_project,
            config=asdict(config),
            init_kwargs={"wandb": {"name": run_name}},
        )

    if not accelerator.is_main_process:
        logging.getLogger().setLevel(logging.WARNING)

    # Load tokenizer and model
    logger.info(f"Loading tokenizer and model: {config.policy_model}")

    try:
        tokenizer = AutoTokenizer.from_pretrained(
            config.policy_model,
            trust_remote_code=True
        )

        # Setup tokenizer
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
        tokenizer.padding_side = "right"

        # Add special tokens
        num_added_tokens = tokenizer.add_special_tokens({
            "additional_special_tokens": SPECIAL_TOKENS
        })

        # Load model
        model = AutoModelForCausalLM.from_pretrained(
            config.policy_model,
            torch_dtype=torch.bfloat16 if config.use_bf16 else torch.float16,
            trust_remote_code=True,
            attn_implementation=config.attn_implementation,
        )

        # Resize embeddings if needed
        if num_added_tokens > 0:
            model.resize_token_embeddings(len(tokenizer))
            logger.info(f"Resized token embeddings to {len(tokenizer)}")

        # Configure model for training
        model.config.use_cache = False
        if config.gradient_checkpointing:
            model.gradient_checkpointing_enable()

        if config.torch_compile:
            logger.info("Compiling the model with torch.compile...")
            model = torch.compile(model, mode="reduce-overhead")

        model.enable_input_require_grads()

    except Exception as e:
        logger.error(f"Failed to load model or tokenizer: {e}")
        raise

    # Load datasets
    logger.info("Loading datasets...")

    cache_dir = Path(LOCAL_SCRATCH_PATH) / ".cache" / "huggingface" / "datasets"
    num_proc = max(1, config.cpus - 1)

    def _add_length_column(batch):
        """Batched function to compute and add tokenized length."""
        texts = []
        for i in range(len(batch["solution"])):
            task_json = batch["task_json"][i]
            solution = batch["solution"][i]
            text = (
                    SFT_SYSTEM_PROMPT
                    + task_to_prompt(ARCTask.from_dict(task_json))
                    + SFT_IN_BETWEEN_PROMPT
                    + solution
                    + tokenizer.eos_token
            )
            texts.append(text)

        tokenized = tokenizer(texts)
        return {"length": [len(ids) for ids in tokenized["input_ids"]]}

    # Load and filter training data
    raw_train_data = load_dataset(
        "json",
        data_files={"train": str(train_path)},
        cache_dir=str(cache_dir),
    )["train"]

    orig_train_len = len(raw_train_data)
    train_data_with_length = raw_train_data.map(
        _add_length_column,
        batched=True,
        batch_size=1000,
        num_proc=num_proc,
    )
    filtered_train_data = train_data_with_length.filter(
        lambda x: x["length"] <= config.max_seq_len,
        num_proc=num_proc
    )

    logger.info(
        f"Filtered training data: {orig_train_len} -> {len(filtered_train_data)} "
        f"({orig_train_len - len(filtered_train_data)} examples removed)"
    )

    # Create validation splits
    train_ds, val_task_ds, val_example_ds = create_validation_splits(
        filtered_train_data, config, config.seed or 42
    )

    # Load external validation data
    raw_val_data = load_dataset(
        "json",
        data_files={"validation": str(val_path)},
        cache_dir=str(cache_dir),
    )["validation"]

    orig_val_len = len(raw_val_data)
    val_data_with_length = raw_val_data.map(
        _add_length_column,
        batched=True,
        batch_size=1000,
        num_proc=num_proc,
    )
    val_val_ds = val_data_with_length.filter(
        lambda x: x["length"] <= config.max_seq_len,
        num_proc=num_proc
    )

    logger.info(
        f"Filtered validation data: {orig_val_len} -> {len(val_val_ds)} "
        f"({orig_val_len - len(val_val_ds)} examples removed)"
    )

    # Renormalize weights
    train_ds = renormalize_task_weights(train_ds)
    if val_task_ds:
        val_task_ds = renormalize_task_weights(val_task_ds)
    if val_example_ds:
        val_example_ds = renormalize_task_weights(val_example_ds)
    if len(val_val_ds) > 0:
        val_val_ds = renormalize_task_weights(val_val_ds)
    else:
        val_val_ds = None

    # Shuffle training data
    train_ds = train_ds.shuffle(seed=config.seed or 42)

    # Log dataset statistics
    logger.info(f"Dataset sizes:")
    logger.info(f"  Train: {len(train_ds)}")
    if val_task_ds:
        logger.info(f"  Val (task): {len(val_task_ds)}")
    if val_example_ds:
        logger.info(f"  Val (example): {len(val_example_ds)}")
    if val_val_ds:
        logger.info(f"  Val (external): {len(val_val_ds)}")

    # Create data loaders
    collator = DataCollatorForSFT(tokenizer=tokenizer, max_len=config.max_seq_len)

    train_dataloader = DataLoader(
        train_ds,
        batch_size=config.per_device_train_batch_size,
        shuffle=True,
        collate_fn=collator,
        num_workers=min(4, config.cpus),
        pin_memory=True,
    )

    val_dataloaders = {}
    for name, ds in [
        ("val_task", val_task_ds),
        ("val_example", val_example_ds),
        ("val_val", val_val_ds),
    ]:
        if ds is not None:
            val_dataloaders[name] = DataLoader(
                ds,
                batch_size=config.per_device_eval_batch_size,
                shuffle=False,
                collate_fn=collator,
                num_workers=min(4, config.cpus),
                pin_memory=True,
            )

    # Setup optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
        betas=(0.9, 0.999),
        eps=1e-8,
    )

    # Prepare model, optimizer, and dataloader with accelerator
    model, optimizer, train_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader
    )

    # --- FIX: Calculate training steps *after* accelerator has sharded the dataloader ---
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / config.gradient_accumulation_steps
    )
    max_train_steps = num_update_steps_per_epoch * config.num_train_epochs

    # Now create the scheduler with the correct number of steps
    lr_scheduler = get_scheduler(
        name=config.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=int(config.warmup_ratio * max_train_steps),
        num_training_steps=max_train_steps,
    )

    # Prepare the scheduler
    lr_scheduler = accelerator.prepare(lr_scheduler)

    # Prepare validation dataloaders
    prepared_val_dataloaders = {}
    for name, loader in val_dataloaders.items():
        prepared_val_dataloaders[name] = accelerator.prepare(loader)

    # Initialize trainer
    trainer = SFTTrainer(
        config=config,
        model=model,
        tokenizer=tokenizer,
        accelerator=accelerator,
        output_dir=output_dir,
    )

    # Train
    logger.info(f"Starting training for {max_train_steps} steps...")
    trainer.train(
        train_dataloader=train_dataloader,
        val_dataloaders=prepared_val_dataloaders,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        max_train_steps=max_train_steps,
    )

    # Final evaluation
    logger.info("Running final evaluation...")
    for name, loader in prepared_val_dataloaders.items():
        eval_loss = trainer.evaluate(loader, f"final_{name}")
        if accelerator.is_main_process:
            logger.info(f"Final {name} loss: {eval_loss:.4f}")

    # Log best results
    if accelerator.is_main_process:
        logger.info(
            f"Training completed. Best validation loss: {trainer.metrics.best_eval_loss:.4f} "
            f"at step {trainer.metrics.best_step}"
        )

    # Cleanup
    if config.report_to == "wandb":
        accelerator.end_training()


if __name__ == "__main__":
    # Load configuration
    config = Config()

    # Run training
    try:
        main(config)
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
    except Exception as e:
        logger.error(f"Training failed with error: {e}", exc_info=True)
        raise
