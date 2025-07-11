import logging
import math
import os
import sys
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path

import datasets
import torch
from accelerate import Accelerator
from accelerate.utils import set_seed, broadcast, ProjectConfiguration
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
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
    NET_SCRATCH_PATH,
)
from rstar_deepthink import Config
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

    def __call__(self, features: list[dict]) -> dict[str, torch.Tensor]:
        """Collate a batch of raw or pre-tokenized examples."""
        all_input_ids = [ex["input_ids"] for ex in features]
        all_labels = [ex["labels"] for ex in features]
        weights = torch.tensor([ex["weight"] for ex in features], dtype=torch.float32)

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
    def _pad_sequences(sequences: list[list[int]], pad_value: int) -> torch.Tensor:
        """Efficiently pad sequences to the same length."""
        max_len = max(len(seq) for seq in sequences)
        padded = torch.full((len(sequences), max_len), pad_value, dtype=torch.long)

        for i, seq in enumerate(sequences):
            padded[i, :len(seq)] = seq

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

        self.loss_window: deque[torch.Tensor] = deque(
            maxlen=config.logging_steps  # keeps exactly the last N steps
        )

        # Create directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.best_model_dir.mkdir(parents=True, exist_ok=True)

    def compute_loss(
            self, logits: torch.Tensor, labels: torch.Tensor, weights: torch.Tensor, reduction: str = "mean"
    ) -> torch.Tensor:
        """Computes weighted cross-entropy loss. Can return total loss or per-sequence loss."""
        shift_logits = logits[..., :-1, :].contiguous()
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

    def evaluate(self, dataloader: DataLoader, prefix: str) -> float | None:
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
        total_loss_tensor = broadcast(total_loss_tensor, from_process=0)
        return total_loss_tensor.item()

    def save_checkpoint(self, is_best: bool = False, step: int | None = None):
        """Save model checkpoint with proper synchronization and versioning."""
        if is_best:
            save_dir = self.best_model_dir
        elif step is not None:
            save_dir = self.output_dir / f"checkpoint-{step}"
        else:
            save_dir = self.output_dir

        save_dir.mkdir(parents=True, exist_ok=True)

        self.accelerator.wait_for_everyone()

        self.accelerator.save_state(str(save_dir))

        if self.accelerator.is_main_process:
            # Save training state
            state = {
                "global_step": self.metrics.global_step,
                "best_eval_loss": self.metrics.best_eval_loss,
                "best_step": self.metrics.best_step,
                "config": asdict(self.config),
            }
            torch.save(state, save_dir / "training_state.pt")

        self.accelerator.wait_for_everyone()

    def train(
            self,
            train_dataloader: DataLoader,
            val_dataloaders: dict[str, DataLoader],
            optimizer: torch.optim.Optimizer,
            lr_scheduler: torch.optim.lr_scheduler.LRScheduler,
            max_train_steps: int,
    ):

        self.model.train()

        progress_bar = tqdm(
            initial=self.metrics.global_step,
            total=max_train_steps,
            disable=not self.accelerator.is_main_process,
            desc="Training",
            dynamic_ncols=True,
        )

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
                    del outputs
                    # torch.cuda.empty_cache()

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
                    progress_bar.update(1)
                    self.metrics.global_step += 1

                    # Logging
                    batch_loss = self.accelerator.gather_for_metrics(loss.detach()).mean()
                    self.loss_window.append(batch_loss)

                    if self.metrics.global_step % self.config.logging_steps == 0:
                        # torch.stack is safe: deque always contains tensors of identical shape
                        avg_loss = torch.stack(list(self.loss_window)).mean()

                        if self.accelerator.is_main_process:
                            progress_bar.set_postfix(loss=f"{avg_loss.item():.4f}")

                        self.accelerator.log(
                            {
                                "train/loss": avg_loss.item(),
                                "train/learning_rate": lr_scheduler.get_last_lr()[0],
                            },
                            step=self.metrics.global_step,
                        )

                    # Evaluation
                    if (self.metrics.global_step % self.config.eval_steps == 0 and self.metrics.global_step > 0) or self.metrics.global_step >= max_train_steps:
                        for name, val_loader in val_dataloaders.items():
                            eval_loss = self.evaluate(val_loader, name)

                            # Check for best model on primary validation set
                            if name == "val_val":
                                if eval_loss < self.metrics.best_eval_loss:
                                    self.metrics.best_eval_loss = eval_loss
                                    self.metrics.best_step = self.metrics.global_step

                                    if self.accelerator.is_main_process:
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

                    if self.metrics.global_step % self.config.save_steps == 0:
                        self.save_checkpoint(is_best=False, step=self.metrics.global_step)

                if self.metrics.global_step >= max_train_steps:
                    self.save_checkpoint(is_best=False, step=self.metrics.global_step)
                    return
        progress_bar.close()


def main(config: Config):
    """Main training function with improved structure."""
    # Setup
    processed_dataset_dir = Path(NET_SCRATCH_PATH) / "sft_data" / f"round_{config.round_number}"

    set_seed(config.seed or 42)
    setup_logging(config.numeric_log_level)
    os.environ["NCCL_DEBUG"] = "WARN"

    run_name = config.run_name or f"policy-ft-{Path(config.policy_model).name}-new"
    output_dir = Path("/scratch") / "net_scratch" / "models" / "fine_tuned" / "policy" / run_name
    policy_model_dir = Path("/scratch") / "net_scratch" / "models" / "policy"

    resume_from_checkpoint = None
    if config.resume_from_checkpoint and output_dir.exists():
        checkpoint_dirs = [p for p in output_dir.glob("checkpoint-*") if p.is_dir()]
        if checkpoint_dirs:
            # Find the checkpoint with the highest step number
            latest_checkpoint = max(checkpoint_dirs, key=lambda p: int(p.name.split('-')[-1]))
            resume_from_checkpoint = latest_checkpoint
            logger.info(f"Resuming training from checkpoint: {resume_from_checkpoint}")

    # Initialize accelerator
    accelerator_config = {
        "log_with": "wandb" if config.report_to == "wandb" else None,
        "mixed_precision": "bf16" if config.use_bf16 else "fp16",
        "gradient_accumulation_steps": config.gradient_accumulation_steps,
        "dynamo_backend": "no",
    }

    project_config = ProjectConfiguration(
        project_dir=str(output_dir),
        automatic_checkpoint_naming=False,
    )
    accelerator = Accelerator(project_config=project_config, **accelerator_config)

    if accelerator.is_main_process and config.report_to == "wandb":
        accelerator.init_trackers(
            project_name=config.wandb_project,
            config=asdict(config),
            init_kwargs={"wandb": {"name": run_name}},
        )

    if not accelerator.is_main_process:
        logging.getLogger().setLevel(logging.WARNING)

    logger.info("Loading tokenizer and model …")
    try:
        tokenizer_path = processed_dataset_dir / "tokenizer"
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)

        # padding settings (identical to before – the saved vocab already contains SPECIAL_TOKENS)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
        tokenizer.padding_side = "right"

        # model comes from HF / fine-tuned checkpoint exactly as before
        model_source = (
            config.policy_model
            if not config.fine_tuned
            else str(policy_model_dir / config.policy_model)
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_source,
            torch_dtype=torch.bfloat16 if config.use_bf16 else torch.float16,
            trust_remote_code=True,
            attn_implementation=config.attn_implementation,
        )
        model.resize_token_embeddings(len(tokenizer))  # no-op if size unchanged

        # Configure model for training
        model.config.use_cache = False
        if config.gradient_checkpointing:
            model.gradient_checkpointing_enable()
            logger.info(f"Gradient checkpointing enabled: {config.gradient_checkpointing}")

        model.enable_input_require_grads()

        if accelerator.is_main_process and not config.resume_from_checkpoint:
            model.config.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)

    except Exception as e:
        logger.error(f"Failed to load model or tokenizer: {e}")
        raise

    # ------------------------------------------------------------------------ datasets
    logger.info("Opening pre-processed Arrow datasets from %s", processed_dataset_dir)
    ds_root = processed_dataset_dir

    def must_load(name: str):
        p = ds_root / name
        if not p.exists():
            raise FileNotFoundError(f"Required split {name!r} is missing under {ds_root}")
        return datasets.load_from_disk(p)

    train_ds = must_load("train")
    val_task_ds = must_load("val_task") if (ds_root / "val_task").exists() else None
    val_example_ds = must_load("val_example") if (ds_root / "val_example").exists() else None
    val_val_ds = must_load("val_val")

    # set correct format (retained by save_to_disk, but explicit is clearer)
    for ds in (train_ds, val_task_ds, val_example_ds, val_val_ds):
        if ds is not None:
            ds.set_format(type="torch")

    # Log dataset statistics
    logger.info(f"Dataset sizes:")
    logger.info(f"  Train: {len(train_ds)}")
    if val_task_ds:
        logger.info(f"  Val (task): {len(val_task_ds)}")
    if val_example_ds:
        logger.info(f"  Val (example): {len(val_example_ds)}")
    logger.info(f"  Val (external): {len(val_val_ds)}")

    # Create data loaders
    collator = DataCollatorForSFT(tokenizer=tokenizer, max_len=config.max_seq_len)
    num_proc = max(1, os.cpu_count() // accelerator.num_processes - 1)

    train_dataloader = DataLoader(
        train_ds,
        batch_size=config.per_device_train_batch_size,
        shuffle=False,  # Already shuffled
        collate_fn=collator,
        num_workers=num_proc,
        pin_memory=True,
        persistent_workers=True,
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
                num_workers=num_proc,
                pin_memory=True,
                persistent_workers=True,
            )

    # Setup optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
        betas=(0.9, 0.999),
        eps=1e-8,
    )

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

    # Prepare model, optimizer, and dataloader with accelerator
    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler
    )

    # register scheduler so its state is checkpointed
    accelerator.register_for_checkpointing(lr_scheduler)

    max_train_steps //= accelerator.num_processes

    if resume_from_checkpoint:
        logger.info("Loading checkpoint state via Accelerate...")
        accelerator.load_state(str(resume_from_checkpoint))

    if config.torch_compile:
        logger.info("Compiling the model with torch.compile...")
        model = torch.compile(model)

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

    if resume_from_checkpoint:
        logger.info("Loading training state...")
        state = torch.load(resume_from_checkpoint / "training_state.pt")
        trainer.metrics.global_step = state["global_step"]
        trainer.metrics.best_eval_loss = state["best_eval_loss"]
        trainer.metrics.best_step = state["best_step"]

        # This is the number of update steps already completed
        completed_steps = trainer.metrics.global_step
        logger.info(f"Resuming from global step: {completed_steps}")

        # Skip batches in the dataloader that have already been processed
        # This is the key to starting in the middle of an epoch
        train_dataloader = accelerator.skip_first_batches(train_dataloader,
                                                          completed_steps * config.gradient_accumulation_steps)

    # Train
    logger.info(f"Starting training for {max_train_steps} steps...")
    trainer.train(
        train_dataloader=train_dataloader,
        val_dataloaders=prepared_val_dataloaders,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        max_train_steps=max_train_steps,
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
