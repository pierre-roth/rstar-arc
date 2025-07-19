import logging
import math
import os
import sys
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import datasets
import torch
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration, broadcast, set_seed
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

# -----------------------------------------------------------------------------
# Project‑local imports (assume project root was added to PYTHONPATH beforehand)
# -----------------------------------------------------------------------------
project_root = Path(__file__).parent.parent.absolute()
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from constants import (
    NET_SCRATCH_PATH,
    SFT_IN_BETWEEN_PROMPT,
    SFT_SYSTEM_PROMPT,
    SPECIAL_TOKENS,
)
from rstar_deepthink import Config
from rstar_deepthink.arc_task import ARCTask
from rstar_deepthink.arc_task.task_utils import task_to_prompt
from rstar_deepthink.llms.reward import RewardModelModule
from train_utils import maybe_peft_wrap, renormalize_task_weights
from utils import setup_logging

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Utility dataclasses / helpers
# -----------------------------------------------------------------------------

@dataclass
class TrainingMetrics:
    """Track best accuracy & step while training."""

    best_eval_acc: float = 0.0
    best_step: int = 0
    global_step: int = 0


class DataCollatorForPairwise:
    """Efficient padding for pairwise reward datasets."""

    def __init__(self, tokenizer: PreTrainedTokenizerBase):
        self.tok = tokenizer
        self.pad_id = tokenizer.pad_token_id

    def _pad(self, seqs: List[List[int]]) -> torch.Tensor:
        max_len = max(len(s) for s in seqs)
        out = torch.full((len(seqs), max_len), self.pad_id, dtype=torch.long)
        for i, s in enumerate(seqs):
            out[i, : len(s)] = torch.tensor(s, dtype=torch.long)
        return out

    def __call__(self, batch: List[Dict]):
        # In each element we already have chosen/rejected tokenised lists
        chosen_ids = [ex["chosen_input_ids"] for ex in batch]
        rejected_ids = [ex["rejected_input_ids"] for ex in batch]
        weights = torch.tensor([ex["weight"] for ex in batch], dtype=torch.float32)

        # Pad separately then concat along batch dim: first chosen then rejected
        chosen_pad = self._pad(chosen_ids)
        rejected_pad = self._pad(rejected_ids)
        input_ids = torch.cat([chosen_pad, rejected_pad], dim=0)
        attn_mask = (input_ids != self.pad_id).long()

        return {
            "input_ids": input_ids,
            "attention_mask": attn_mask,
            "weight": weights,
        }


class PairwiseTrainerLoop:
    """Custom Accelerate training loop for reward models."""

    def __init__(
            self,
            config: Config,
            model: PreTrainedModel,
            accelerator: Accelerator,
            output_dir: Path,
    ):
        self.cfg = config
        self.model = model
        self.acc = accelerator
        self.output_dir = output_dir
        self.best_dir = output_dir / "best_model"
        self.metrics = TrainingMetrics()
        self.loss_window: deque[torch.Tensor] = deque(maxlen=config.logging_steps)
        self.best_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------------------
    # Loss + metric helpers
    # ---------------------------------------------------------------------

    @staticmethod
    def _pairwise_loss(rewards_chosen: torch.Tensor, rewards_rejected: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        # L = -log σ(r_c - r_r) weighted
        loss_per_pair = -torch.nn.functional.logsigmoid(rewards_chosen - rewards_rejected)
        weighted = (loss_per_pair * w).sum() / w.sum().clamp_min(1e-8)
        return weighted

    def _compute_batch(
            self, batch: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return (loss, chosen_rewards, rejected_rewards)."""
        w = batch.pop("weight")
        # Forward
        rewards = self.model(**batch)  # (2B,)
        B = rewards.size(0) // 2
        r_c, r_r = rewards[:B], rewards[B:]
        loss = self._pairwise_loss(r_c, r_r, w.to(rewards.device))
        return loss, r_c.detach(), r_r.detach()

    # ---------------------------------------------------------------------
    # Evaluation
    # ---------------------------------------------------------------------

    def evaluate(self, dataloader: DataLoader, prefix: str) -> float:
        self.model.eval()
        all_corr = 0
        all_cnt = 0
        with torch.no_grad():
            for batch in dataloader:
                loss, r_c, r_r = self._compute_batch(batch)
                correct = (r_c > r_r).sum()
                all_corr += self.acc.gather_for_metrics(correct)
                all_cnt += self.acc.gather_for_metrics(torch.tensor(len(r_c), device=correct.device))

        if self.acc.is_main_process:
            acc = all_corr.float().sum() / all_cnt.float().sum().clamp_min(1)
            self.acc.log({f"{prefix}/accuracy": acc.item()}, step=self.metrics.global_step)
        else:
            acc = torch.tensor(0.0, device=self.acc.device)
        self.model.train()
        acc = broadcast(acc, from_process=0)
        return acc.item()

    # ---------------------------------------------------------------------
    # Saving helpers
    # ---------------------------------------------------------------------

    def _save(self, is_best: bool = False, step: int | None = None):
        if is_best:
            tgt = self.best_dir
        elif step is not None:
            tgt = self.output_dir / f"checkpoint-{step}"
        else:
            tgt = self.output_dir
        tgt.mkdir(parents=True, exist_ok=True)
        self.acc.wait_for_everyone()
        self.acc.save_state(str(tgt))

    # ---------------------------------------------------------------------
    # Main training loop
    # ---------------------------------------------------------------------

    def train(
            self,
            train_dl: DataLoader,
            val_dl: DataLoader,
            optimizer: torch.optim.Optimizer,
            scheduler: torch.optim.lr_scheduler.LRScheduler,
            max_steps: int,
    ):
        self.model.train()
        pbar = tqdm(initial=self.metrics.global_step, total=max_steps, disable=not self.acc.is_main_process)
        for _ in range(self.cfg.num_train_epochs):
            for batch in train_dl:
                with self.acc.accumulate(self.model):
                    loss, r_c, r_r = self._compute_batch(batch)
                    self.acc.backward(loss)

                    if self.acc.sync_gradients:
                        if self.cfg.max_grad_norm > 0:
                            self.acc.clip_grad_norm_(self.model.parameters(), self.cfg.max_grad_norm)
                        optimizer.step()
                        scheduler.step()
                        optimizer.zero_grad()

                if self.acc.sync_gradients:
                    pbar.update(1)
                    self.metrics.global_step += 1
                    self.loss_window.append(self.acc.gather_for_metrics(loss.detach()).mean())

                    # logging
                    if self.metrics.global_step % self.cfg.logging_steps == 0:
                        avg_loss = torch.stack(list(self.loss_window)).mean()
                        if self.acc.is_main_process:
                            pbar.set_postfix(loss=f"{avg_loss.item():.4f}")
                        self.acc.log({"train/loss": avg_loss.item()}, step=self.metrics.global_step)

                    # evaluation
                    if (self.metrics.global_step % self.cfg.eval_steps == 0) or self.metrics.global_step >= max_steps:
                        acc_val = self.evaluate(val_dl, "val")
                        if acc_val > self.metrics.best_eval_acc and self.acc.is_main_process:
                            self.metrics.best_eval_acc = acc_val
                            self.metrics.best_step = self.metrics.global_step
                            self.acc.log({"val/best_accuracy": acc_val}, step=self.metrics.global_step)
                            self._save(is_best=True)
                    # save checkpoint
                    if self.metrics.global_step % self.cfg.save_steps == 0:
                        self._save(step=self.metrics.global_step)

                if self.metrics.global_step >= max_steps:
                    self._save(step=self.metrics.global_step)
                    pbar.close()
                    return
        pbar.close()


# -----------------------------------------------------------------------------
# Data preprocessing
# -----------------------------------------------------------------------------

def preprocess_pairwise(batch, *, tok: PreTrainedTokenizerBase, max_len: int):
    out = {
        "chosen_input_ids": [],
        "rejected_input_ids": [],
        "weight": [],
    }
    for tj, prefix, chosen, rejected, w in zip(
            batch["task_json"], batch["prefix"], batch["chosen"], batch["rejected"], batch["weight"]
    ):
        prompt = (
                SFT_SYSTEM_PROMPT
                + task_to_prompt(ARCTask.from_dict(tj))
                + SFT_IN_BETWEEN_PROMPT
                + prefix
        )
        chosen_ids = tok(
            prompt + chosen,
            max_length=max_len,
            truncation=True,
            padding=False,
            add_special_tokens=False,
        )["input_ids"]
        rejected_ids = tok(
            prompt + rejected,
            max_length=max_len,
            truncation=True,
            padding=False,
            add_special_tokens=False,
        )["input_ids"]
        out["chosen_input_ids"].append(chosen_ids)
        out["rejected_input_ids"].append(rejected_ids)
        out["weight"].append(float(w))
    return out


# -----------------------------------------------------------------------------
# Main entry point
# -----------------------------------------------------------------------------

def main(cfg: Config):
    # -------------------------------- setup ---------------------------------
    set_seed(cfg.seed or 42)
    setup_logging(cfg.numeric_log_level)

    run_name = cfg.run_name or f"reward-ft-{Path(cfg.reward_model).name}-new"
    out_dir = Path(NET_SCRATCH_PATH) / "models" / "fine_tuned" / "reward" / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    # accelerator ----------------------------------------------------------------
    acc = Accelerator(
        project_config=ProjectConfiguration(project_dir=str(out_dir), automatic_checkpoint_naming=False),
        log_with="wandb" if cfg.report_to == "wandb" else None,
        mixed_precision="bf16" if cfg.use_bf16 else "fp16",
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        dynamo_backend="no",
    )

    if acc.is_main_process and cfg.report_to == "wandb":
        acc.init_trackers(cfg.wandb_project, config=asdict(cfg), init_kwargs={"wandb": {"name": run_name}})

    if not acc.is_main_process:
        logging.getLogger().setLevel(logging.WARNING)

    # tokenizer ------------------------------------------------------------------
    logger.info("Loading tokenizer …")
    tokenizer = AutoTokenizer.from_pretrained(cfg.reward_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    added = tokenizer.add_special_tokens({"additional_special_tokens": SPECIAL_TOKENS})
    if added:
        logger.info(f"Added {added} special tokens")

    # model ----------------------------------------------------------------------
    logger.info("Loading base model …")
    base_model = AutoModelForCausalLM.from_pretrained(
        cfg.reward_model,
        torch_dtype=torch.bfloat16 if cfg.use_bf16 else torch.float16,
        trust_remote_code=True,
        attn_implementation="torch",
    )
    base_model.resize_token_embeddings(len(tokenizer))
    base_model.config.use_cache = False
    if cfg.gradient_checkpointing:
        base_model.gradient_checkpointing_enable()
    base_model.enable_input_require_grads()

    peft_wrapped = maybe_peft_wrap(base_model, cfg)
    model = RewardModelModule(peft_wrapped, dtype=(torch.bfloat16 if cfg.use_bf16 else torch.float16),
                              dropout=cfg.reward_value_head_dropout)
    model.tokenizer = tokenizer

    # dataset --------------------------------------------------------------------
    logger.info("Loading datasets …")
    proc_root = Path(NET_SCRATCH_PATH) / "sft_data" / f"round_{cfg.round_number}"
    train_path = proc_root / "reward_dataset_training.jsonl"
    val_path = proc_root / "reward_dataset_validation.jsonl"

    raw_train = datasets.load_dataset("json", data_files={"train": str(train_path)})["train"]
    raw_val = datasets.load_dataset("json", data_files={"validation": str(val_path)})["validation"]

    raw_train = renormalize_task_weights(raw_train)
    raw_val = renormalize_task_weights(raw_val)

    num_proc = max(1, os.cpu_count() - 1)
    preprocess_fn = lambda batch: preprocess_pairwise(batch, tok=tokenizer, max_len=cfg.max_seq_len)

    train_ds = raw_train.map(preprocess_fn, batched=True, remove_columns=[c for c in raw_train.column_names if
                                                                          c not in {"chosen_input_ids",
                                                                                    "rejected_input_ids", "weight"}],
                             num_proc=num_proc)
    val_ds = raw_val.map(preprocess_fn, batched=True, remove_columns=[c for c in raw_val.column_names if
                                                                      c not in {"chosen_input_ids",
                                                                                "rejected_input_ids", "weight"}],
                         num_proc=num_proc)

    # dataloaders ----------------------------------------------------------------
    collator = DataCollatorForPairwise(tokenizer)
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.per_device_train_batch_size,
        shuffle=True,
        collate_fn=collator,
        num_workers=num_proc,
        pin_memory=True,
        persistent_workers=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.per_device_eval_batch_size,
        shuffle=False,
        collate_fn=collator,
        num_workers=num_proc,
        pin_memory=True,
        persistent_workers=True,
    )

    # optimizer + scheduler ------------------------------------------------------
    optimizer = AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay, betas=(0.9, 0.999),
                      eps=1e-8)
    updates_per_epoch = math.ceil(len(train_loader) / cfg.gradient_accumulation_steps)
    max_steps = updates_per_epoch * cfg.num_train_epochs
    scheduler = get_scheduler(cfg.lr_scheduler_type, optimizer=optimizer,
                              num_warmup_steps=int(max_steps * cfg.warmup_ratio), num_training_steps=max_steps)

    # prepare with accelerator ---------------------------------------------------
    model, optimizer, train_loader, scheduler = acc.prepare(model, optimizer, train_loader, scheduler)
    acc.register_for_checkpointing(scheduler)

    # trainer --------------------------------------------------------------------
    loop = PairwiseTrainerLoop(cfg, model, acc, out_dir)
    logger.info(f"Starting training for {max_steps} steps …")
    loop.train(train_loader, val_loader, optimizer, scheduler, max_steps)

    if cfg.report_to == "wandb":
        acc.end_training()


if __name__ == "__main__":
    cfg = Config()
    try:
        main(cfg)
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        raise
