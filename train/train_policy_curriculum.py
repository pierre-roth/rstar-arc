"""
Curriculum‑aware fine‑tuning script for the ARC **policy** model
===============================================================

*Implements the exact training schedule we discussed.*

Key features
------------
1. **Per‑task split** –for every ARC task we create **train / validation / test** splits *within the training JSONL*:
   * tasks with < `2×(val+test)` lines are dropped.
   * `val_examples_per_task` + `test_examples_per_task` examples are sliced off first, the rest form the training subset.
2. **Task filtering** – tasks whose textual description length exceeds
   `max_task_description_chars` are removed *before* all other filtering.
3. **Curriculum order** – remaining tasks are sorted by code‑complexity, which in this case is defined as the number of
   *non‑comment lines*
4. **Dynamic training loop**
   * Start with the `min_active_tasks` easiest tasks.
   * **Epoch = one full pass** over *only* the training examples of the current
     active set.  A fresh `WeightedTrainer` is spawned every epoch so we can
     swap datasets without side effects.
   * After each epoch we run *pass@k* on **all tasks the model should know**
     (active∪learned).  A task is *learned* when it passes **all** its val
     examples at `pass@k`.
   * If a previously learned task’s failure‑fraction exceeds
     `task_forgetting_threshold`, it is re‑added to the active set.
   * When the active set becomes smaller than `min_active_tasks` (after the
     learned/forgotten update) we *fill it back up* with harder tasks in
     curriculum order *until* the size requirement is met or no tasks remain.
   * **Early‑stop** when either all tasks are learned or **no brand‑new task has
     been added** for `max_stagnation_epochs` consecutive epochs.
5. **End‑of‑run test** – pass@k on the per‑task test slice.

The script can be launched exactly like the old `train_policy.py`, e.g.:
```bash
accelerate launch train/train_policy_curriculum.py --config-file configs/train_policy.yaml
```

--------------------------------------------------------------------
"""

from __future__ import annotations

import logging
import os
import random
import sys
from collections import defaultdict
from dataclasses import asdict
from typing import Any, dict, list, set, Tuple

import torch
import wandb
from datasets import Dataset, load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    set_seed
)

# ---------------- project imports ----------------
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from constants import (
    SFT_SYSTEM_PROMPT,
    SFT_IN_BETWEEN_PROMPT,
    LOCAL_SCRATCH_PATH,
    NET_SCRATCH_PATH,
)
from utils import setup_logging
from train_utils import maybe_peft_wrap
from rstar_deepthink import Config
from rstar_deepthink.arc_task import ARCTask
from rstar_deepthink.arc_task.task_utils import task_to_prompt
from rstar_deepthink.tools.python_tool import remove_markers, run_examples
from train.data_utils import get_code_length

# ---------------- helpers ----------------
logger = logging.getLogger(__name__)


# ------------- preprocessing (prompt→tokens) -------------

def build_prompt(task_json: dict[str, Any], solution: str | None) -> str:
    """Return full training text (prompt + optional solution)."""
    base = (
            SFT_SYSTEM_PROMPT
            + task_to_prompt(ARCTask.from_dict(task_json))
            + SFT_IN_BETWEEN_PROMPT
    )
    return base + (solution if solution is not None else "")


# Tokeniser and collator are created *after* Config is loaded ----------------
class WeightedCollator:
    """Pads dynamically and keeps per‑row weight tensor."""

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, features: list[dict[str, Any]]):
        weights = torch.tensor([f.pop("weight") for f in features], dtype=torch.float32)
        batch = self.tokenizer.pad(features, padding=True, return_tensors="pt")
        if "labels" in batch:
            batch["labels"][batch["labels"] == self.tokenizer.pad_token_id] = -100
        batch["weight"] = weights
        return batch


# ------------- Trainer subclass with weighted loss -------------
class WeightedTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        w = inputs.pop("weight")
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits  # (bs, seq, vocab)
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
        per_token = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        per_token = per_token.view(shift_labels.size())
        active = shift_labels.ne(-100)
        per_seq = (per_token * active).sum(dim=1) / active.sum(dim=1).clamp_min(1)
        loss = (per_seq * w.to(per_seq.device)).sum() / w.sum().clamp_min(1e-8)
        return (loss, outputs) if return_outputs else loss


# ------------- pass@k evaluation -------------

def pass_k_for_examples(
        model,
        tok,
        examples: list[dict[str, Any]],
        cfg: Config,
        device: torch.device,
) -> Tuple[bool, float]:
    """Return (passed_all, fail_fraction)."""
    if not examples:
        return False, 1.0

    temps = cfg.curriculum_eval_temperatures or [0.8]
    per_temp = cfg.pass_k // len(temps)
    leftover = cfg.pass_k % len(temps)

    fails = 0
    for ex in examples:
        task = ARCTask.from_dict(ex["task_json"])
        prompt = build_prompt(ex["task_json"], solution=None)
        passed_once = False
        for i, t in enumerate(temps):
            attempts = per_temp + (1 if i < leftover else 0)
            for _ in range(attempts):
                inputs = tok(prompt, return_tensors="pt").to(device)
                with torch.no_grad():
                    gen_ids = model.generate(
                        **inputs,
                        do_sample=True,
                        temperature=t,
                        top_p=cfg.top_p,
                        top_k=cfg.top_k if cfg.top_k > 0 else None,
                        max_new_tokens=cfg.max_tokens,
                    )
                gen_text = tok.decode(gen_ids[0], skip_special_tokens=True)
                code = remove_markers(gen_text[len(prompt):])  # strip prompt prefix
                err, passed_train, _ = run_examples(task, code)
                if err is None and passed_train:
                    passed_once = True
                    break
            if passed_once:
                break
        if not passed_once:
            fails += 1
    total = len(examples)
    return fails == 0, fails / total


# ------------- main curriculum routine -------------

def main():
    config = Config()
    set_seed(config.seed or 42)
    setup_logging(config.numeric_log_level)
    logger.info("Using model %s", config.policy_model)

    # ---------------- tokenizer / model ----------------
    tok = AutoTokenizer.from_pretrained(config.policy_model, trust_remote_code=True)
    tok.pad_token = tok.pad_token or tok.eos_token
    tok.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        config.policy_model,
        torch_dtype=torch.bfloat16 if config.use_bf16 else torch.float16,
        trust_remote_code=True,
    )
    if model.config.pad_token_id is None:
        model.config.pad_token_id = model.config.eos_token_id
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model.config.use_cache = False
    model = maybe_peft_wrap(model, config)

    # ---------------- data loading & filtering ----------------
    train_path = os.path.join(
        NET_SCRATCH_PATH,
        "sft_data",
        f"round_{config.round_number}",
        "policy_dataset_training.jsonl",
    )
    raw = load_dataset(
        "json",
        data_files={"train": train_path},
        cache_dir=os.path.join(LOCAL_SCRATCH_PATH, ".cache/huggingface/datasets"),
    )["train"]
    logger.info("Loaded %d lines from %s", len(raw), train_path)

    rng = random.Random(config.seed or 42)

    # --- first filter: description length ---
    by_task: dict[str, list[dict]] = defaultdict(list)
    for ex in raw:
        desc_len = len(build_prompt(ex["task_json"], None))
        if desc_len <= config.max_task_description_chars:
            by_task[ex["task_name"]].append(ex)

    # --- second filter: augmentation + min lines for splitting ---
    val_plus_test = config.val_examples_per_task + config.test_examples_per_task
    kept_tasks: dict[str, list[dict]] = {}
    for t, lst in by_task.items():
        if len(lst) >= 2 * val_plus_test:
            kept_tasks[t] = lst
    logger.info("Kept %d tasks after filtering", len(kept_tasks))
    if not kept_tasks:
        raise RuntimeError("No tasks remain after filtering – please loosen thresholds.")

    # --- per‑task split + complexity ---
    train_rows: list[dict] = []
    val_rows_by_task: dict[str, list[dict]] = {}
    test_rows_by_task: dict[str, list[dict]] = {}
    complexity_by_task: dict[str, int] = {}

    for task_name, rows in kept_tasks.items():
        rng.shuffle(rows)
        v = config.val_examples_per_task
        t = config.test_examples_per_task
        val_rows = rows[:v]
        test_rows = rows[v: v + t]
        train_rows_task = rows[v + t:]
        # renormalize weights (sum=1 per task)
        w = 1.0 / len(train_rows_task)
        for r in train_rows_task:
            r["weight"] = w
        # store
        train_rows.extend(train_rows_task)
        val_rows_by_task[task_name] = val_rows
        test_rows_by_task[task_name] = test_rows
        complexity_by_task[task_name] = get_code_length(rows[0]["solution"])

    # --- curriculum order ---
    sorted_tasks = sorted(complexity_by_task, key=lambda x: complexity_by_task[x])
    logger.info("Curriculum order prepared with %d tasks", len(sorted_tasks))

    # --- tokenise *once* all training rows ---
    def encode_row(row):
        text = build_prompt(row["task_json"], row["solution"]) + tok.eos_token
        enc = tok(
            text,
            max_length=config.max_seq_len,
            truncation=True,
            padding=False,
            return_attention_mask=True,
        )
        enc["labels"] = enc["input_ids"].copy()
        enc["weight"] = row["weight"]
        return enc

    logger.info("Tokenising %d training rows …", len(train_rows))
    encoded_rows = [encode_row(r) for r in train_rows]
    task_of_row = [r["task_name"] for r in train_rows]

    # --- collator ---
    collator = WeightedCollator(tok)

    # --- wandb ---
    if config.report_to == "wandb":
        os.environ["WANDB_SILENT"] = "true"
        wandb.init(
            project=config.wandb_project,
            entity=config.wandb_entity,
            name=f"policy-curriculum-{config.policy_model.split('/')[-1]}",
            config=asdict(config),
        )

    # --- curriculum state ---
    active_tasks: set[str] = set(sorted_tasks[: config.min_active_tasks])
    learned_tasks: set[str] = set()
    next_task_idx = config.min_active_tasks
    stagnation_epochs = 0

    def make_dataset(task_set: set[str]) -> Dataset:
        """Return HF Dataset built from encoded rows belonging to tasks in set."""
        rows = [encoded_rows[i] for i, t in enumerate(task_of_row) if t in task_set]
        return Dataset.from_list(rows)

    epoch = 0
    while True:
        epoch += 1
        logger.info("\n===== Curriculum Epoch %d | active=%d learned=%d =====", epoch, len(active_tasks),
                    len(learned_tasks))

        # ----------   FINE‑TUNE for 1 epoch on active set   ----------
        train_ds = make_dataset(active_tasks)
        if len(train_ds) == 0:
            logger.warning("Active set is empty – breaking out.")
            break

        args = TrainingArguments(
            output_dir=os.path.join(NET_SCRATCH_PATH, "models", "tmp_curriculum"),
            per_device_train_batch_size=config.per_device_train_batch_size,
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            learning_rate=config.learning_rate,
            num_train_epochs=1,
            warmup_ratio=config.warmup_ratio,
            lr_scheduler_type=config.lr_scheduler_type,
            logging_steps=config.logging_steps,
            save_strategy="no",
            bf16=config.use_bf16,
            fp16=not config.use_bf16,
            gradient_checkpointing=config.gradient_checkpointing,
            report_to=[],
            remove_unused_columns=False,
        )
        trainer = WeightedTrainer(
            model=model,
            args=args,
            train_dataset=train_ds,
            tokenizer=tok,
            data_collator=collator,
        )
        trainer.train()
        train_loss = trainer.state.log_history[-1]["loss"] if trainer.state.log_history else None
        logger.info("epoch %d train‑loss %.4f", epoch, train_loss or -1)

        # ----------   VALIDATION   ----------
        device = trainer.args.device
        newly_learned: set[str] = set()
        forgotten: set[str] = set()
        to_check = active_tasks | learned_tasks
        for t in to_check:
            passed, fail_frac = pass_k_for_examples(model, tok, val_rows_by_task[t], config, device)
            if passed and t in active_tasks:
                newly_learned.add(t)
            elif (not passed) and t in learned_tasks and fail_frac > config.task_forgetting_threshold:
                forgotten.add(t)
        # update sets
        active_tasks -= newly_learned
        learned_tasks |= newly_learned
        active_tasks |= forgotten
        learned_tasks -= forgotten

        # ----------   REFILL if active < threshold   ----------
        added_brand_new = 0
        while len(active_tasks) < config.min_active_tasks and next_task_idx < len(sorted_tasks):
            candidate = sorted_tasks[next_task_idx]
            next_task_idx += 1
            if candidate in active_tasks or candidate in learned_tasks:
                continue  # skip already seen
            active_tasks.add(candidate)
            added_brand_new += 1
        # stagnation bookkeeping
        if added_brand_new == 0:
            stagnation_epochs += 1
        else:
            stagnation_epochs = 0
        # wandb log
        if config.report_to == "wandb":
            wandb.log({
                "epoch": epoch,
                "train/loss": train_loss,
                "active_tasks": len(active_tasks),
                "learned_tasks": len(learned_tasks),
                "newly_learned": len(newly_learned),
                "forgotten": len(forgotten),
                "brand_new_added": added_brand_new,
                "stagnation_epochs": stagnation_epochs,
            })

        # ----------   TERMINATION   ----------
        if len(learned_tasks) == len(sorted_tasks):
            logger.info("All tasks learned – stopping.")
            break
        if stagnation_epochs >= config.max_stagnation_epochs:
            logger.warning("Stopping – no brand‑new tasks for %d epochs.", stagnation_epochs)
            break

    # ---------------- final test ----------------
    logger.info("\n===== Running final TEST evaluation =====")
    passed_tasks = 0
    for t, test_ex in test_rows_by_task.items():
        passed, _ = pass_k_for_examples(model, tok, test_ex, config, trainer.args.device)
        if passed:
            passed_tasks += 1
    logger.info("Test pass@k: %d / %d (%.2f%%)", passed_tasks, len(test_rows_by_task),
                100 * passed_tasks / len(test_rows_by_task))
    if config.report_to == "wandb":
        wandb.log({"test/passed_tasks": passed_tasks, "test/total_tasks": len(test_rows_by_task)})
        wandb.finish()

    # save final adapter / model
    out_dir = os.path.join(NET_SCRATCH_PATH, "models", "fine_tuned", "policy", "curriculum_final")
    os.makedirs(out_dir, exist_ok=True)
    model.save_pretrained(out_dir)
    tok.save_pretrained(out_dir)
    logger.info("Saved final model to %s", out_dir)


if __name__ == "__main__":
    main()
