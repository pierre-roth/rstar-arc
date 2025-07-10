#!/usr/bin/env python
# prepare_dataset.py
"""
One-shot ARC SFT dataset preparation.

• Reads the raw JSON files exactly as the training script does.
• Tokenises, filters, splits (task-level + example-level), weight-renormalises,
  shuffles, and finally saves each split to `Dataset.save_to_disk()`.
• The resulting directories can be re-opened with: ds = datasets.load_from_disk("…/processed/train")
  and plugged straight into a DataLoader.

Typical usage
-------------
python prepare_dataset.py \
    --train_json /scratch/net_scratch/sft_data/round_0/train.jsonl \
    --val_json   /scratch/net_scratch/sft_data/round_0/val.jsonl \
    --policy_model meta-llama/Llama-3-8b \
    --out_dir    /scratch/net_scratch/processed_sft/round_0 \
    --max_seq_len 4096 \
    --num_proc 32

You only need to run this once per dataset / max_seq_len / tokenizer version.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
from pathlib import Path
from typing import Tuple
import sys

import datasets
from transformers import AutoTokenizer
from tqdm.auto import tqdm

project_root = Path(__file__).parent.parent.absolute()
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from constants import (  
    SFT_IN_BETWEEN_PROMPT,
    SFT_SYSTEM_PROMPT,
    SPECIAL_TOKENS,
)
# --- project-specific bits ----------------------------------------------------
from rstar_deepthink import Config  
from rstar_deepthink.arc_task import ARCTask  
from rstar_deepthink.arc_task.task_utils import task_to_prompt  
from train_utils import renormalize_task_weights  

# -----------------------------------------------------------------------------


_LOG = logging.getLogger(__name__)
datasets.logging.enable_progress_bar()


# -----------------------------------------------------------------------------#
#                             Helper functions                                 #
# -----------------------------------------------------------------------------#
def tokenize_example(
        batch: dict,
        tokenizer: AutoTokenizer,
        max_seq_len: int,
) -> dict:
    """Tokenise prompt+completion pairs and build labels/masks."""
    prompts = [
        SFT_SYSTEM_PROMPT + task_to_prompt(ARCTask.from_dict(j)) + SFT_IN_BETWEEN_PROMPT
        for j in batch["task_json"]
    ]
    completions = batch["solution"]

    p_tok = tokenizer(prompts, add_special_tokens=False, padding=False)["input_ids"]
    c_tok = tokenizer(completions, add_special_tokens=False, padding=False)["input_ids"]

    input_ids, labels, lengths = [], [], []
    for p, c in zip(p_tok, c_tok):
        c = c + [tokenizer.eos_token_id]  # force EOS
        full = p + c
        p_len = len(p)
        if len(full) > max_seq_len:
            if p_len < max_seq_len:  # truncate completion only
                full = full[:max_seq_len]
                p_len = p_len
            else:  # truncate prompt too
                full = p[:max_seq_len]
                p_len = max_seq_len

        lbl = full.copy()
        lbl[:p_len] = [-100] * p_len  # mask prompt
        input_ids.append(full)
        labels.append(lbl)
        lengths.append(len(full))

    return {"input_ids": input_ids, "labels": labels, "length": lengths}


def create_validation_splits(
        dataset: datasets.Dataset,
        cfg: Config,
        seed: int = 42,
) -> Tuple[datasets.Dataset, datasets.Dataset | None, datasets.Dataset | None]:
    """
    Identical logic to training script:
        • task-level hold-out   (val_task_ds)
        • per-task example hold-out (val_example_ds)
        • remainder             (train_ds)
    """
    rng = random.Random(seed)
    task_to_idx: dict[str, list[int]] = {}
    for i, ex in enumerate(tqdm(dataset, desc="Indexing tasks")):
        task_to_idx.setdefault(ex["task_name"], []).append(i)

    val_task_idx = []
    if cfg.task_validation_fraction > 0:
        tasks = list(task_to_idx.keys())
        rng.shuffle(tasks)
        n = int(len(tasks) * cfg.task_validation_fraction)
        val_tasks = set(tasks[:n])
        for t in val_tasks:
            val_task_idx.extend(task_to_idx[t])
    else:
        val_tasks = set()

    val_ex_idx, train_idx = [], []
    for t, indices in tqdm(task_to_idx.items(), desc="Splitting examples by task"):
        if t in val_tasks:
            continue

        if (
                cfg.example_validation_num > 0
                and len(indices) >= cfg.example_validation_threshold
        ):
            indices_ = list(indices)
            rng.shuffle(indices_)
            n_candidates = min(
                cfg.example_validation_num
                * (rng.random() < cfg.example_validation_probability),
                len(indices_) // 2,
            )
            val_ex_idx.extend(indices_[:n_candidates])
            train_idx.extend(indices_[n_candidates:])
        else:
            train_idx.extend(indices)

    train_ds = dataset.select(train_idx)
    val_task_ds = dataset.select(val_task_idx) if val_task_idx else None
    val_example_ds = dataset.select(val_ex_idx) if val_ex_idx else None
    return train_ds, val_task_ds, val_example_ds


# -----------------------------------------------------------------------------#
#                                Main                                          #
# -----------------------------------------------------------------------------#
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--train_json", required=True, help="Path to ARC train .jsonl")
    p.add_argument("--val_json", required=True, help="Path to ARC val .jsonl")
    p.add_argument("--policy_model", required=True, help="HF hub id or local dir")
    p.add_argument("--out_dir", required=True, help="Folder where splits are saved")
    p.add_argument("--max_seq_len", type=int, default=4096)
    p.add_argument("--num_proc", type=int, default=max(1, os.cpu_count() // 2))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--log_level", default="INFO")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=args.log_level, format="%(asctime)s %(levelname)s: %(message)s")

    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ tokeniser
    _LOG.info("Loading tokenizer %s", args.policy_model)
    tok = AutoTokenizer.from_pretrained(args.policy_model, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
        tok.pad_token_id = tok.eos_token_id
    tok.padding_side = "right"
    tok.add_special_tokens({"additional_special_tokens": SPECIAL_TOKENS})

    # Save a *frozen* copy alongside datasets so future runs use identical vocab
    tok.save_pretrained(out_dir / "tokenizer")

    # ------------------------------------------------------------------ load raw
    _LOG.info("Loading raw datasets …")
    train_raw = datasets.load_dataset("json", data_files={"train": args.train_json})["train"]
    val_raw = datasets.load_dataset("json", data_files={"validation": args.val_json})["validation"]

    def filter_bad_examples(ex: dict) -> bool:
        """Return True for rows that should be kept.

        Current heuristic drops placeholder rows whose solution string contains
        the word "dummy".  It preserves all other rows, including those with
        a non-empty solution that legitimately contains substrings such as
        "dummies" etc.  You may want to tighten this further, but for now we
        merely implement the correctly-spelled helper so static analysers and
        readers are not confused.
        """

        return "dummy" not in ex["solution"]

    # Apply the filter to both splits.
    train_raw = train_raw.filter(filter_bad_examples, num_proc=args.num_proc)
    val_raw = val_raw.filter(filter_bad_examples, num_proc=args.num_proc)

    # Keep non-text columns used later
    keep_cols = ["weight", "task_name", "task_json", "solution"]

    # ------------------------------------------------------------------ tokenise
    _LOG.info("Tokenising & building labels … (%d + %d examples)", len(train_raw), len(val_raw))
    fn = lambda b: tokenize_example(b, tok, args.max_seq_len)

    train_tok = train_raw.map(
        fn,
        batched=True,
        batch_size=1000,
        num_proc=args.num_proc,
        remove_columns=[c for c in train_raw.column_names if c not in keep_cols],
        desc="train-tokenise",
    )
    val_tok = val_raw.map(
        fn,
        batched=True,
        batch_size=1000,
        num_proc=args.num_proc,
        remove_columns=[c for c in val_raw.column_names if c not in keep_cols],
        desc="val-tokenise",
    )

    # ------------------------------------------------------------------ filter
    # _LOG.info("Filtering sequences > %d tokens …", args.max_seq_len)
    # train_tok = train_tok.filter(lambda x: x["length"] <= args.max_seq_len, num_proc=args.num_proc).remove_columns("length")
    # val_tok = val_tok.filter(lambda x: x["length"] <= args.max_seq_len, num_proc=args.num_proc).remove_columns("length")

    # ------------------------------------------------------------------ splits
    cfg = Config()  # use same split hyper-params
    cfg.seed = args.seed
    train_ds, val_task_ds, val_example_ds = create_validation_splits(train_tok, cfg, seed=args.seed)

    # external validation (unchanged) -> val_val_ds
    val_val_ds = val_tok

    # ------------------------------------------------------------------ weights
    train_ds = renormalize_task_weights(train_ds, num_proc=args.num_proc)
    if val_task_ds:
        val_task_ds = renormalize_task_weights(val_task_ds, num_proc=args.num_proc)
    if val_example_ds:
        val_example_ds = renormalize_task_weights(val_example_ds, num_proc=args.num_proc)
    val_val_ds = renormalize_task_weights(val_val_ds, num_proc=args.num_proc)

    # ------------------------------------------------------------------ shuffle & format
    train_ds = train_ds.shuffle(seed=args.seed)
    for ds in (train_ds, val_task_ds, val_example_ds, val_val_ds):
        if ds is not None:
            ds.set_format(type="torch")

    # ------------------------------------------------------------------ save
    def _save(name: str, ds: datasets.Dataset | None):
        if ds is None:
            return
        path = out_dir / name
        _LOG.info("Saving %s split to %s (rows=%d)…", name, path, len(ds))
        ds.save_to_disk(path)  # zero-copy Arrow serialization
        (path / "meta.json").write_text(
            json.dumps(
                {
                    "rows": len(ds),
                    "max_seq_len": args.max_seq_len,
                    "tokenizer": str(out_dir / "tokenizer"),
                },
                indent=2,
            )
        )

    _save("train", train_ds)
    _save("val_task", val_task_ds)
    _save("val_example", val_example_ds)
    _save("val_val", val_val_ds)

    _LOG.info("✅ All splits written under %s", out_dir)
    _LOG.info("   Re-load with:  ds = datasets.load_from_disk('%s/train')", out_dir)

    # ------------------------------------------------------------------ done
    _LOG.info("Finished.")


if __name__ == "__main__":
    main()
