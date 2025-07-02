"""
Filter a JSON‑Lines task file, keeping only rows whose `solution_code` passes
`verify_prefixes_and_code`, renormalise per‑task weights with
`renormalize_task_weights`, and write the **result back to a JSONL file**.

Usage
-----
```bash
python filter_tasks_to_jsonl.py \
       --input  /path/to/input_dataset.jsonl \
       --output /path/to/filtered_dataset.jsonl \
       --max-workers 16
```
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Tuple

from datasets import Dataset  # Used only to leverage renormalize_task_weights
# Assume this helper is available in the environment as requested.

# ------------------- project imports -------------------
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from rstar_deepthink.tools import verify_prefixes_and_code
from train_utils import renormalize_task_weights

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Verification helper
# ---------------------------------------------------------------------------

def _verify_task(task: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
    """Return (passes?, task) after running prefix verification."""
    try:
        code = task.get("solution_code")
        examples = task.get("examples")
        if not code or not examples:
            return False, task

        inputs = [ex["input"] for ex in examples]
        expected = [ex["output"] for ex in examples]

        success, _, err_full, passed_full, _ = verify_prefixes_and_code(
            code, inputs, expected
        )
        return bool(success and not err_full and passed_full), task
    except Exception as exc:  # noqa: BLE001
        logger.debug("Verification error: %s", exc, exc_info=True)
        return False, task


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def _load_jsonl(path: str) -> List[Dict[str, Any]]:
    """Read *path* and return list of JSON objects (one per line)."""
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _write_jsonl(path: str, rows: List[Dict[str, Any]]) -> None:
    """Write *rows* to *path* in JSON‑Lines format."""
    with open(path, "w", encoding="utf-8") as fh:
        for row in rows:
            json.dump(row, fh, ensure_ascii=False)
            fh.write("\n")


# ---------------------------------------------------------------------------
# CLI parsing & main driver
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:  # noqa: D401
    parser = argparse.ArgumentParser(description="Filter tasks and emit JSONL")
    parser.add_argument("--input", required=True, help="Input JSONL path")
    parser.add_argument("--output", required=True, help="Output JSONL path")
    parser.add_argument("--max-workers", type=int, default=os.cpu_count() or 4)
    return parser.parse_args()


def main() -> None:  # pragma: no cover
    args = _parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    logger.info("Loading tasks from %s …", args.input)
    tasks = _load_jsonl(args.input)
    logger.info("Loaded %d total tasks", len(tasks))

    passing_tasks: List[Dict[str, Any]] = []

    with ThreadPoolExecutor(max_workers=args.max_workers) as pool:
        futures = [pool.submit(_verify_task, t) for t in tasks]
        for fut in as_completed(futures):
            passes, task = fut.result()
            if passes:
                task.setdefault("weight", 1.0)  # ensure numeric weight
                passing_tasks.append(task)

    kept = len(passing_tasks)
    logger.info(
        "Verification complete → kept %d / %d tasks (%.2f%%)",
        kept,
        len(tasks),
        100 * kept / max(1, len(tasks)),
    )

    # Renormalise weights using the provided helper via Dataset
    ds = Dataset.from_list(passing_tasks)
    ds = renormalize_task_weights(ds)

    # Convert back to list of dicts
    renormed_tasks = [dict(row) for row in ds]

    logger.info("Writing filtered tasks to %s …", args.output)
    _write_jsonl(args.output, renormed_tasks)
    logger.info("Done – wrote %d tasks.", len(renormed_tasks))


if __name__ == "__main__":  # pragma: no cover
    main()
