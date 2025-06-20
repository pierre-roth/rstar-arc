import json
import logging
import os
import random
import sys

# --- Project Setup ---
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

from rstar_deepthink.config import Config
from utils import setup_logging
from data_utils import write_batch_data

logger = logging.getLogger(__name__)

WRITE_BACK_BATCH_SIZE = 100


def _sample_split(num_examples: int, rng: random.Random) -> tuple[int, int]:
    """Determine number of training and test examples for a chunk."""
    if num_examples <= 16:
        return 3, 1
    n_train = rng.choices([2, 3, 4, 5], weights=[0.1, 0.7, 0.1, 0.1])[0]
    n_test = rng.choices([1, 2], weights=[0.9, 0.1])[0]
    return n_train, n_test


def _chunk_examples(task_name: str, examples: list[dict]) -> list[tuple[list, list]]:
    """Split examples into (train, test) chunks deterministically for a task."""
    rng = random.Random(task_name)
    chunks = []
    idx = 0
    total = len(examples)
    while idx < total:
        n_train, n_test = _sample_split(total, rng)
        chunk_size = n_train + n_test
        if idx + chunk_size > total:
            break
        train_ex = examples[idx: idx + n_train]
        test_ex = examples[idx + n_train: idx + chunk_size]
        chunks.append((train_ex, test_ex))
        idx += chunk_size
    return chunks


def main(config: Config):
    setup_logging(config.numeric_log_level)
    logger.info("Creating policy training dataset from BARC data…")

    # sft_data_dir = os.path.join(config.sft_data_dir, f"round_{config.round_number}")
    # os.makedirs(sft_data_dir, exist_ok=True)
    # input_path = os.path.join(sft_data_dir, "barc_converted.jsonl")
    # output_path = os.path.join(sft_data_dir, "policy_dataset_training_barc.jsonl")

    input_path = "/Users/piroth/Downloads/output_dataset.jsonl"
    output_path = "/Users/piroth/Downloads/barc_dataset.jsonl"

    # --- Pass 1: determine total output entries per task ---
    entries_per_task: dict[str, int] = {}
    try:
        with open(input_path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f):
                if not line.strip():
                    continue
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    logger.warning(f"Skipping invalid JSON line {line_num + 1}")
                    continue
                task_name = data.get("task_name")
                examples = data.get("examples")
                if not task_name or not isinstance(examples, list):
                    logger.warning(f"Skipping line {line_num + 1}: missing fields")
                    continue
                chunks = _chunk_examples(task_name, examples)
                if chunks:
                    entries_per_task[task_name] = len(chunks)
                else:
                    logger.warning(f"Task {task_name} produced no valid chunks")
    except FileNotFoundError:
        logger.error(f"BARC input file not found: {input_path}")
        sys.exit(1)

    # Prepare output file
    with open(output_path, "w"):
        pass

    # --- Pass 2: process and write dataset ---
    results_batch = []
    processed = 0
    written = 0
    with open(input_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f):
            processed += 1
            if not line.strip():
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                logger.warning(f"Skipping invalid JSON line {line_num + 1}")
                continue
            task_name = data.get("task_name")
            solution_code = data.get("solution_code")
            examples = data.get("examples")
            if not all([task_name, solution_code, isinstance(examples, list)]):
                logger.warning(f"Skipping line {line_num + 1}: missing fields")
                continue
            chunks = _chunk_examples(task_name, examples)
            total_entries = entries_per_task.get(task_name, len(chunks))
            if total_entries == 0:
                continue
            weight = 1.0 / total_entries
            for train_ex, test_ex in chunks:
                entry = {
                    "task_name": task_name,
                    "task_json": {"train": train_ex, "test": test_ex},
                    "solution": solution_code,
                    "weight": weight,
                }
                results_batch.append(entry)
                written += 1
                if len(results_batch) >= WRITE_BACK_BATCH_SIZE:
                    write_batch_data(output_path, results_batch)
                    results_batch = []

    if results_batch:
        write_batch_data(output_path, results_batch)

    logger.info(f"Processed {processed} lines from {input_path}")
    logger.info(f"Wrote {written} entries to {output_path}")


if __name__ == "__main__":
    config = Config()
    config.numeric_log_level = logging.DEBUG
    main(config)
