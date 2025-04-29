#!/usr/bin/env python3
"""
Combine non-augmentable solutions from round 0 to round 1.

This script reads the solutions_training.jsonl file for round 0, filters out
entries where the task is in the 'training' folder, and appends the remaining
solutions to the round 1 solutions_training.jsonl file. Uses the Config class
for all directory paths.
"""

import json
import os
import sys
import logging

# --- Project Setup ---
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

from rstar_deepthink.config import Config
from utils import setup_logging
from train.data_utils import load_task_info

logger = logging.getLogger(__name__)


def main():
    # Load default configuration
    config = Config()
    setup_logging(config.numeric_log_level)

    # Hardcoded source and destination rounds
    source_round = 0
    dest_round = 1

    # Paths to solution files
    source_file = os.path.join(
        config.sft_data_dir, f"round_{source_round}", "solutions_training.jsonl"
    )
    dest_dir = os.path.join(
        config.sft_data_dir, f"round_{dest_round}"
    )
    dest_file = os.path.join(dest_dir, "solutions_training.jsonl")

    # Verify source file exists
    if not os.path.exists(source_file):
        sys.exit(f"Source solutions file not found: {source_file}")

    # Ensure destination directory exists
    os.makedirs(dest_dir, exist_ok=True)

    # Load task directory structure to identify training tasks
    _, dir_structure = load_task_info(config.task_dir)
    training_tasks = set(dir_structure.get("training", []))

    appended = 0
    # Read source and append to destination
    with open(source_file, 'r', encoding='utf-8') as sf, \
            open(dest_file, 'a', encoding='utf-8') as df:
        for lineno, line in enumerate(sf, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                print(f"Skipping invalid JSON at line {lineno}", file=sys.stderr)
                continue

            task_name = record.get("task_name")
            # If task is not in training folder, append the solution
            if task_name and task_name not in training_tasks:
                df.write(line)
                appended += 1

    print(f"Appended {appended} non-training solutions from round {source_round} to round {dest_round}.")


if __name__ == '__main__':
    main()
