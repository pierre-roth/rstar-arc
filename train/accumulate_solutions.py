import json
import os
import re
import sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from rstar_deepthink.config import Config
from rstar_deepthink.tools.python_tool import remove_markers


def rename_solve_function(code: str, task_name: str) -> str:
    """Rename the solve function to be task specific."""
    pattern = r"def\s+solve\s*\("
    return re.sub(pattern, f"def solve_{task_name}(", code, count=1)


def main():
    config = Config()

    sft_data_dir = os.path.join(config.sft_data_dir, f"round_{config.round_number}")
    input_path = os.path.join(sft_data_dir, "solutions_training_augmented.jsonl")
    output_path = os.path.join(sft_data_dir, "aggregated_solutions.py")

    shortest: dict[str, tuple[str, int]] = {}
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"{input_path} not found")

    # Load all (task, code) pairs into a set for deduplication
    code_set: set[tuple[str, str]] = set()
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue

            task = data.get("original_task_name") or data.get("task_name")
            code = data.get("solution_code", "")
            if task and code:
                code_set.add((task, code))

    # Keep only the shortest solution per task
    for task, code in code_set:
        cleaned = remove_markers(code)
        length = len(cleaned)
        if task not in shortest or length < shortest[task][1]:
            shortest[task] = (cleaned, length)

    with open(output_path, "w", encoding="utf-8") as out:
        for task, (code, _) in sorted(shortest.items()):
            renamed = rename_solve_function(code, task)
            out.write(renamed.rstrip() + "\n\n")


if __name__ == "__main__":
    main()
