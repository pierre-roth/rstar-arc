import json
import os
import sys

# Add the project root to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)

from constants import NET_SCRATCH_PATH


def calculate_coverage(round_num: int):
    sft_path = os.path.join(NET_SCRATCH_PATH, "sft_data", f"round_{round_num}", "raw.jsonl")
    task_path = os.path.join(NET_SCRATCH_PATH, "task_data")

    task_names = set()

    # Collect all task names from the task data
    with open(sft_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                solution = json.loads(line)
                task_names.add(solution["task_name"] + ".json")

    # per folder covered tasks
    folders = {}

    # go through every directory in task path and calculate coverage for each one
    for dir_name in os.listdir(task_path):
        dir_path = os.path.join(task_path, dir_name)
        if os.path.isdir(dir_path):
            total_tasks = 0
            covered_tasks = 0

            folders[dir_name] = set()
            for task_file in os.listdir(dir_path):
                if task_file.endswith(".json"):
                    total_tasks += 1
                    if task_file in task_names:
                        folders[dir_name].add(task_file)
                        covered_tasks += 1
            print(f"{dir_name}: {covered_tasks}/{total_tasks} = {covered_tasks / total_tasks:.2%} coverage")

    print(folders.get("training", set()).difference(folders.get("training2", set())))
    print(f"Total tasks: {len(task_names)}")


if __name__ == "__main__":
    # Example usage
    round_number = int(input("Round number: "))
    calculate_coverage(round_number)
