import json
import os
import sys

# Add the project root to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)

from constants import NET_SCRATCH_SFT_DATA_DIR, NET_SCRATCH_TASK_DATA_DIR


def calculate_coverage(round_num: int):
    solutions_path_training = os.path.join(NET_SCRATCH_SFT_DATA_DIR, f"round_{round_num}", "solutions_training.jsonl")
    solutions_path_evaluation = os.path.join(NET_SCRATCH_SFT_DATA_DIR, f"round_{round_num}",
                                             "solutions_evaluation.jsonl")
    task_path = NET_SCRATCH_TASK_DATA_DIR

    task_names = set()

    # Collect all task names from the task data
    with open(solutions_path_training, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                solution = json.loads(line)
                task_names.add(solution["task_name"] + ".json")

    with open(solutions_path_evaluation, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                solution = json.loads(line)
                task_names.add(solution["task_name"] + ".json")

    # per folder covered tasks
    folders = {}
    solved = {}

    # go through every directory in task path and calculate coverage for each one
    for dir_name in os.listdir(task_path):
        dir_path = os.path.join(task_path, dir_name)
        if os.path.isdir(dir_path):
            total_tasks = 0
            covered_tasks = 0

            folders[dir_name] = set()
            solved[dir_name] = set()
            for task_file in os.listdir(dir_path):
                if task_file.endswith(".json"):
                    total_tasks += 1
                    folders[dir_name].add(task_file)
                    if task_file in task_names:
                        solved[dir_name].add(task_file)
                        covered_tasks += 1
            print(f"{dir_name}: {covered_tasks}/{total_tasks} = {covered_tasks / total_tasks:.2%} coverage")

    print(f"Total unique tasks: {len(task_names)}")
    return folders, solved


if __name__ == "__main__":
    # Example usage
    round_number = int(input("Round number: "))

    coverages = []
    print(f"Calculating coverage for round {round_number}: ")
    coverages.append(calculate_coverage(round_number))
    print("\n")

    all_tasks = {}
    all_solved = {}
    for fol, sol in coverages:
        for folder, tasks in sol.items():
            if folder not in all_solved:
                all_solved[folder] = set()
            all_solved[folder].update(tasks)
        for folder, tasks in fol.items():
            if folder not in all_tasks:
                all_tasks[folder] = set()
            all_tasks[folder].update(tasks)

    print("Ensemble coverage:")
    total_unique = set()
    for folder, tasks in all_solved.items():
        total_unique.update(tasks)
        print(f"  - {folder}: {len(tasks)} unique tasks")
    print(f"Total unique tasks: {len(total_unique)}")

    # ask whether to print all tasks
    print_all = input("Print all tasks? (y/n): ").strip().lower()
    if print_all == "y":
        print(f"Solved training tasks: {list(all_solved['training'])}")
        print(f"Unsolved training tasks: {list(all_tasks['training'] - all_solved['training'])}")
