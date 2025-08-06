import json
import os
import sys

# Add the project root to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)

from constants import NET_SCRATCH_SFT_DATA_DIR, NET_SCRATCH_TASK_DATA_DIR


def calculate_coverage(round_num: int):
    print(f"Calculating coverage for round {round_num}")

    solutions_path_training = os.path.join(NET_SCRATCH_SFT_DATA_DIR, f"round_{round_num}", "solutions_training.jsonl")
    solutions_path_evaluation = os.path.join(NET_SCRATCH_SFT_DATA_DIR, f"round_{round_num}", "solutions_evaluation.jsonl")
    task_path = NET_SCRATCH_TASK_DATA_DIR

    task_names = set()

    # Collect all task names from the solution files
    for path in [solutions_path_training, solutions_path_evaluation]:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        solution = json.loads(line)
                        task_names.add(solution["task_name"] + ".json")

    folders = {}
    solved = {}

    # Recursively walk through the task path
    for dirpath, dirnames, filenames in os.walk(task_path):
        # Skip folders that contain sub-folders, process only leaf directories
        if not dirnames:
            relative_dir_path = os.path.relpath(dirpath, task_path)
            total_tasks = 0
            covered_tasks = 0

            folders[relative_dir_path] = set()
            solved[relative_dir_path] = set()

            for task_file in filenames:
                if task_file.endswith(".json"):
                    total_tasks += 1
                    folders[relative_dir_path].add(task_file)
                    if task_file in task_names:
                        solved[relative_dir_path].add(task_file)
                        covered_tasks += 1

            if total_tasks > 0:
                print(f"{relative_dir_path}: {covered_tasks}/{total_tasks} = {covered_tasks / total_tasks:.2%} coverage")

    print(f"Total unique tasks: {len(task_names)}")
    return folders, solved


if __name__ == "__main__":
    # Example usage
    from_round_number = int(input("From round number: "))
    to_round_number = int(input("To round number: "))

    coverages = []
    for i in range(from_round_number, to_round_number + 1):
        coverages.append(calculate_coverage(i))
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
