import heapq
import os
import sys
from typing import Dict, List

# --- Project Setup ---
# Assuming these imports work in the target environment
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

from constants import NET_SCRATCH_TASK_DATA_DIR
from rstar_deepthink.config import Config
# remove_markers is specific to rstar_deepthink, keep it if solutions have markers
# from rstar_deepthink.tools.python_tool import remove_markers
from utils import setup_logging
from data_utils import *

logger = logging.getLogger(__name__)

# --- Constants ---
NUM_SOLUTIONS_Q_VALUE = 32
NUM_SOLUTIONS_LENGTH = 16
NUM_SOLUTIONS_DIVERSITY = 4
WRITE_BACK_BATCH_SIZE = 1000


# --- Main Script Logic ---
def main(config: Config):
    """Generates validation dataset by curating solutions and reformatting."""
    setup_logging(config.numeric_log_level)
    logger.info("Starting validation dataset generation...")

    # --- Define Paths ---
    sft_data_dir = os.path.join(config.sft_data_dir, f"round_{config.round_number}")
    os.makedirs(sft_data_dir, exist_ok=True)
    solutions_file_path = os.path.join(sft_data_dir, "solutions_evaluation.jsonl")  # Input file
    dataset_file_path = os.path.join(sft_data_dir, "policy_dataset_validation.jsonl")  # Output file
    arc_tasks_base_dir = NET_SCRATCH_TASK_DATA_DIR

    # --- Scan Task Directory ---
    task_name_to_path, directory_structure = load_task_info(arc_tasks_base_dir)

    # --- Load and Group Solutions by Task ---
    solutions_by_task: Dict[str, List[Dict]] = defaultdict(list)
    logger.info(f"Loading solutions from {solutions_file_path}...")
    try:
        with open(solutions_file_path, 'r', encoding='utf-8') as infile:
            for line_num, line in enumerate(infile):
                if not line.strip():
                    continue
                try:
                    data = json.loads(line)
                    task_name = data.get("task_name")

                    if task_name and task_name in task_name_to_path:
                        if "solution_code" in data:
                            # Store the original data dict, keyed by the base task name
                            solutions_by_task[task_name].append(data)
                        else:
                            logger.warning(f"Skipping line {line_num + 1}: Missing 'solution_code'.")
                    elif task_name:
                        logger.warning(
                            f"Task '{task_name}' from line {line_num + 1} not found in task directory scan.")

                except json.JSONDecodeError:
                    logger.warning(f"Skipping invalid JSON in input file line {line_num + 1}: {line.strip()}")
    except FileNotFoundError:
        logger.error(f"Evaluation solutions file not found: {solutions_file_path}")
        sys.exit(1)

    logger.info(f"Loaded solutions for {len(solutions_by_task)} tasks from {solutions_file_path}.")

    # --- Curate Solutions for Each Task ---
    curated_solutions_per_task: Dict[str, List[Dict]] = {}
    logger.info("Curating solutions for each task...")
    for task_name, solutions in solutions_by_task.items():
        logger.debug(f"Curating solutions for task: {task_name} ({len(solutions)} found)")

        # 1. Select top N by Q-value
        solutions_with_q = [(calculate_avg_q(sol), sol) for sol in solutions]
        top_q_solutions = [sol for q, sol in
                           heapq.nlargest(NUM_SOLUTIONS_Q_VALUE, solutions_with_q, key=lambda x: x[0])]
        logger.debug(f"Task {task_name}: Selected {len(top_q_solutions)} solutions based on Q-value.")

        # 2. Select top M by shortest code length (comment-removed)
        for sol in top_q_solutions:
            sol['temp_length'] = get_code_length(sol['solution_code'])
        top_q_solutions.sort(key=lambda x: x['temp_length'])
        shortest_solutions = top_q_solutions[:NUM_SOLUTIONS_LENGTH]
        for sol in top_q_solutions:
            del sol['temp_length']
        logger.debug(f"Task {task_name}: Selected {len(shortest_solutions)} shortest solutions.")

        # 3. Select top K most diverse solutions
        # Precompute clean_code for diversity selection
        for sol in shortest_solutions:
            sol['clean_code'] = remove_comments(sol['solution_code'])
        diverse_solutions = select_diverse_subset(shortest_solutions, NUM_SOLUTIONS_DIVERSITY)
        # Clean up temporary key
        for sol in shortest_solutions:
            if 'clean_code' in sol:
                del sol['clean_code']

        logger.debug(f"Task {task_name}: Selected {len(diverse_solutions)} diverse solutions.")

        if diverse_solutions:
            curated_solutions_per_task[task_name] = diverse_solutions
        else:
            logger.warning(f"Task {task_name}: No solutions remained after curation process.")

    logger.info(f"Finished curation. Found curated solutions for {len(curated_solutions_per_task)} tasks.")

    # --- Prepare Weights ---
    weights_per_task: Dict[str, float] = {}
    for task_name, curated_list in curated_solutions_per_task.items():
        num_curated = len(curated_list)
        if num_curated > 0:
            weights_per_task[task_name] = 1.0 / num_curated
        else:
            logger.warning(f"Task {task_name} has 0 curated solutions, weight calculation skipped.")

    # --- Clear Output File ---
    try:
        with open(dataset_file_path, 'w') as _:
            pass
        logger.info(f"Cleared/Created output dataset file: {dataset_file_path}")
    except IOError as e:
        logger.error(f"Could not open or clear output dataset file {dataset_file_path}: {e}")
        sys.exit(1)

    # --- Generate Final Dataset (Sequential) ---
    logger.info(f"Generating final validation dataset...")
    results_batch_to_write = []
    output_entries = 0
    processed_tasks = 0

    for task_name, curated_solutions in curated_solutions_per_task.items():
        processed_tasks += 1
        logger.debug(f"Processing task {task_name} ({len(curated_solutions)} curated solutions)")

        # --- Load Original Task JSON ---
        original_task_path = task_name_to_path.get(task_name)
        if not original_task_path:
            logger.warning(f"Original task path not found for '{task_name}' during output generation. Skipping task.")
            continue
        original_task_json = load_json_file(original_task_path)
        if not original_task_json:
            logger.warning(f"Could not load original task JSON for '{task_name}'. Skipping task.")
            continue
        # Basic validation of original task structure
        if not isinstance(original_task_json.get('train'), list) or not isinstance(original_task_json.get('test'),
                                                                                   list):
            logger.warning(f"Original task JSON for '{task_name}' has invalid train/test structure. Skipping task.")
            continue

        # --- Get Weight ---
        weight = weights_per_task.get(task_name)
        if weight is None:
            logger.error(
                f"Weight not found for task '{task_name}'. Skipping task.")  # Should not happen if curation worked
            continue

        # --- Create Output Entry for Each Curated Solution ---
        for solution_data in curated_solutions:
            output_data = {
                "task_name": task_name,  # Original task name
                "task_json": original_task_json,  # The *entire* original task JSON
                "solution": solution_data['solution_code'],  # Solution code from curated data
                "weight": weight
            }
            results_batch_to_write.append(output_data)
            output_entries += 1

        # --- Write Batch Periodically ---
        if len(results_batch_to_write) >= WRITE_BACK_BATCH_SIZE:
            write_batch_data(dataset_file_path, results_batch_to_write)
            logger.debug(
                f"Written batch of {len(results_batch_to_write)} entries. Total output entries: {output_entries}")
            results_batch_to_write = []

    # --- Write Final Batch ---
    if results_batch_to_write:
        write_batch_data(dataset_file_path, results_batch_to_write)
        logger.debug(f"Written final batch of {len(results_batch_to_write)} entries.")

    logger.info(f"--- Validation Dataset Creation Summary ---")
    logger.info(f"Processed solutions for {len(solutions_by_task)} tasks from {solutions_file_path}")
    logger.info(f"Generated curated solutions for {len(curated_solutions_per_task)} tasks.")
    logger.info(f"Wrote {output_entries} entries to {dataset_file_path}")


if __name__ == "__main__":
    config_instance = Config()
    config_instance.numeric_log_level = logging.DEBUG

    main(config_instance)
