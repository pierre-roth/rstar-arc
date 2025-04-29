import os
import sys

# --- Project Setup ---
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

# Assuming these imports work in the target environment
from constants import NET_SCRATCH_TASK_DATA_DIR
from rstar_deepthink.config import Config
from utils import setup_logging
from data_utils import *

logger = logging.getLogger(__name__)

# --- Constants ---
WRITE_BACK_BATCH_SIZE = 100


# --- Main Orchestration ---
def main(config: Config):
    """Processes augmented data and expands it into the final dataset format with correct weighting."""
    setup_logging(config.numeric_log_level)
    logger.info("Starting final dataset creation from augmented data...")

    # --- Define Paths ---
    sft_data_dir = os.path.join(config.sft_data_dir, f"round_{config.round_number}")
    os.makedirs(sft_data_dir, exist_ok=True)
    augmented_file_path = os.path.join(sft_data_dir, "solutions_training_augmented.jsonl")
    dataset_file_path = os.path.join(sft_data_dir, "policy_dataset_training.jsonl")
    arc_tasks_base_dir = NET_SCRATCH_TASK_DATA_DIR

    # --- Scan Task Directory (Needed for Pass 1) ---
    task_name_to_path, directory_structure = load_task_info(arc_tasks_base_dir)

    # --- Pass 1: Calculate Total Output Entries per Original Task ---
    logger.info(f"Calculating weights: Reading {augmented_file_path} to count total output entries per task...")
    total_output_entries_per_task = defaultdict(int)
    total_lines_pass1 = 0
    processed_task_names = set()

    try:
        with open(augmented_file_path, 'r', encoding='utf-8') as infile:
            for line_num, line in enumerate(infile):
                total_lines_pass1 += 1
                if not line.strip():
                    continue
                try:
                    data = json.loads(line)
                    original_task_name = data.get("original_task_name")
                    augmented_examples = data.get("examples")

                    if not original_task_name or not isinstance(augmented_examples, list):
                        logger.warning(f"Skipping line {line_num + 1} in Pass 1 due to missing/invalid fields.")
                        continue

                    processed_task_names.add(original_task_name)

                    # --- Load Original Task Info for chunk size ---
                    original_task_path = task_name_to_path.get(original_task_name)
                    if not original_task_path:
                        logger.warning(
                            f"Original task path not found for '{original_task_name}' during Pass 1. Cannot count entries accurately for this line.")
                        continue
                    original_task_json = load_json_file(original_task_path)
                    if not original_task_json:
                        logger.warning(
                            f"Could not load original task JSON for '{original_task_name}' during Pass 1. Cannot count entries accurately for this line.")
                        continue

                    # --- Determine Chunk Size ---
                    n_train = len(original_task_json.get('train', []))
                    n_test = len(original_task_json.get('test', []))
                    chunk_size = n_train + n_test

                    # --- Count Complete Chunks ---
                    num_complete_chunks = len(augmented_examples) // chunk_size
                    total_output_entries_per_task[original_task_name] += num_complete_chunks + 1

                except json.JSONDecodeError:
                    logger.warning(f"Skipping invalid JSON line {line_num + 1} during weight calculation.")

    except FileNotFoundError:
        logger.error(f"Augmented input file not found for weight calculation: {augmented_file_path}")
        sys.exit(1)

    if total_lines_pass1 == 0:
        logger.error(f"No valid lines found in {augmented_file_path}. Cannot proceed.")
        sys.exit(1)

    logger.info(
        f"Calculated total output entries for {len(total_output_entries_per_task)} unique original tasks (encountered {len(processed_task_names)}).")

    # --- Clear Output File ---
    try:
        with open(dataset_file_path, 'w') as _:
            pass
        logger.info(f"Cleared/Created output dataset file: {dataset_file_path}")
    except IOError as e:
        logger.error(f"Could not open or clear output dataset file {dataset_file_path}: {e}")
        sys.exit(1)

    # --- Pass 2: Process and Write Final Dataset ---
    logger.info(f"Processing augmented data and writing to {dataset_file_path}...")
    results_batch_to_write = []
    processed_lines_pass2 = 0
    output_entries = 0
    try:
        with open(augmented_file_path, 'r', encoding='utf-8') as infile:
            for line_num, line in enumerate(infile):
                processed_lines_pass2 += 1
                if not line.strip():
                    continue
                try:
                    data = json.loads(line)
                    original_task_name = data.get("original_task_name")
                    solution_code = data.get("solution_code")
                    augmented_examples = data.get("examples")

                    if not all([original_task_name, solution_code, isinstance(augmented_examples, list)]):
                        logger.warning(f"Skipping line {line_num + 1} in Pass 2 due to missing/invalid fields.")
                        continue

                except json.JSONDecodeError:
                    logger.warning(f"Skipping invalid JSON line {line_num + 1}: {line.strip()}")
                    continue

                # --- Get Total Entries for Weight ---
                total_entries = total_output_entries_per_task.get(original_task_name)
                if not total_entries or total_entries == 0:
                    logger.warning(
                        f"Task '{original_task_name}' has zero total output entries calculated. Skipping line {line_num + 1}.")
                    continue
                weight = 1.0 / total_entries

                # --- Load Original Task Info Again for Chunk Size ---
                # (Could potentially cache this, but loading again ensures consistency)
                original_task_path = task_name_to_path.get(original_task_name)
                if not original_task_path:
                    continue  # Already warned in Pass 1
                original_task_json = load_json_file(original_task_path)
                if not original_task_json:
                    continue  # Already warned in Pass 1

                # --- Determine Chunk Size Again ---
                n_train = len(original_task_json.get('train', []))
                n_test = len(original_task_json.get('test', []))
                chunk_size = n_train + n_test

                # --- Process and Chunk Examples ---
                for i in range(0, len(augmented_examples), chunk_size):
                    chunk = augmented_examples[i: i + chunk_size]

                    if len(chunk) == chunk_size:  # Only process complete chunks
                        new_train_examples = chunk[:n_train]
                        new_test_examples = chunk[n_train:]

                        # Ensure test examples format (list)
                        if 0 < n_test != len(new_test_examples):
                            logger.warning(
                                f"Chunking mismatch for test examples in task {original_task_name}, chunk {i // chunk_size}. Expected {n_test}, got {len(new_test_examples)}. Skipping this chunk.")
                            continue
                        if n_test == 1 and not isinstance(new_test_examples, list):
                            new_test_examples = [new_test_examples]  # Wrap single test example in list

                        # Construct final output format
                        output_data = {
                            "task_name": original_task_name,  # Use original task name
                            "task_json": {
                                "train": new_train_examples,
                                "test": new_test_examples
                            },
                            "solution": solution_code,
                            "weight": weight
                        }
                        results_batch_to_write.append(output_data)
                        output_entries += 1

                output_data = {
                    "task_name": original_task_name,  # Use original task name
                    "task_json": original_task_json,
                    "solution": solution_code,
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

    except FileNotFoundError:
        logger.error(f"Augmented input file not found during Pass 2: {augmented_file_path}")
    except Exception as e:
        logger.error(f"An unexpected error occurred during Pass 2 processing: {e}", exc_info=True)

    # --- Write Final Batch ---
    if results_batch_to_write:
        write_batch_data(dataset_file_path, results_batch_to_write)
        logger.debug(f"Written final batch of {len(results_batch_to_write)} entries.")

    logger.info(f"--- Dataset Creation Summary ---")
    logger.info(f"Processed {processed_lines_pass2} lines from {augmented_file_path}")
    logger.info(f"Wrote {output_entries} entries to {dataset_file_path}")


if __name__ == "__main__":
    # Instantiate config directly
    config_instance = Config()
    config_instance.numeric_log_level = logging.DEBUG

    main(config_instance)
