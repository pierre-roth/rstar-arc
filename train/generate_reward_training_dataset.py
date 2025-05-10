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
    augmented_file_path = os.path.join(sft_data_dir, "preference_pairs_training_augmented.jsonl")
    dataset_file_path = os.path.join(sft_data_dir, "reward_dataset_training.jsonl")
    arc_tasks_base_dir = NET_SCRATCH_TASK_DATA_DIR

    # --- Scan Task Directory (Needed for Pass 1) ---
    task_name_to_path, directory_structure = load_task_info(arc_tasks_base_dir)

    # --- Pass 1: Calculate number of output entries per original preference pair ---
    logger.info(f"Calculating weights: reading {augmented_file_path} to count entries per pair...")
    total_output_entries_per_pair: dict[int, int] = {}
    total_output_entries_per_task: dict[str, int] = {}

    with open(augmented_file_path, 'r', encoding='utf-8') as infile:
        for line_num, line in enumerate(infile):
            if not line.strip():
                continue
            data = json.loads(line)
            original_task_name = data.get("original_task_name")
            augmented_examples = data.get("examples", [])

            # Load original task to determine chunk size
            original_task_path = task_name_to_path.get(original_task_name)
            original_task_json = load_json_file(original_task_path)
            n_train = len(original_task_json.get('train', []))
            n_test = len(original_task_json.get('test', []))
            chunk_size = n_train + n_test

            # Total output entries = one per complete chunk + one full-task entry
            num_complete_chunks = len(augmented_examples) // chunk_size if chunk_size > 0 else 0
            total_output_entries_per_pair[line_num] = num_complete_chunks + 1
            total_output_entries_per_task[original_task_name] = total_output_entries_per_task.get(original_task_name, 0) + num_complete_chunks + 1

    # --- Clear Output File ---
    with open(dataset_file_path, 'w') as _:
        pass
    logger.info(f"Cleared/Created output dataset file: {dataset_file_path}")

    # --- Pass 2: Process and Write Final Dataset ---
    logger.info(f"Processing augmented data and writing to {dataset_file_path}...")
    results_batch_to_write = []
    processed_lines_pass2 = 0
    output_entries = 0
    with open(augmented_file_path, 'r', encoding='utf-8') as infile:
        for line_num, line in enumerate(infile):
            processed_lines_pass2 += 1
            if not line.strip():
                continue

            data = json.loads(line)
            original_task_name = data.get("original_task_name")
            prefix = data.get("prefix")
            chosen = data.get("chosen")
            rejected = data.get("rejected")
            # metadata = data.get("metadata")

            augmented_examples = data.get("examples")

            # total_entries = total_output_entries_per_pair.get(line_num, 1)
            total_entries_per_task = total_output_entries_per_task.get(original_task_name, 1)

            # weight = 1.0 / total_entries
            weight = 1.0 / total_entries_per_task

            # --- Load Original Task Info Again for Chunk Size ---
            original_task_path = task_name_to_path.get(original_task_name)
            original_task_json = load_json_file(original_task_path)

            # --- Determine Chunk Size Again ---
            n_train = len(original_task_json['train'])
            n_test = len(original_task_json['test'])
            chunk_size = n_train + n_test

            # --- Process and Chunk Examples ---
            for i in range(0, len(augmented_examples), chunk_size):
                chunk = augmented_examples[i: i + chunk_size]

                if len(chunk) == chunk_size:  # Only process complete chunks
                    new_train_examples = chunk[:n_train]
                    new_test_examples = chunk[n_train:]

                    # Construct final output format
                    output_data = {
                        "task_name": original_task_name,
                        "task_json": {
                            "train": new_train_examples,
                            "test": new_test_examples
                        },
                        "prefix": prefix,
                        "chosen": chosen,
                        "rejected": rejected,
                        "weight": weight
                    }
                    results_batch_to_write.append(output_data)
                    output_entries += 1

            output_data = {
                "task_name": original_task_name,
                "task_json": original_task_json,
                "prefix": prefix,
                "chosen": chosen,
                "rejected": rejected,
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

    logger.info(f"--- Dataset Creation Summary ---")
    logger.info(f"Processed {processed_lines_pass2} lines from {augmented_file_path}")
    logger.info(f"Wrote {output_entries} entries to {dataset_file_path}")


if __name__ == "__main__":
    # Instantiate config directly
    config_instance = Config()
    config_instance.numeric_log_level = logging.DEBUG

    main(config_instance)
