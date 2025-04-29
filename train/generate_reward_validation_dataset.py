import os
import sys

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
WRITE_BACK_BATCH_SIZE = 100


# --- Main Script Logic ---
def main(config: Config):
    """Generates validation dataset by curating solutions and reformatting."""
    setup_logging(config.numeric_log_level)
    logger.info("Starting validation dataset generation...")

    # --- Define Paths ---
    sft_data_dir = os.path.join(config.sft_data_dir, f"round_{config.round_number}")
    os.makedirs(sft_data_dir, exist_ok=True)
    preference_pairs_file_path = os.path.join(sft_data_dir, "preference_pairs_evaluation.jsonl")  # Input file
    dataset_file_path = os.path.join(sft_data_dir, "reward_dataset_validation.jsonl")  # Output file
    arc_tasks_base_dir = NET_SCRATCH_TASK_DATA_DIR

    # --- Scan Task Directory ---
    task_name_to_path, directory_structure = load_task_info(arc_tasks_base_dir)

    # --- Clear Output File ---
    with open(dataset_file_path, 'w') as _:
        pass
    logger.info(f"Cleared/Created output dataset file: {dataset_file_path}")

    # --- Generate Final Dataset (Sequential) ---
    logger.info(f"Generating final validation dataset...")
    results_batch_to_write = []
    output_entries = 0

    with open(preference_pairs_file_path, 'r', encoding='utf-8') as infile:
        for line_num, line in enumerate(infile):
            if not line.strip():
                continue

            data = json.loads(line)

            task_name = data.get("task_name")
            task_path = task_name_to_path.get(task_name)
            task_json = load_json_file(task_path)

            prefix = data.get("prefix")
            chosen = data.get("chosen")
            rejected = data.get("rejected")

            weight = 1

            output_data = {
                "task_name": task_name,
                "task_json": task_json,
                "prefix": prefix,
                "chosen": chosen,
                "rejected": rejected,
                "weight": weight
            }
            results_batch_to_write.append(output_data)
            output_entries += 1

            if len(results_batch_to_write) >= WRITE_BACK_BATCH_SIZE:
                write_batch_data(dataset_file_path, results_batch_to_write)
                logger.debug(
                    f"Written batch of {len(results_batch_to_write)} entries. Total output entries: {output_entries}")
                results_batch_to_write = []

    # --- Write Final Batch ---
    if results_batch_to_write:
        write_batch_data(dataset_file_path, results_batch_to_write)
        logger.debug(f"Written final batch of {len(results_batch_to_write)} entries.")

    logger.info(f"--- Policy Validation Dataset Creation Summary ---")
    logger.info(f"Processed pairs: {output_entries}")


if __name__ == "__main__":
    config_instance = Config()
    config_instance.numeric_log_level = logging.DEBUG

    main(config_instance)
