import json
import logging
import os
import random
import sys
import argparse
import pathlib
from typing import Optional, Dict, List, Any

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

from constants import NET_SCRATCH_TASK_DATA_DIR_PATH, NET_SCRATCH_RE_ARC_DATA_PATH

from rstar_deepthink.config import Config
from rstar_deepthink.arc_task import ARCTask  # Assuming ARCTask can load from path or dict
from rstar_deepthink.tools import execute_code_with_task
from utils import setup_logging

logger = logging.getLogger(__name__)


def load_arc_tasks(base_dir: str, config: Config) -> Dict[str, ARCTask]:
    """
    Recursively scans base_dir for .json files, loads them as ARCTask objects.
    Returns a dictionary mapping task_name to ARCTask object.
    """
    tasks = {}
    logger.info(f"Loading ARC tasks from base directory: {base_dir}")
    if not os.path.isdir(base_dir):
        logger.error(f"ARC task base directory not found: {base_dir}")
        return tasks

    count = 0
    for filepath in pathlib.Path(base_dir).rglob('*.json'):
        task_name = filepath.stem
        try:
            # Pass config to ARCTask constructor
            tasks[task_name] = ARCTask(config=config, path=str(filepath))
            count += 1
        except Exception as e:
            logger.error(f"Failed to load task {task_name} from {filepath}: {e}")
    logger.info(f"Loaded {count} ARC tasks.")
    return tasks


def load_rearc_examples(rearc_dir: str, task_name: str) -> Optional[List[Dict[str, Any]]]:
    """
    Loads reARC data for a specific task from {rearc_dir}/{task_name}.json.
    Expects a JSON list of {"input": grid, "output": grid} dictionaries.
    Returns the list of examples or None if not found or on error.
    """
    rearc_file = os.path.join(rearc_dir, f"{task_name}.json")
    try:
        with open(rearc_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

            logger.debug(f"Loaded {len(data)} reARC examples for task {task_name}.")
            return data
    except json.JSONDecodeError:
        logger.error(f"Failed to parse JSON from reARC file: {rearc_file}")
        return None
    except Exception as e:
        logger.error(f"Error loading reARC file {rearc_file}: {e}")
        return None


def test_solution_on_rearc_example(solution_code: str, rearc_example: Dict[str, Any]) -> bool:
    """
    Tests if the given solution_code correctly solves a single reARC example.
    Uses execute_code_with_task for validation.
    """
    try:
        input_grid = rearc_example['input']
        expected_output = rearc_example['output']

        # Test the code against this single pair
        error, passed, _ = execute_code_with_task(solution_code, [input_grid], [expected_output])

        # Return True only if execution had no errors AND the example passed
        return not error and passed
    except KeyError:
        logger.warning("Malformed reARC example dictionary passed to testing function.")
        return False
    except Exception as e:
        logger.error(f"Error during solution testing on reARC example: {e}")
        return False


def save_augmented_data(filepath: str, data: Dict):
    """Appends a JSON line to the output file."""
    try:
        with open(filepath, 'a', encoding='utf-8') as f:
            f.write(json.dumps(data) + '\n')
    except IOError as e:
        logger.error(f"Failed to write to augmented data file {filepath}: {e}")


def main(config: Config):
    """Main function to process cleaned data and generate augmented SFT data."""
    setup_logging(config)
    logger.info("Starting SFT data augmentation process ...")

    sft_data_dir = config.sft_data_dir
    cleaned_file = os.path.join(sft_data_dir, "cleaned.jsonl")
    augmented_file = os.path.join(sft_data_dir, "augmented.jsonl")
    arc_tasks_base_dir = NET_SCRATCH_TASK_DATA_DIR_PATH
    arc_training_dir = os.path.join(arc_tasks_base_dir, "training")
    rearc_data_dir = NET_SCRATCH_RE_ARC_DATA_PATH

    logger.debug(f"Cleaned SFT input file: {cleaned_file}")
    logger.debug(f"Augmented SFT output file: {augmented_file}")
    logger.debug(f"Original ARC task base directory: {arc_tasks_base_dir}")
    logger.debug(f"Original ARC training directory: {arc_training_dir}")
    logger.debug(f"reARC data directory: {rearc_data_dir}")

    all_arc_tasks = load_arc_tasks(arc_tasks_base_dir, config)
    training_names = set(os.path.splitext(file_name) for file_name in os.listdir(arc_training_dir))

    # --- Process Cleaned Data ---
    processed_lines = 0
    augmented_tasks_saved = 0

    # Overwrite augmented file at the start
    try:
        with open(augmented_file, 'w') as f:
            logger.info(f"Cleared/Created output file: {augmented_file}")
    except IOError as e:
        logger.error(f"Could not open or clear output file {augmented_file}: {e}")
        sys.exit(1)

    try:
        with open(cleaned_file, 'r', encoding='utf-8') as infile:
            for line_num, line in enumerate(infile):
                processed_lines += 1
                if not line.strip():
                    continue

                try:
                    data = json.loads(line)
                    task_name = data.get("task_name")
                    solution_code = data.get("solution_code")
                    metadata = data.get("metadata")  # Keep metadata

                    original_task = all_arc_tasks.get(task_name)

                    is_training = original_task.name in training_names

                    # --- Always save the original task data first for all tasks ---
                    # This ensures every solution from cleaned.jsonl has at least one entry in augmented.jsonl
                    output_data_original = {
                        "task_name": task_name,
                        "task_json": original_task.json_data,
                        "solution": solution_code,
                        "metadata": metadata
                    }
                    save_augmented_data(augmented_file, output_data_original)

                    # --- If Training Task, attempt augmentation ---
                    if is_training:
                        logger.info(f"Processing training task '{task_name}' for augmentation...")
                        rearc_examples = load_rearc_examples(rearc_data_dir, task_name)

                        if rearc_examples:
                            # Filter reARC examples: only keep those the solution solves
                            logger.debug(f"Filtering {len(rearc_examples)} reARC examples for task '{task_name}'...")
                            filtered_examples = [
                                ex for ex in rearc_examples
                                if test_solution_on_rearc_example(solution_code, ex)
                            ]
                            logger.debug(f"Found {len(filtered_examples)} valid reARC examples for task '{task_name}'.")

                            # Determine k (num original training examples)
                            k = len(original_task.training_examples) + len(original_task.test_examples)
                            n = len(filtered_examples)

                            if n < k:
                                logger.warning(
                                    f"Task '{task_name}': Not enough valid reARC examples ({n}) to match original training examples ({k}). No augmentation performed.")
                            else:
                                # Proceed with augmentation
                                num_augmented_tasks = n // k
                                logger.info(
                                    f"Task '{task_name}': Found {n} valid reARC examples (k={k}). Generating {num_augmented_tasks} augmented tasks.")
                                random.shuffle(filtered_examples)

                                for i in range(num_augmented_tasks):
                                    # Select k examples for the new 'train' part
                                    new_examples = filtered_examples[i * k: (i + 1) * k]

                                    # Create the new task JSON
                                    new_task_json = {
                                        "train": new_examples[:-1],  # All but the last example
                                        "test": [new_examples[-1]]  # the last example is the test
                                    }

                                    # Format and save
                                    output_data_augmented = {
                                        "task_name": f"{task_name}_augmented_{i + 1}",
                                        "task_json": new_task_json,
                                        "solution": solution_code,
                                        "metadata": metadata
                                    }
                                    save_augmented_data(augmented_file, output_data_augmented)
                                    augmented_tasks_saved += 1  # Count each augmented save

                        else:
                            logger.warning(
                                f"No reARC data found for training task '{task_name}'. Only original task saved.")

                except json.JSONDecodeError:
                    logger.warning(f"Skipping line {line_num + 1}: Invalid JSON.")
                except Exception as e:
                    logger.error(f"Error processing line {line_num + 1}: {e}", exc_info=True)

                logger.debug(f"Processed {line_num + 1} lines from cleaned file. Saved {augmented_tasks_saved} entries so far.")

    except FileNotFoundError:
        logger.error(f"Input file not found: {cleaned_file}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"An unexpected error occurred during processing: {e}", exc_info=True)

    logger.info(f"Finished augmentation. Processed {processed_lines} lines from cleaned file.")
    logger.info(f"Total entries saved to augmented file: {augmented_tasks_saved}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Augment SFT data for rSTAR-ARC.')
    args = parser.parse_args()

    config_instance = Config()

    main(config_instance)
