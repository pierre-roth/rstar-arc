import argparse
import json
import logging
import os
import pathlib
import random
import sys
from concurrent.futures import TimeoutError
from typing import Optional, Dict, List, Any, Set, Tuple

from pebble import ProcessPool

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

from constants import NET_SCRATCH_TASK_DATA_DIR_PATH, NET_SCRATCH_RE_ARC_DATA_PATH
from rstar_deepthink.config import Config
from rstar_deepthink.tools import execute_code_with_task
from rstar_deepthink.tools.python_tool import remove_markers
from utils import setup_logging

logger = logging.getLogger(__name__)


def load_task_info(base_dir: str) -> Tuple[Dict[str, str], Dict[str, Set[str]]]:
    """
    Scans the base directory for .json files.
    Returns:
        - task_name_to_path (Dict[str, str]): Maps task name to its full file path.
        - structure (Dict[str, Set[str]]): Maps subdir name ('training', etc.) to set of task names.
    """
    task_name_to_path: Dict[str, str] = {}
    structure: Dict[str, Set[str]] = {}
    logger.info(f"Scanning task directory structure: {base_dir}")
    base_path = pathlib.Path(base_dir)

    for filepath in base_path.rglob('*.json'):
        task_name = filepath.stem
        task_name_to_path[task_name] = str(filepath)

        # Store structure (subdir -> task_names)
        relative_dir = filepath.parent.relative_to(base_path)
        subdir_key = str(relative_dir).split(os.path.sep)[0] if relative_dir.parts else '.'
        if subdir_key not in structure:
            structure[subdir_key] = set()
        structure[subdir_key].add(task_name)

    logger.info(f"Finished scanning. Found {len(task_name_to_path)} tasks in subdirectories: {list(structure.keys())}")
    return task_name_to_path, structure


def load_json_file(filepath: str) -> Optional[Any]:
    """Loads JSON data from a single file."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        logger.error(f"File not found: {filepath}")
    except json.JSONDecodeError:
        logger.error(f"Invalid JSON in file: {filepath}")
    except Exception as e:
        logger.error(f"Error loading file {filepath}: {e}")
    return None


def load_rearc_examples(rearc_dir: str, task_name: str) -> Optional[List[Dict[str, Any]]]:
    """Loads reARC data for a specific task."""
    rearc_file = os.path.join(rearc_dir, f"{task_name}.json")
    if not os.path.exists(rearc_file):
        logger.debug(f"No reARC data file for task {task_name} at {rearc_file}")
        return None
    return load_json_file(rearc_file)


def test_solution_on_rearc_example(solution_code: str, rearc_example: Dict[str, Any]) -> bool:
    """Tests a solution against a single reARC example."""
    try:
        error, passed, _ = execute_code_with_task(remove_markers(solution_code), [rearc_example['input']],
                                                  [rearc_example['output']])
        return not error and passed
    except Exception:
        return False  # Assume failure on any error


def write_batch_data(filepath: str, data_batch: List[Dict]):
    """Writes a batch of JSON lines to the output file."""
    if not data_batch:
        return

    try:
        with open(filepath, 'a', encoding='utf-8') as f:
            for data in data_batch:
                f.write(json.dumps(data) + '\n')
    except IOError as e:
        logger.error(f"Failed to write batch to augmented data file {filepath}: {e}")


def process_augmentation_job(job_data: Tuple[str, str, Dict[str, Any], int, str]) -> List[Dict]:
    """Loads reARC, filters, generates, and returns augmented task data."""
    task_name, solution_code, original_task_json, k, rearc_data_dir = job_data
    augmented_results = []

    rearc_examples = load_rearc_examples(rearc_data_dir, task_name)

    if rearc_examples:
        filtered_examples = [ex for ex in rearc_examples if test_solution_on_rearc_example(solution_code, ex)]
        n = len(filtered_examples)
        logger.debug(f"{task_name}: out of {len(rearc_examples)} reARC examples, {n} passed the test.")

        if n >= k > 0:
            num_augmented_tasks = n // k
            random.shuffle(filtered_examples)

            for i in range(num_augmented_tasks):
                new_examples = filtered_examples[i * k: (i + 1) * k]
                if len(new_examples) < k:
                    break

                new_train_examples = new_examples[:-1]
                new_test_example = [new_examples[-1]]

                new_task_json = {"train": new_train_examples, "test": new_test_example}
                output_data = {
                    "task_name": f"{task_name}_augmented_{i + 1}",
                    "task_json": new_task_json,
                    "solution": solution_code,
                }
                augmented_results.append(output_data)
            logger.debug(f"Task '{task_name}' generated {num_augmented_tasks} augmentations.")

    return augmented_results


def main(config: Config):
    """Main function orchestrating the efficient augmentation process."""
    setup_logging(config.numeric_log_level)
    logger.info("Starting SFT data augmentation process (Simplified)...")

    # --- Define Paths ---
    sft_data_dir = os.path.join(config.sft_data_dir, f"round_{config.round_number}")
    cleaned_file = os.path.join(sft_data_dir, "cleaned.jsonl")
    cleaned_file = os.path.join(sft_data_dir, "curated_solutions.jsonl")
    augmented_file = os.path.join(sft_data_dir, "augmented.jsonl")
    arc_tasks_base_dir = NET_SCRATCH_TASK_DATA_DIR_PATH
    rearc_data_dir = NET_SCRATCH_RE_ARC_DATA_PATH

    # --- Scan Task Directory & Load Cleaned Data ---
    task_name_to_path, directory_structure = load_task_info(arc_tasks_base_dir)
    training_task_names = directory_structure.get('training', set())

    cleaned_data_cache: List[Dict] = []
    required_task_names: Set[str] = set()
    try:
        with open(cleaned_file, 'r', encoding='utf-8') as infile:
            for line in infile:
                if line.strip():
                    try:
                        data = json.loads(line)
                        task_name = data.get("task_name")
                        if task_name and task_name in task_name_to_path:
                            required_task_names.add(task_name)
                            cleaned_data_cache.append(data)
                        elif task_name:
                            logger.warning(f"Task '{task_name}' from cleaned file not found in task directory scan.")
                    except json.JSONDecodeError:
                        logger.warning(f"Skipping invalid JSON in cleaned file: {line.strip()}")
    except FileNotFoundError:
        logger.error(f"Cleaned input file not found: {cleaned_file}")
        sys.exit(1)

    logger.info(f"Found {len(cleaned_data_cache)} cleaned solutions for {len(required_task_names)} existing tasks.")

    # --- Prepare Augmentation Jobs & Save Originals ---
    augmentation_jobs = []
    original_task_batch = []

    # Overwrite augmented file
    try:
        with open(augmented_file, 'w') as f:
            pass  # Just create/clear
        logger.info(f"Cleared/Created output file: {augmented_file}")
    except IOError as e:
        logger.error(f"Could not open or clear output file {augmented_file}: {e}")
        sys.exit(1)

    original_tasks_saved = 0
    BATCH_SIZE = 1000  # Define a reasonable batch size for writing operations

    for data in cleaned_data_cache:
        task_name = data["task_name"]  # Assumed to exist now
        solution_code = data["solution_code"]

        original_task_json = load_json_file(task_name_to_path[task_name])
        if not original_task_json:
            logger.warning(f"Could not load original task JSON for '{task_name}' - skipping.")
            continue

        # Prepare the original task data
        output_data_original = {
            "task_name": task_name,
            "task_json": original_task_json,
            "solution": solution_code,
        }
        original_task_batch.append(output_data_original)
        original_tasks_saved += 1

        # Batch write when we reach BATCH_SIZE
        if len(original_task_batch) >= BATCH_SIZE:
            write_batch_data(augmented_file, original_task_batch)
            original_task_batch = []

        # If Training Task, prepare job
        if task_name in training_task_names:
            k = len(original_task_json.get('train', [])) + len(original_task_json.get('test', []))
            job = (task_name, solution_code, original_task_json, k, rearc_data_dir)
            augmentation_jobs.append(job)

    # Write any remaining original task entries
    if original_task_batch:
        write_batch_data(augmented_file, original_task_batch)

    logger.info(f"Saved {original_tasks_saved} original task entries.")
    logger.info(f"Prepared {len(augmentation_jobs)} augmentation jobs for parallel processing.")

    # --- Parallel Augmentation ---
    augmented_tasks_saved_count = 0
    if augmentation_jobs:
        num_workers = max(1, config.cpus - 1 if config.cpus > 1 else 1)
        logger.info(f"Starting parallel augmentation using {num_workers} workers...")
        task_timeout = 600  # Seconds per job

        augmented_results_batch = []
        try:
            with ProcessPool(max_workers=num_workers) as pool:
                future = pool.map(process_augmentation_job, augmentation_jobs, timeout=task_timeout)
                results_iterator = future.result()
                processed_jobs = 0
                while True:
                    try:
                        augmented_data_list = next(results_iterator)
                        augmented_results_batch.extend(augmented_data_list)
                        augmented_tasks_saved_count += len(augmented_data_list)
                        processed_jobs += 1

                        # Batch write when we reach BATCH_SIZE
                        if len(augmented_results_batch) >= BATCH_SIZE:
                            write_batch_data(augmented_file, augmented_results_batch)
                            augmented_results_batch = []

                        logger.info(
                            f"Completed processing {processed_jobs}/{len(augmentation_jobs)} augmentation jobs...")
                    except StopIteration:
                        break
                    except TimeoutError as error:
                        logger.error(f"Job timed out after {error.args[1]}s. Skipping results for that task.")
                    except Exception as error:
                        logger.error(f"Job failed: {error}", exc_info=False)  # Less verbose error for worker failure

            # Write any remaining augmented entries
            if augmented_results_batch:
                write_batch_data(augmented_file, augmented_results_batch)

            logger.info(f"Finished parallel processing. Saved {augmented_tasks_saved_count} augmented entries.")

        except Exception as e:
            logger.error(f"Error during parallel pool execution: {e}", exc_info=True)

    logger.info(f"--- Augmentation Summary ---")
    logger.info(f"Total original entries saved: {original_tasks_saved}")
    logger.info(f"Total augmented entries generated & saved: {augmented_tasks_saved_count}")
    logger.info(f"Total entries in augmented file: {original_tasks_saved + augmented_tasks_saved_count}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Augment SFT data for rSTAR-ARC (Simplified).')
    args = parser.parse_args()
    config_instance = Config()
    main(config_instance)