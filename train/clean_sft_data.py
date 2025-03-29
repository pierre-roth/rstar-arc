import json
import logging
import os
import sys
import itertools  # For efficient line skipping
import collections  # For efficient line skipping with deque

from pebble import ProcessPool

# Add the project root to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)

from constants import NET_SCRATCH_PATH, STEP_END
from rstar_deepthink.arc_task import ARCTask
from rstar_deepthink.config import Config
from rstar_deepthink.tools import execute_code_with_task
from utils import batch, setup_logging

logger = logging.getLogger(__name__)


def process_solution(args: (ARCTask, str)) -> str:
    """
    Process a solution to remove unnecessary steps using a single backward pass.
    O(N) complexity where N is the initial number of steps.

    Args:
        args: tuple containing (task, solution_code)

    Returns:
        The cleaned solution code.
    """
    task, solution_code = args
    task_name = task.name  # For logging

    original_parts: list[str] = [part for part in solution_code.split(STEP_END) if part]
    num_original_steps = len(original_parts)

    if num_original_steps <= 1:
        return solution_code

    try:
        input_grids = [example.input_grid.grid for example in task.training_examples + task.test_examples]
        expected_outputs = [example.output_grid.grid for example in task.training_examples + task.test_examples]
    except AttributeError as e:
        logger.error(f"Task {task_name}: Error accessing grid data - {e}. Skipping cleaning.")
        return solution_code

    indices_to_keep = list(range(num_original_steps))
    steps_removed_count = 0

    for i in range(num_original_steps - 1, -1, -1):
        if i not in indices_to_keep:
            continue
        temp_indices = [idx for idx in indices_to_keep if idx != i]
        if not temp_indices:
            continue

        temp_parts = [original_parts[idx] for idx in sorted(temp_indices)]
        new_solution_code = STEP_END.join(temp_parts)
        error, passed, _ = execute_code_with_task(new_solution_code, input_grids, expected_outputs)

        if not error and passed:
            indices_to_keep = temp_indices
            steps_removed_count += 1

    current_parts = [original_parts[idx] for idx in sorted(indices_to_keep)]
    current_solution_code = STEP_END.join(current_parts)

    return current_solution_code


def count_lines(filename):
    """Counts lines in a file, returns 0 if file not found or on error."""
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            # Efficiently count lines using a generator expression
            return sum(1 for _ in f)
    except FileNotFoundError:
        return 0
    except Exception as e:
        # Log error but return 0 to allow starting fresh if count fails
        logger.error(f"Error counting lines in {filename}: {e}")
        return 0


def main():
    # Create config
    config = Config()

    # Consider using config.cpus directly if main thread is not busy
    config.batch_size = max(1, config.cpus)  # Leave one CPU for main thread

    # Assuming setup_logging configures the root logger or specific loggers
    setup_logging(config)

    # Define file paths
    sft_data_dir = os.path.join(NET_SCRATCH_PATH, "sft_data", f"round_{config.round_number}")
    raw_file = os.path.join(sft_data_dir, "raw.jsonl")
    cleaned_file = os.path.join(sft_data_dir, "cleaned.jsonl")
    task_dir = os.path.join(NET_SCRATCH_PATH, "task_data")
    # Define curated file path here as well
    curated_file = os.path.join(sft_data_dir, "curated_solutions.jsonl")

    # Create output directory if it doesn't exist
    os.makedirs(sft_data_dir, exist_ok=True)

    # Count existing lines in cleaned file to determine where to resume
    num_already_cleaned = count_lines(cleaned_file)
    if num_already_cleaned > 0:
        logger.info(f"Found {num_already_cleaned} existing lines in {cleaned_file}. Will resume.")
    else:
        logger.info(f"Starting clean run. {cleaned_file} is empty or not found.")

    logger.info(f"Starting SFT data cleaning processing from {raw_file} to {cleaned_file}")
    logger.info(f"Using {config.cpus} CPUs and batch size {config.batch_size}")

    # --- Load Tasks ---
    # This part still loads all tasks mentioned in raw_file, simple approach.
    solved_set = set()
    try:
        with open(raw_file, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    data = json.loads(line)
                    task_name = data.get("task_name")
                    if task_name:
                        solved_set.add(task_name)
                except json.JSONDecodeError:
                    # Reduce log noise for invalid lines during initial scan
                    pass
    except FileNotFoundError:
        logger.error(f"Raw input file not found: {raw_file}")
        return

    name_to_task: dict[str, ARCTask] = {}
    if not os.path.isdir(task_dir):
        logger.error(f"Task data directory not found: {task_dir}")
        return

    tasks_loaded_count = 0
    for dir_name in os.listdir(task_dir):
        dir_path = os.path.join(task_dir, dir_name)
        if os.path.isdir(dir_path):
            for task_file in os.listdir(dir_path):
                task_name = task_file.removesuffix(".json")
                if task_file.endswith(".json") and task_name in solved_set:
                    task_path = os.path.join(dir_path, task_file)
                    try:
                        task = ARCTask(config, task_path)
                        name_to_task[task.name] = task
                        tasks_loaded_count += 1
                    except Exception as e:
                        logger.error(f"Failed to load task {task_name} from {task_path}: {e}")

    logger.info(f"Successfully loaded {tasks_loaded_count} task objects.")

    # --- Process Solutions ---
    processed_in_this_run = 0  # Count only items processed now
    try:
        # Open output file in append mode to continue where we left off
        with ProcessPool(max_workers=config.cpus) as pool, \
                open(raw_file, 'r', encoding='utf-8') as infile, \
                open(cleaned_file, 'a', encoding='utf-8') as outfile:  # Use 'a' append mode

            batch_num = 0  # Batch number for logging this run

            # Skip already processed lines in input file
            infile_iter = iter(infile)  # Get iterator from file handle
            if num_already_cleaned > 0:
                logger.info(f"Attempting to skip first {num_already_cleaned} lines from {raw_file} to resume...")
                try:
                    # Use islice and deque to efficiently consume the iterator N times
                    consumer = itertools.islice(infile_iter, num_already_cleaned)
                    # collections.deque efficiently exhausts the islice iterator without storing results
                    collections.deque(consumer, maxlen=0)
                    logger.info(f"Successfully skipped {num_already_cleaned} lines.")
                except Exception as e:
                    # If skipping fails, log error and potentially stop to avoid processing wrong lines
                    logger.error(f"Error trying to skip lines in {raw_file}: {e}", exc_info=True)
                    raise  # Re-raise error, as continuing could lead to data corruption

            # Process the remaining lines using the potentially advanced iterator
            for lines_batch in batch(infile_iter, config.batch_size):  # Pass infile_iter
                batch_num += 1
                # Reduce log noise, use debug if needed
                # logger.info(f"Processing batch {batch_num} (size {len(lines_batch)})...")
                batch_futures = []

                # Schedule jobs for the current batch
                for line in lines_batch:
                    if not line.strip():
                        continue
                    try:
                        data = json.loads(line)
                        task_name = data.get("task_name")
                        solution_code = data.get("solution_code")

                        if not task_name or solution_code is None:
                            continue

                        task = name_to_task.get(task_name)
                        if not task:
                            continue

                        future = pool.schedule(process_solution, args=[(task, solution_code)])
                        batch_futures.append((task_name, future))

                    except json.JSONDecodeError:
                        # Reduce log noise
                        # logger.warning(f"Skipping invalid JSON line in batch {batch_num}: {line.strip()}")
                        pass
                    except Exception as e:
                        logger.error(f"Error scheduling task for line: {line.strip()} - {e}")

                # Collect results for the current batch
                results_this_batch = 0
                for task_name, future in batch_futures:
                    try:
                        # Wait for the result
                        cleaned_code = future.result()
                        # Write the cleaned solution
                        entry = json.dumps({"task_name": task_name, "solution_code": cleaned_code})
                        outfile.write(entry + '\n')
                        results_this_batch += 1
                    except Exception as e:
                        # Log errors from the worker process execution
                        logger.error(f"Error processing task {task_name} in worker process: {e}")

                processed_in_this_run += results_this_batch
                # Log progress periodically, maybe not every batch if too fast
                if batch_num % 10 == 0:  # Log every 10 batches, adjust as needed
                    logger.info(
                        f"Processed batch {batch_num}. Total processed in this run: {processed_in_this_run}. Total lines in file now: ~{num_already_cleaned + processed_in_this_run}")

    except Exception as e:
        # Catch potential errors during file handling or pool processing
        logger.error(f"An error occurred during the main processing loop: {e}", exc_info=True)
        # Depending on severity, might want to exit or allow curation to run on partial data

    # Log final counts
    total_processed_lines = count_lines(cleaned_file)  # Recount final lines for accuracy
    logger.info(f"Finished cleaning SFT data processing. Processed {processed_in_this_run} solutions in this run.")
    logger.info(f"Total lines now in {cleaned_file}: {total_processed_lines}")


    # --- Curate Best Solution per Task ---
    # This part remains unchanged - it reads the final cleaned_file (potentially appended to)
    # It will correctly find the best solution among all lines present.
    logger.info(f"Starting curation of best solutions from {cleaned_file}...")
    best_solutions = {}  # Dictionary to store {task_name: best_solution_code}

    try:
        # Read the potentially appended cleaned file
        with open(cleaned_file, 'r', encoding='utf-8') as infile:
            for line_num, line in enumerate(infile):
                if not line.strip():
                    continue
                try:
                    data = json.loads(line)
                    task_name = data.get("task_name")
                    current_solution_code = data.get("solution_code")

                    if not task_name or current_solution_code is None:
                        continue

                    # Apply heuristic: shortest solution wins
                    existing_best_code = best_solutions.get(task_name)
                    if existing_best_code is None or len(current_solution_code) < len(existing_best_code):
                        best_solutions[task_name] = current_solution_code

                except json.JSONDecodeError:
                    # Reduce log noise
                    # logger.warning(f"Skipping invalid JSON line in {cleaned_file} line {line_num + 1}: {line.strip()}")
                    pass
                except Exception as e:
                    logger.error(
                        f"Error processing curation line {line_num + 1} from {cleaned_file}: {line.strip()} - {e}")

    except FileNotFoundError:
        logger.error(f"Cleaned file not found for curation: {cleaned_file}. Cannot proceed.")
        return
    except Exception as e:
        logger.error(f"Failed to read {cleaned_file} for curation: {e}")
        return

    curated_count = 0
    logger.info(f"Found best solutions for {len(best_solutions)} unique tasks. Writing to {curated_file}...")
    try:
        # Write mode 'w' is correct here, always overwrite/create curated file
        with open(curated_file, 'w', encoding='utf-8') as outfile:
            # Sort items by task name for deterministic output order
            for task_name, solution_code in sorted(best_solutions.items()):
                entry = json.dumps({"task_name": task_name, "solution_code": solution_code})
                outfile.write(entry + '\n')
                curated_count += 1
        logger.info(f"Finished curation. Selected and saved {curated_count} best solutions to {curated_file}")
    except IOError as e:
        logger.error(f"Failed to write curated solutions to {curated_file}: {e}")


if __name__ == "__main__":
    main()
