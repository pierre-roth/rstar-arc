import json
import logging
import os

from pebble import ProcessPool

from constants import NET_SCRATCH_PATH, STEP_END
from rstar_deepthink.arc_task import ARCTask
from rstar_deepthink.config import Config
from rstar_deepthink.tools import execute_code_with_task
from utils import batch  # Assuming batch(iterable, size) yields lists of items

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')  # Added basic logging config


def process_solution(args: (ARCTask, str)) -> str:  # Corrected type hint syntax
    """
    Process a solution to remove unnecessary steps in a greedy way.

    Args:
        args: Tuple containing (task, solution_code)

    Returns:
        The cleaned solution code with minimal necessary steps
    """
    task, solution_code = args

    # Split the solution into parts (by STEP_END)
    # Ensure we handle potential empty strings from split if STEP_END is at the start/end
    parts = [part for part in solution_code.split(STEP_END) if part]  # Filter out empty parts

    # If there's only one step or fewer, we can't remove anything
    if len(parts) <= 1:
        return solution_code

    # Prepare inputs and expected outputs once
    input_grids = [example.input_grid.grid for example in task.training_examples + task.test_examples]
    expected_outputs = [example.output_grid.grid for example in task.training_examples + task.test_examples]

    current_solution_code = STEP_END.join(parts)  # Rejoin parts consistently
    current_parts = parts

    # Try removing each step one by one iteratively
    changed = True
    while changed:
        changed = False
        step_to_remove = -1  # Index of the step that was successfully removed

        for i in range(len(current_parts)):
            # Create a new solution without the i-th step
            temp_parts = current_parts[:i] + current_parts[i + 1:]

            # If removing the step leaves no parts, it's invalid (unless original was just one step)
            if not temp_parts:
                continue

            new_solution_code = STEP_END.join(temp_parts)

            error, passed, _ = execute_code_with_task(new_solution_code, input_grids, expected_outputs)

            if not error and passed:
                logger.debug(f"Task {task.name}: Successfully removed step {i}")
                current_solution_code = new_solution_code
                current_parts = temp_parts  # Update parts for the next iteration of the while loop
                changed = True
                step_to_remove = i  # Record which step was removed
                break  # Restart the removal check from the beginning with the smaller solution

        if changed:
            logger.info(
                f"Task {task.name}: Removed step at index {step_to_remove} (original index). New length: {len(current_parts)} steps.")

    return current_solution_code


def main():
    # Create config
    config = Config()

    config.batch_size = max(1, config.cpus - 1)

    # Define file paths
    sft_data_dir = os.path.join(NET_SCRATCH_PATH, "sft_data", f"round_{config.round_number}")
    raw_file = os.path.join(sft_data_dir, "raw.jsonl")
    cleaned_file = os.path.join(sft_data_dir, "cleaned.jsonl")
    task_dir = os.path.join(NET_SCRATCH_PATH, "task_data")

    # Create output directory if it doesn't exist
    os.makedirs(sft_data_dir, exist_ok=True)

    logger.info(f"Starting SFT data cleaning from {raw_file} to {cleaned_file}")
    logger.info(f"Using {config.cpus} CPUs and batch size {config.batch_size}")

    # --- Load Tasks ---
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
                    else:
                        logger.warning(f"Skipping line in {raw_file} without 'task_name': {line.strip()}")
                except json.JSONDecodeError:
                    logger.warning(f"Skipping invalid JSON line in {raw_file}: {line.strip()}")
    except FileNotFoundError:
        logger.error(f"Raw input file not found: {raw_file}")
        return  # Exit if input file doesn't exist

    logger.info(f"Found {len(solved_set)} unique task names in {raw_file}")

    name_to_task: dict[str, ARCTask] = {}
    # Walk the task_data dir to load ARCTask objects
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

    logger.info(f"Successfully loaded {tasks_loaded_count} task objects corresponding to solved tasks.")

    # --- Process Solutions ---
    # Create the process pool *once* here
    with ProcessPool(max_workers=config.cpus) as pool, \
            open(raw_file, 'r', encoding='utf-8') as infile, \
            open(cleaned_file, 'w', encoding='utf-8') as outfile:

        processed_count = 0
        batch_num = 0
        # Process the raw file in batches using the utility function
        for lines_batch in batch(infile, config.batch_size):
            batch_num += 1
            logger.info(f"Processing batch {batch_num} with {len(lines_batch)} solutions...")
            batch_futures = []  # Store futures for this batch

            # Schedule jobs for the current batch
            for line in lines_batch:
                if not line.strip():
                    continue
                try:
                    data = json.loads(line)
                    task_name = data.get("task_name")
                    solution_code = data.get("solution_code")

                    if not task_name or solution_code is None:  # Check for None explicitly
                        logger.warning(f"Skipping invalid entry in batch {batch_num}: {line.strip()}")
                        continue

                    task = name_to_task.get(task_name)
                    if not task:
                        logger.warning(f"Task '{task_name}' found in raw file but not loaded from task data. Skipping.")
                        continue

                    # Schedule the process_solution function
                    future = pool.schedule(process_solution, args=[(task, solution_code)])
                    batch_futures.append((task_name, future))

                except json.JSONDecodeError:
                    logger.warning(f"Skipping invalid JSON line in batch {batch_num}: {line.strip()}")
                except Exception as e:
                    logger.error(f"Unexpected error scheduling task from line: {line.strip()} - {e}")

            # Collect results for the current batch
            for task_name, future in batch_futures:
                try:
                    # Wait for the result (this blocks until the specific process is done)
                    cleaned_code = future.result()

                    # Write the cleaned solution to the output file
                    entry = json.dumps({"task_name": task_name, "solution_code": cleaned_code})
                    outfile.write(entry + '\n')
                    processed_count += 1

                except Exception as e:
                    # Log errors from the process_solution execution
                    logger.error(f"Error processing task {task_name} in worker process: {e}")

            logger.info(f"Finished processing batch {batch_num}. Total processed so far: {processed_count}")

    logger.info(f"Finished cleaning SFT data. {processed_count} solutions processed and saved to {cleaned_file}")


if __name__ == "__main__":
    main()
