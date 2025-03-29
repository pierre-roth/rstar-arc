import json
import logging
import os
import sys

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
        args: Tuple containing (task, solution_code)

    Returns:
        The cleaned solution code.
    """
    task, solution_code = args
    task_name = task.name  # For logging

    # Split the solution into parts (by STEP_END) - same as greedy version
    # Ensure we handle potential empty strings from split if STEP_END is at the start/end
    original_parts: list[str] = [part for part in solution_code.split(STEP_END) if part]  # Filter out empty parts
    num_original_steps = len(original_parts)

    # If there's only one step or fewer, we can't remove anything - same as greedy version
    if num_original_steps <= 1:
        # Optional: Add log matching the style if desired
        # logger.info(f"Task {task_name}: Solution has {num_original_steps} steps, no cleaning possible.")
        return solution_code

    # Prepare inputs and expected outputs once - same as greedy version
    try:
        input_grids = [example.input_grid.grid for example in task.training_examples + task.test_examples]
        expected_outputs = [example.output_grid.grid for example in task.training_examples + task.test_examples]
    except AttributeError as e:
        logger.error(f"Task {task_name}: Error accessing grid data - {e}. Skipping cleaning.")
        return solution_code  # Return original if task structure is unexpected

    # --- Backward Pass Removal (O(N) logic) ---
    # Keep track of the indices of the original steps that are currently kept
    indices_to_keep = list(range(num_original_steps))
    steps_removed_count = 0

    logger.info(f"Task {task_name}: Starting O(N) backward pass cleaning with {num_original_steps} steps.")

    # Iterate backwards through the *original* indices
    for i in range(num_original_steps - 1, -1, -1):

        # Check if step 'i' is currently among the ones we are keeping.
        # If not, it was already removed implicitly when testing a later step.
        if i not in indices_to_keep:
            continue

        # Create the list of indices *without* step 'i'
        temp_indices = [idx for idx in indices_to_keep if idx != i]

        # Cannot remove the very last step remaining
        if not temp_indices:
            logger.debug(f"Task {task_name}: Cannot remove step {i}, would result in empty solution.")
            continue

        # Build the temporary solution code from the parts corresponding to temp_indices
        # Ensure steps are joined in their original relative order by sorting indices
        temp_parts = [original_parts[idx] for idx in sorted(temp_indices)]
        # Use variable name similar to greedy version
        new_solution_code = STEP_END.join(temp_parts)

        # Test if the solution passes without step i - similar call structure
        error, passed, _ = execute_code_with_task(new_solution_code, input_grids, expected_outputs)

        if not error and passed:
            # Success! Removing step 'i' (original index) works.
            # Use debug log similar to greedy version's internal success log
            logger.debug(f"Task {task.name}: Successfully removed step with original index {i}")
            indices_to_keep = temp_indices  # Permanently remove index i from our keeper list
            steps_removed_count += 1
        else:
            # Failed: Step 'i' (original index) is necessary given the other steps currently kept.
            logger.debug(f"Task {task.name}: Step with original index {i} is necessary, keeping.")
            # Do nothing, indices_to_keep remains unchanged for the next iteration

    # --- Build final solution ---
    # Use variable names similar to the end of the greedy version
    current_parts = [original_parts[idx] for idx in sorted(indices_to_keep)]
    current_solution_code = STEP_END.join(current_parts)

    # Final log message summarizing the result
    logger.info(
        f"Task {task.name}: Finished O(N) backward pass. Removed {steps_removed_count} steps. Final length: {len(current_parts)} steps."
    )

    # Optional: Final verification (as added in the previous O(N) version)
    # error, passed, _ = execute_code_with_task(current_solution_code, input_grids, expected_outputs)
    # if error or not passed:
    #      logger.warning(f"Task {task_name}: Final code after O(N) cleaning failed verification! Error: {error}, Passed: {passed}. Returning original code.")
    #      return solution_code # Revert to original if final check fails

    return current_solution_code


def main():
    # Create config
    config = Config()

    config.batch_size = max(1, config.cpus - 1)

    setup_logging(config)

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

    # --- Curate Best Solution per Task ---
    logger.info(f"Starting curation of best solutions from {cleaned_file}...")

    # Define the output file path for curated solutions
    curated_file = os.path.join(sft_data_dir, "curated_solutions.jsonl")

    # Dictionary to store the best solution found so far for each task
    # Structure: {task_name: best_solution_code_string}
    best_solutions = {}

    try:
        # Read the cleaned file line by line (efficient for large files)
        with open(cleaned_file, 'r', encoding='utf-8') as infile:
            for line_num, line in enumerate(infile):
                if not line.strip():
                    continue
                try:
                    data = json.loads(line)
                    task_name = data.get("task_name")
                    current_solution_code = data.get("solution_code")

                    # Validate data presence
                    if not task_name or current_solution_code is None:
                        logger.warning(f"Skipping invalid entry in {cleaned_file} line {line_num + 1}: {line.strip()}")
                        continue

                    # --- Heuristic Application ---
                    # Get the best solution currently stored for this task (if any)
                    existing_best_code = best_solutions.get(task_name)

                    # If no solution stored yet, or if the current one is shorter
                    if existing_best_code is None or len(current_solution_code) < len(existing_best_code):
                        # Update the dictionary with the new best solution
                        best_solutions[task_name] = current_solution_code
                        # Optional: Log when a better solution is found for a task
                        # logger.debug(f"Task {task_name}: Found new best solution (length {len(current_solution_code)}).")

                except json.JSONDecodeError:
                    logger.warning(f"Skipping invalid JSON line in {cleaned_file} line {line_num + 1}: {line.strip()}")
                except Exception as e:
                    logger.error(
                        f"Unexpected error processing line {line_num + 1} from {cleaned_file}: {line.strip()} - {e}")

    except FileNotFoundError:
        logger.error(f"Cleaned file not found for curation: {cleaned_file}. Cannot proceed.")
        # Decide if you want to return or exit here based on your workflow
        return  # Exiting main function as curation cannot proceed
    except Exception as e:
        logger.error(f"Failed to read {cleaned_file} for curation: {e}")
        return  # Exiting main function

    # --- Write Curated Solutions ---
    curated_count = 0
    logger.info(f"Found best solutions for {len(best_solutions)} unique tasks. Writing to {curated_file}...")
    try:
        # Write the selected best solutions to the curated file
        with open(curated_file, 'w', encoding='utf-8') as outfile:
            # Sort items by task name for deterministic output order (optional but good practice)
            for task_name, solution_code in sorted(best_solutions.items()):
                entry = json.dumps({"task_name": task_name, "solution_code": solution_code})
                outfile.write(entry + '\n')
                curated_count += 1
        logger.info(f"Finished curation. Selected and saved {curated_count} best solutions to {curated_file}")
    except IOError as e:
        logger.error(f"Failed to write curated solutions to {curated_file}: {e}")


if __name__ == "__main__":
    main()
