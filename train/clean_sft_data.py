import json
import logging
import os
from typing import Any

from pebble import ProcessPool

from constants import NET_SCRATCH_PATH, STEP_END
from rstar_deepthink.arc_task import ARCTask
from rstar_deepthink.config import Config
from rstar_deepthink.tools import execute_code_with_task
from utils import batch

logger = logging.getLogger(__name__)


def process_solution(args: (Config, ARCTask, str)) -> str:
    """
    Process a solution to remove unnecessary steps in a greedy way.

    Args:
        args: Tuple containing (config, task, solution_code)

    Returns:
        The cleaned solution code with minimal necessary steps
    """
    config, task, solution_code = args

    # Split the solution into parts (by STEP_END)
    parts = solution_code.split(STEP_END)

    # If there's only one step or fewer, we can't remove anything
    if len(parts) <= 1:
        return solution_code

    input_grids = [example.input_grid.grid for example in task.training_examples + task.test_examples]
    expected_outputs = [example.output_grid.grid for example in task.training_examples + task.test_examples]

    # Try removing each step one by one
    changed = True
    while changed:
        changed = False
        for i in range(len(parts) - 1):  # Last part doesn't have STEP_END marker
            # Create a new solution without the step
            new_parts = parts.copy()
            del new_parts[i]
            new_solution_code = STEP_END.join(new_parts)

            # Test if the solution passes examples
            error, passed, _ = execute_code_with_task(solution_code, input_grids, expected_outputs)

            if not error and passed:
                logger.info(f"Removed step {i} from solution")
                solution_code = new_solution_code
                parts = solution_code.split(STEP_END)
                changed = True
                break

    return solution_code


def process_batch(config: Config, lines: list[dict[str, Any]], task_dict: dict[str, Any], out_file):
    """
    Process a batch of solutions.

    Args:
        config: The configuration object
        lines: A list of dictionaries with task_name and solution_code
        task_dict: Dictionary mapping task names to task objects
        out_file: Output file to write results to
    """
    # For each task in batch, get the corresponding task
    tasks_and_solutions = []
    for data in lines:
        task_name = data["task_name"]
        solution_code = data["solution_code"]

        task = task_dict.get(task_name)
        if not task:
            logger.warning(f"Task {task_name} not found in loaded tasks")
            continue

        tasks_and_solutions.append((task, task_name, solution_code))

    # Using process pool, for each solution, try leaving out steps
    with ProcessPool(max_workers=config.cpus) as pool:
        futures = []
        for task, task_name, solution_code in tasks_and_solutions:
            future = pool.schedule(process_solution, args=(config, task, solution_code))
            futures.append((task_name, future))

        # Save the cleaned solution to a new file
        for task_name, future in futures:
            try:
                cleaned_code = future.result()
                entry = json.dumps({"task_name": task_name, "solution_code": cleaned_code})
                out_file.write(entry + '\n')
                logger.info(f"Processed and saved solution for task {task_name}")
            except Exception as e:
                logger.error(f"Error processing task {task_name}: {e}")


def main():
    # Create config
    config = Config()

    config.batch_size = config.cpus - 1

    # Create directory for cleaned data if it doesn't exist
    raw_file = os.path.join(NET_SCRATCH_PATH, "sft_data", f"round_{config.round_number}", "raw.jsonl")
    cleaned_file = os.path.join(NET_SCRATCH_PATH, "sft_data", f"round_{config.round_number}", "cleaned.jsonl")
    task_dir = os.path.join(NET_SCRATCH_PATH, "task_data")

    logger.info(f"Starting SFT data cleaning from {raw_file} to {cleaned_file}")

    solved_set = set()
    with open(raw_file, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            data = json.loads(line)
            task_name = data["task_name"]
            solved_set.add(task_name)

    name_to_task = {}
    # walk the task_data dir
    for dir_name in os.listdir(task_dir):
        dir_path = os.path.join(task_dir, dir_name)
        if os.path.isdir(dir_path):
            for task_file in os.listdir(dir_path):
                if task_file.endswith(".json") and task_file.removesuffix(".json") in solved_set:
                    task_path = os.path.join(dir_path, task_file)
                    task = ARCTask(config, task_path)
                    name_to_task[task.name] = task

    # Open the output file
    with open(cleaned_file, 'w') as out:
        # Read the raw.jsonl file in batches
        batch_size = config.cpus - 1

        logger.info(f"Opening raw.jsonl file: {raw_file}")
        with open(raw_file, 'r') as f:

            for lines in batch(f, batch_size):
                pass

            for line in f:
                if not line.strip():
                    continue

                # Parse line as JSON
                data = json.loads(line)
                batch.append(data)

                if len(batch) >= batch_size:
                    logger.info(f"Processing batch of {len(batch)} solutions")
                    process_batch(config, batch, task_dict, out)
                    batch = []

            # Process the remaining data
            if batch:
                logger.info(f"Processing final batch of {len(batch)} solutions")
                process_batch(config, batch, task_dict, out)

    logger.info(f"Finished cleaning SFT data. Output saved to {cleaned_file}")


if __name__ == "__main__":
    main()
