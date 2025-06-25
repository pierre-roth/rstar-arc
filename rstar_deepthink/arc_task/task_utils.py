import json
import logging
import os
import sys
from random import shuffle

from rstar_deepthink.arc_task import ARCTask
from rstar_deepthink.config import Config
from rstar_deepthink.prompt.prompt_utils import task_to_prompt

logger = logging.getLogger(__name__)


def load_tasks(config: Config) -> list[ARCTask]:
    """
    Load all ARC tasks from the configured data folder

    This method:
    1. Verifies the data directory exists
    2. Finds all JSON files in the directory
    3. Sorts them alphabetically (case-insensitive)
    4. Loads each file as an ARCTask object

    Returns:
        List of ARCTask objects

    Raises:
        SystemExit: If directory doesn't exist or no JSON files are found
    """
    # Verify the data directory exists
    if not os.path.isdir(config.data_folder):
        logger.error(f"Directory '{config.data_folder}' not found.")
        sys.exit(1)

    # Get all JSON files and sort them alphabetically (case-insensitive)
    files = [f for f in os.listdir(config.data_folder) if f.endswith('.json')]

    if config.solve_only_unsolved:
        # remove already solved tasks
        if not config.evaluation:
            solutions_file = os.path.join(config.sft_data_dir, f"round_{config.round_number}", "solutions_training.jsonl")
        else:
            solutions_file = os.path.join(config.sft_data_dir, f"round_{config.round_number}", "solutions_evaluation.jsonl")

        solved_set = set()
        # create empty solutions-file if it does not exist
        if not os.path.exists(solutions_file):
            with open(solutions_file, "w", encoding="utf-8") as f:
                pass

        with open(solutions_file, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                data = json.loads(line)
                task_name = data["task_name"]
                solved_set.add(task_name)

        # Add example tasks to the solved set
        """for file_name in os.listdir(DEFAULT_EXAMPLE_DATA_PATH):
            if file_name.endswith(".json"):
                task_name = os.path.splitext(file_name)[0]
                solved_set.add(task_name)"""

        files = [f for f in files if os.path.splitext(f)[0] not in solved_set]

    if config.task_names is not None:
        # Filter files based on task names
        files = [f for f in files if os.path.splitext(f)[0] in config.task_names]

    # Load each file as an ARCTask
    tasks = []
    for file_name in files:
        task_file_path = os.path.join(config.data_folder, file_name)
        task = ARCTask(config, task_file_path)
        tasks.append(task)

    if config.sort_by_length:
        tasks.sort(key=lambda t: len(task_to_prompt(t)))
    else:
        shuffle(tasks)

    if config.num_tasks > 0:
        tasks = tasks[:config.num_tasks]

    # Ensure we found at least one file
    if not tasks:
        logger.error(f"No JSON files found in directory '{config.data_folder}'.")
        sys.exit(1)

    logger.info(f"Processing {len(tasks)} (randomly chosen) tasks from '{config.data_folder}'")

    return tasks


def filter_tasks_by_length(
    tasks: list[ARCTask],
    tokenizer,
    config: Config,
    agent_cls,
) -> list[ARCTask]:
    """Filter out tasks whose maximum context length would exceed the model window."""

    if not config.length_pre_filtering:
        return tasks

    filtered: list[ARCTask] = []
    for task in tasks:
        agent = agent_cls(config, task)
        agent.update(0, config.policy_temperature)
        prompt = agent.root.collect_prompt_and_code()

        length = len(tokenizer.encode(prompt))

        total = length + config.max_depth * config.max_tokens
        if total <= config.max_seq_len:
            filtered.append(task)
        else:
            logger.warning(
                f"Skipping task {task.name} as maximum possible length {total} would exceed context size {config.max_seq_len}"
            )

    return filtered

