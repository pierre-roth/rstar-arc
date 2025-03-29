import logging
import os
import sys

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

    # Ensure we found at least one file
    if not files:
        logger.error(f"No JSON files found in directory '{config.data_folder}'.")
        sys.exit(1)

    logger.info(f"Found {len(files)} JSON files in '{config.data_folder}'")

    # Load each file as an ARCTask
    tasks = []
    for file_name in files:
        task_file_path = os.path.join(config.data_folder, file_name)
        task = ARCTask(config, task_file_path)
        tasks.append(task)

    tasks.sort(key=lambda task: len(task_to_prompt(task)))

    return tasks
