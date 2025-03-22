import logging
import os
import sys

from arc_rstar.arc_task.task import ARCTask
from config import Config

logger = logging.getLogger(__name__)


def batch(iterable, n=-1):
    l = len(iterable)
    if n <= 0:
        n = l
    for ndx in range(0, l, n):
        yield iterable[ndx: min(ndx + n, l)]


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
    files = sorted([f for f in os.listdir(config.data_folder) if f.endswith('.json')],
                   key=lambda x: x.lower())

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

    return tasks


def setup_logging(config: Config):
    """
    Set up the logging configuration based on settings.

    This configures the Python logging system according to the configuration settings,
    creating file and console handlers with appropriate formats.
    """

    # Create logger
    _logger = logging.getLogger()
    _logger.setLevel(config.numeric_log_level)

    # Clear any existing handlers
    for handler in _logger.handlers[:]:
        _logger.removeHandler(handler)

    log_format_str = '{asctime} | {levelname:<8} | {name}:{lineno} | {message}'

    formatter = logging.Formatter(log_format_str, style='{')

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    _logger.addHandler(console_handler)

    # Log some basic information at startup
    logging.info(f"rStar-ARC initialized with job ID: {config.job_id}")
    logging.info(f"Search mode: {config.search_mode}")
