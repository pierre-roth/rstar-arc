import json
import logging
import os
import sys

from arc_rstar.agents import Node
from arc_rstar.arc_task.task import ARCTask
from arc_rstar.tools.python_tool import extract_python_code
from config import Config, STEP_END, CODE_END

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
    logger.info(f"rStar-ARC initialized with job ID: {config.job_id}")
    logger.info(f"Search mode: {config.search_mode}")


def make_serializable(obj):
    """
    Recursively convert an object into something JSON serializable.
    - For basic types (str, int, float, bool, None) returns the object as is.
    - For lists/tuples, converts each element.
    - For dicts, converts keys and values.
    - For objects with __dict__, returns a dict of its public attributes.
    - Otherwise, returns the string representation.
    """
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    elif isinstance(obj, list):
        return [make_serializable(item) for item in obj]
    elif isinstance(obj, tuple):
        # JSON doesn't support tuples directly, but they'll be converted to lists
        return [make_serializable(item) for item in obj]
    elif isinstance(obj, dict):
        return {make_serializable(key): make_serializable(value) for key, value in obj.items()}
    # elif hasattr(obj, '__dict__'):
    # return {key: make_serializable(val) for key, val in obj.__dict__.items() if not key.startswith("_")}
    else:
        return str(obj)


def serialize_nodes(nodes):
    """
    Convert a list of nodes to a dictionary keyed by node tag.
    For each node, we save all public attributes. For the 'parent' attribute,
    we save parent's tag (or None); for 'children', we save a list of child tags;
    for 'task', we save the task name (if available).
    """
    data = {}
    for node in nodes:
        node_data = {}
        for key, value in node.__dict__.items():
            if key == "parent":
                node_data[key] = value.tag if value is not None else None
            elif key == "children":
                node_data[key] = [child.tag for child in value]
            elif key == "task":
                node_data[key] = value.name if value is not None else None
            elif key == "config":
                node_data[key] = value.config_file if value is not None else None
            else:
                node_data[key] = make_serializable(value)
        data[node.tag] = node_data
    return data


def save_nodes(config, nodes):
    """
    Save a list of nodes to a JSON file.
    The file name is built using the task name from the first node.
    """
    task_name = nodes[0].task.name  # assumes that nodes list is non-empty
    filename = os.path.join(config.temporary_path, f"{task_name}_nodes.json")
    data = serialize_nodes(nodes)
    with open(filename, 'w') as f:
        json.dump(data, f, indent=4)
    print(f"Nodes saved to {filename}")


def load_nodes(filename):
    """
    Load nodes from a JSON file.
    The file is assumed to be a dictionary mapping node tags to node state.
    New Node objects are created (without calling __init__) and then their __dict__
    is updated with the stored state. Then, parent and children attributes are
    replaced with the corresponding Node objects.
    """
    with open(filename, 'r') as f:
        data = json.load(f)

    # First pass: create node instances and store in a mapping.
    nodes_by_tag = {}
    for tag, state in data.items():
        # Create a new instance without calling __init__
        node_instance = Node.__new__(Node)
        node_instance.__dict__.update(state)
        nodes_by_tag[tag] = node_instance

    # Second pass: fix up parent and children references.
    for node in nodes_by_tag.values():
        if node.__dict__.get("parent") is not None:
            parent_tag = node.__dict__["parent"]
            node.__dict__["parent"] = nodes_by_tag.get(parent_tag)
        if "children" in node.__dict__:
            node.__dict__["children"] = [nodes_by_tag.get(child_tag) for child_tag in node.__dict__["children"]]

    return list(nodes_by_tag.values())


def save_summary(config, node_lists: list[list[Node]], batch_number: int):
    result = []
    for nodelist in node_lists:

        task_result = []

        valid_final_answer_nodes = [node for node in nodelist if node.is_valid_final_answer_node()]

        correct_answer_nodes = []
        for node in valid_final_answer_nodes:
            code = extract_python_code(node.collect_partial_solution())
            error, passed, outputs = node.task.run_test_examples(code)
            if passed:
                correct_answer_nodes.append(node)

        # sort nodes by solution length
        correct_answer_nodes.sort(key=lambda node: len(node.collect_partial_solution().split()))

        if correct_answer_nodes:
            task_result.append(
                f"### Found {len(correct_answer_nodes)} correct solutions for task {correct_answer_nodes[0].task.name} ###")
            for i, node in enumerate(correct_answer_nodes):
                task_result.append(f"## Solution {i + 1} ##")
                task_result.append(
                    extract_python_code(node.collect_partial_solution()).replace(STEP_END, f"# {STEP_END}\n").replace(
                        CODE_END, "") + "\n")
        else:
            task_result.append(f"### No correct solutions found for task {nodelist[0].task.name} ###")

        result.append("\n".join(task_result))

    with open(os.path.join(config.temporary_path, f"summary_{batch_number + 1}.py"), "w") as f:
        f.write("\n\n\n\n".join(result))
