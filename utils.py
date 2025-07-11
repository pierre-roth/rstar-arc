import json
import logging
import os
import sys
import itertools

from rstar_deepthink.config import Config
from rstar_deepthink.node import Node
from rstar_deepthink.tools import test_correct
from rstar_deepthink.tools.python_tool import comment_out_markers

logger = logging.getLogger(__name__)


def batch(iterable, n=-1):
    if n <= 0:
        # Convert the entire iterable into a single list (batch).
        # WARNING: This reads EVERYTHING into memory if n <= 0.
        full_batch = list(iterable)
        # Only yield if there was actually content, mimicking iterator exhaustion
        if full_batch:
            yield full_batch
        # Stop iteration after yielding the single batch (or nothing if empty)
        return

    # --- Standard iterator batching for n > 0 ---
    # Ensure we have an iterator
    it = iter(iterable)
    while True:
        # Read n items using islice (doesn't need len)
        chunk = list(itertools.islice(it, n))
        if not chunk:
            # Iterator is exhausted
            return
        # Yield the 0 < sized <= n chunk
        yield chunk


def setup_logging(numeric_log_level: int) -> None:
    """
    Set up the logging configuration based on settings.

    This configures the Python logging system according to the configuration settings,
    creating file and console handlers with appropriate formats.
    """

    # Create logger
    _logger = logging.getLogger()
    _logger.setLevel(numeric_log_level)

    # Clear any existing handlers
    for handler in _logger.handlers[:]:
        _logger.removeHandler(handler)

    log_format_str = '{asctime} | {levelname:<8} | {name}:{lineno} | {message}'

    formatter = logging.Formatter(log_format_str, style='{')

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    _logger.addHandler(console_handler)


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
    elif hasattr(obj, '__dict__'):
        return {key: make_serializable(val) for key, val in obj.__dict__.items()}
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
            elif key == "execution_outputs":
                continue
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
    filename = os.path.join(config.local_job_dir, f"{task_name}_nodes.json")
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


def save_summary(config: Config, node_lists: list[list[Node]], batch_number: int):
    result = []
    num_solved = 0
    for nodelist in node_lists:

        task_result = []

        valid_final_answer_nodes = [node for node in nodelist if node.is_valid_final_answer_node()]

        correct_answer_nodes = []
        for node in valid_final_answer_nodes:
            error, passed, _ = test_correct(node)
            if passed:
                correct_answer_nodes.append(node)

        # sort nodes by solution length
        correct_answer_nodes.sort(key=lambda node: len(node.collect_code()))

        if correct_answer_nodes:
            num_solved += 1
            task_result.append(
                f"### Found {len(correct_answer_nodes)} correct solutions for task {correct_answer_nodes[0].task.name} ###")
            for i, node in enumerate(correct_answer_nodes):
                task_result.append(f"## Solution {i + 1} ##")
                task_result.append(comment_out_markers(node.collect_code() + "\n"))
        else:
            task_result.append(f"### No correct solutions found for task {nodelist[0].task.name} ###")

        result.append("\n".join(task_result))

    logger.info(f"Batch {batch_number + 1} summary: {num_solved} tasks solved out of {len(node_lists)}")

    with open(os.path.join(config.final_job_dir, f"summary_{batch_number + 1}.py"), "w") as f:
        f.write(
            f"# Batch {batch_number + 1} summary: {num_solved} tasks solved out of {len(node_lists)}\n\n" + "\n\n\n\n".join(
                result))
