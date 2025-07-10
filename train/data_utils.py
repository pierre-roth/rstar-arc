import difflib
import json
import logging
import os
import pathlib
import re
import sys
from collections import defaultdict
from typing import Optional, Any, Callable

import generators
import verifiers

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

from rstar_deepthink.tools.python_tool import remove_markers, execute_code_with_task

logger = logging.getLogger(__name__)


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


def remove_comments(code: str) -> str:
    """Removes single-line and multi-line comments from Python code."""
    code = re.sub(r'#.*', '', code)
    code = re.sub(r'\"\"\"[\s\S]*?\"\"\"', '', code)
    code = re.sub(r"\'\'\'[\s\S]*?\'\'\'", '', code)
    code = "\n".join(line for line in code.splitlines() if line.strip())
    return code


def calculate_avg_q(data: dict) -> float:
    """Calculates the average Q-value from solution metadata."""
    metadata = data.get("metadata")
    if metadata:
        q_values = metadata.get("q_values")
        if q_values and isinstance(q_values, list) and len(q_values) > 0:
            try:
                numeric_q_values = [float(q) for q in q_values if isinstance(q, (int, float))]
                if len(numeric_q_values) == len(q_values):
                    return sum(numeric_q_values) / len(numeric_q_values)
                else:
                    logger.warning(f"Non-numeric Q-values found for task {data.get('task_name')}")
            except (ValueError, TypeError) as e:
                logger.warning(f"Error calculating avg Q for task {data.get('task_name')}: {e}")
                return float("-inf")
    return float("-inf")


def get_code_length(code: str) -> int:
    """Calculates the length of the code after removing comments."""
    # Use remove_markers if solutions might contain them, otherwise use remove_comments
    # code_to_measure = remove_markers(code)
    code_to_measure = code  # Assuming cleaned file doesn't have markers
    return len(remove_comments(code_to_measure))


def calculate_code_similarity(code1: str, code2: str) -> float:
    """Calculates similarity using Levenshtein distance ratio on comment-removed code."""
    code1_clean = remove_comments(code1)
    code2_clean = remove_comments(code2)
    # Lower score = less similar (more diverse)
    return 1.0 - difflib.SequenceMatcher(None, code1_clean, code2_clean).ratio()


def select_diverse_subset(solutions: list[dict], k: int) -> list[dict]:
    """Selects a diverse subset of k solutions using a greedy approach."""
    if not solutions:
        return []
    if len(solutions) <= k:
        return solutions

    for sol in solutions:
        # Precompute clean code for efficiency if not already present
        if 'clean_code' not in sol:
            sol['clean_code'] = remove_comments(sol['solution_code'])

    # Start with the first solution (already sorted by length)
    selected_solutions = [solutions[0]]
    remaining_solutions = solutions[1:]

    while len(selected_solutions) < k and remaining_solutions:
        best_candidate = None
        max_min_distance = -1  # Maximize the minimum distance (dissimilarity)

        for candidate in remaining_solutions:
            min_distance_to_selected = float('inf')
            for selected in selected_solutions:
                distance = calculate_code_similarity(candidate['clean_code'], selected['clean_code'])
                min_distance_to_selected = min(min_distance_to_selected, distance)

            # Maximize the minimum distance
            if min_distance_to_selected > max_min_distance:
                max_min_distance = min_distance_to_selected
                best_candidate = candidate

        if best_candidate:
            selected_solutions.append(best_candidate)
            remaining_solutions.remove(best_candidate)
        else:
            # Fallback: if no candidate improves diversity score, add the next shortest
            if remaining_solutions:
                selected_solutions.append(remaining_solutions.pop(0))
            else:
                break  # No more solutions to add

    # Clean up temporary keys
    for sol in solutions:
        if 'clean_code' in sol:
            del sol['clean_code']

    return selected_solutions


def load_task_info(base_dir: str) -> tuple[dict[str, str], dict[str, set[str]]]:
    """Scans the base directory for ARC task JSON files."""
    task_name_to_path: dict[str, str] = {}
    structure: dict[str, set[str]] = defaultdict(set)
    logger.info(f"Scanning task directory structure: {base_dir}")
    base_path = pathlib.Path(base_dir)

    for filepath in base_path.rglob('*.json'):
        task_name = filepath.stem
        task_name_to_path[task_name] = str(filepath)
        try:
            relative_dir = filepath.parent.relative_to(base_path)
            subdir_key = str(relative_dir).split(os.path.sep)[0] if relative_dir.parts else '.'
            structure[subdir_key].add(task_name)
        except ValueError:
            logger.warning(
                f"Could not determine relative path for {filepath} against base {base_path}. Skipping structure assignment.")
            structure['.'].add(task_name)

    logger.info(f"Finished scanning. Found {len(task_name_to_path)} tasks in subdirectories: {list(structure.keys())}")
    return task_name_to_path, dict(structure)


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


def load_rearc_examples(rearc_dir: str, task_name: str) -> Optional[list[dict[str, Any]]]:
    """Loads reARC data for a specific task."""
    rearc_file = os.path.join(rearc_dir, f"{task_name}.json")
    if not os.path.exists(rearc_file):
        logger.debug(f"No reARC data file for task {task_name} at {rearc_file}")
        return None
    data = load_json_file(rearc_file)
    if isinstance(data, list) and all(isinstance(item, dict) for item in data):
        return data
    elif data is not None:
        logger.warning(f"reARC data for task {task_name} is not a list of dicts. Skipping.")
        return None
    return None


def test_solution_on_rearc_example(solution_code: str, rearc_example: dict[str, Any]) -> bool:
    """Tests a solution against a single reARC example."""
    try:
        code_to_execute = remove_markers(solution_code)  # Keep remove_markers if solutions might have them
        # code_to_execute = solution_code # Use if solutions are known to be clean
        error, passed, _ = execute_code_with_task(
            code_to_execute,
            [rearc_example['input']],
            [rearc_example['output']])
        return not error and passed
    except Exception as e:
        logger.debug(f"Exception during solution test: {e}")
        return False


def verify_with_rearc_verifier(verifier_func: Callable, example: dict[str, Any]) -> bool:
    """Tests a generated example against its specific reARC verifier."""
    if not verifier_func:
        logger.warning("Verifier function not provided, cannot verify.")
        return False  # Or True, depending on desired behavior if verifier is missing
    try:
        # Assuming verifier functions take 'input' grid and return 'output' grid
        return verifier_func(example['input']) == example['output']
    except Exception as e:
        logger.debug(f"Exception during reARC verification: {e}")
        return False


def write_batch_data(filepath: str, data_batch: list[dict]):
    """Writes a batch of JSON lines to the output file."""
    if not data_batch:
        return
    try:
        with open(filepath, 'a', encoding='utf-8') as f:
            for data in data_batch:
                f.write(json.dumps(data) + '\n')
    except IOError as e:
        logger.error(f"Failed to write batch to augmented data file {filepath}: {e}")
    except TypeError as e:
        logger.error(f"Data serialization error: {e}. Data: {data_batch[:1]}")


def get_rearc_generator(task_name: str) -> Optional[Callable]:
    """Gets the reARC generator function for a given task name."""
    generator_name = f"generate_{task_name}"
    return getattr(generators, generator_name, None)


def get_rearc_verifier(task_name: str) -> Optional[Callable]:
    """Gets the reARC verifier function for a given task name."""
    verifier_name = f"verify_{task_name}"
    return getattr(verifiers, verifier_name, None)


def pair_valid(pair: dict[str, Any], config) -> bool:
    """Checks if a reARC pair is valid."""
    if not pair.get("metadata", {}).get("chosen_q") or not pair.get("metadata", {}).get("rejected_q"):
        return False
    elif not pair["metadata"]["chosen_q"] > pair["metadata"]["rejected_q"] + config.min_step_margin:
        return False

    return True


