import difflib
import heapq
import json
import logging
import os
import pathlib
import random
import re
import sys
from collections import defaultdict
from concurrent.futures import TimeoutError
from typing import Optional, Dict, List, Any, Set, Tuple, Callable

from pebble import ProcessPool

# --- Project Setup ---
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

from constants import NET_SCRATCH_TASK_DATA_DIR_PATH, NET_SCRATCH_RE_ARC_DATA_PATH
from rstar_deepthink.config import Config
from rstar_deepthink.tools import execute_code_with_task
# remove_markers is specific to rstar_deepthink, keep it if solutions have markers
from rstar_deepthink.tools.python_tool import remove_markers
from utils import setup_logging
# Import reARC generators and verifiers
import generators
import verifiers

logger = logging.getLogger(__name__)

# --- Constants ---
NUM_SOLUTIONS_Q_VALUE = 64
NUM_SOLUTIONS_LENGTH = 32
NUM_SOLUTIONS_DIVERSITY = 8
TARGET_EXAMPLES_PER_TASK = 100  # M value
MAX_GENERATION_ATTEMPTS = 10  # N value
VERIFY_GENERATED_EXAMPLES = True  # Boolean flag for additional verification
BATCH_SIZE = 100
TIMEOUT = 10800


# --- Helper Functions ---

def remove_comments(code: str) -> str:
    """Removes single-line and multi-line comments from Python code."""
    code = re.sub(r'#.*', '', code)
    code = re.sub(r'\"\"\"[\s\S]*?\"\"\"', '', code)
    code = re.sub(r"\'\'\'[\s\S]*?\'\'\'", '', code)
    code = "\n".join(line for line in code.splitlines() if line.strip())
    return code


def calculate_avg_q(data: Dict) -> float:
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


def select_diverse_subset(solutions: List[Dict], k: int) -> List[Dict]:
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
        if 'clean_code' in sol: del sol['clean_code']

    return selected_solutions


def load_task_info(base_dir: str) -> Tuple[Dict[str, str], Dict[str, Set[str]]]:
    """Scans the base directory for ARC task JSON files."""
    task_name_to_path: Dict[str, str] = {}
    structure: Dict[str, Set[str]] = defaultdict(set)
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


def load_rearc_examples(rearc_dir: str, task_name: str) -> Optional[List[Dict[str, Any]]]:
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


def test_solution_on_rearc_example(solution_code: str, rearc_example: Dict[str, Any]) -> bool:
    """Tests a solution against a single reARC example."""
    try:
        code_to_execute = remove_markers(solution_code)  # Keep remove_markers if solutions might have them
        # code_to_execute = solution_code # Use if solutions are known to be clean
        error, passed, _ = execute_code_with_task(code_to_execute, [rearc_example['input']],
                                                  [rearc_example['output']])
        return not error and passed
    except Exception as e:
        logger.debug(f"Exception during solution test: {e}")
        return False


def verify_with_rearc_verifier(verifier_func: Callable, example: Dict[str, Any]) -> bool:
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


def write_batch_data(filepath: str, data_batch: List[Dict]):
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


# --- Main Processing Function (for Parallel Execution) ---
def process_task_augmentation(job_data: Tuple[str, List[Dict], str, int, int]) -> List[Dict]:
    """Processes augmentation for a single task using its curated solutions."""
    task_name, curated_solutions, rearc_data_dir, target_examples_m, max_attempts_n = job_data
    logger.debug(
        f"Processing task: {task_name} with {len(curated_solutions)} solutions. M={target_examples_m}, N={max_attempts_n}")

    assigned_examples: Dict[int, List[Dict]] = defaultdict(list)
    solved_example_identifiers: Set[str] = set()
    solved_count = 0
    tried_count = 0

    solution_indices = list(range(len(curated_solutions)))

    # 1. Process Existing reARC Examples
    existing_examples = load_rearc_examples(rearc_data_dir, task_name)
    if existing_examples:
        logger.debug(f"Task {task_name}: Found {len(existing_examples)} existing reARC examples.")

        for example in existing_examples:
            tried_count += 1
            example_identifier = json.dumps(example, sort_keys=True)
            if example_identifier in solved_example_identifiers:
                continue

            random.shuffle(solution_indices)
            for sol_index in solution_indices:
                solution_data = curated_solutions[sol_index]
                if test_solution_on_rearc_example(solution_data['solution_code'], example):
                    assigned_examples[sol_index].append(example)
                    solved_example_identifiers.add(example_identifier)
                    solved_count += 1
                    break
        logger.debug(f"Task {task_name}: Processed existing examples. Solved count: {solved_count}")

    # 2. Generate New reARC Examples if Needed
    rearc_generator = get_rearc_generator(task_name)
    rearc_verifier = get_rearc_verifier(task_name) if VERIFY_GENERATED_EXAMPLES else None

    if not rearc_generator:
        logger.warning(f"Task {task_name}: Cannot generate new examples, generator not found.")
    elif VERIFY_GENERATED_EXAMPLES and not rearc_verifier:
        logger.warning(
            f"Task {task_name}: Cannot verify generated examples, verifier not found (VERIFY_GENERATED_EXAMPLES=True).")
    else:
        logger.debug(
            f"Task {task_name}: Current solved: {solved_count}, Target: {target_examples_m}, Max attempts: {max_attempts_n}")
        generation_attempts = 0  # Track attempts separate from existing examples tested

        while solved_count < target_examples_m and generation_attempts < max_attempts_n:
            generation_attempts += 1
            tried_count += 1  # Increment total attempts

            try:
                diff_lb = random.random() * 0.5  # Skew towards lower difficulty
                diff_ub = random.uniform(diff_lb, 1.0)
                new_example = rearc_generator(diff_lb=diff_lb, diff_ub=diff_ub)

                if not new_example or 'input' not in new_example or 'output' not in new_example:
                    logger.warning(
                        f"Task {task_name}: Generator produced invalid structure. Attempt {generation_attempts}")
                    continue

                example_identifier = json.dumps(new_example, sort_keys=True)
                if example_identifier in solved_example_identifiers:
                    continue

                # Optional: Verify with reARC verifier first
                if VERIFY_GENERATED_EXAMPLES:
                    if not verify_with_rearc_verifier(rearc_verifier, new_example):
                        # logger.debug(f"Task {task_name}: Generated example failed verification.")
                        continue  # Skip example if verification fails

                # Test against curated solutions
                random.shuffle(solution_indices)
                for sol_index in solution_indices:
                    solution_data = curated_solutions[sol_index]
                    if test_solution_on_rearc_example(solution_data['solution_code'], new_example):
                        assigned_examples[sol_index].append(new_example)
                        solved_example_identifiers.add(example_identifier)
                        solved_count += 1
                        break

            except Exception as e:
                logger.error(f"Task {task_name}: Error during generation/testing attempt {generation_attempts}: {e}",
                             exc_info=False)

        if solved_count < target_examples_m:
            logger.warning(
                f"Task {task_name}: Reached generation attempt limit ({generation_attempts}/{max_attempts_n}) or total limit ({tried_count}/{max_attempts_n}). Found {solved_count}/{target_examples_m} examples.")
        else:
            logger.debug(
                f"Task {task_name}: Reached target examples ({solved_count}/{target_examples_m}). Total attempts: {tried_count}")

    # 3. Format Output
    output_batch = []
    for sol_index, examples in assigned_examples.items():
        if examples:
            solution_data = curated_solutions[sol_index]
            output_data = {
                "task_name": f"{task_name}_curated_sol_{sol_index}",
                "original_task_name": task_name,
                "solution_code": solution_data['solution_code'],
                "examples": examples,
                "metadata": solution_data.get("metadata")
            }
            output_batch.append(output_data)

    logger.debug(f"Task {task_name}: Finished processing. Generated {len(output_batch)} output entries.")
    return output_batch


# --- Main Orchestration ---
def main(config: Config):
    """Main function orchestrating the curation and augmentation process."""
    setup_logging(config.numeric_log_level)
    logger.info("Starting SFT data augmentation process with curation...")
    logger.info(f"VERIFY_GENERATED_EXAMPLES set to: {VERIFY_GENERATED_EXAMPLES}")
    logger.info(f"Target Examples (M): {TARGET_EXAMPLES_PER_TASK}, Max Attempts (N): {MAX_GENERATION_ATTEMPTS}")

    # --- Define Paths (Using Config) ---
    sft_data_dir = os.path.join(config.sft_data_dir, f"round_{config.round_number}")
    os.makedirs(sft_data_dir, exist_ok=True)
    cleaned_file = os.path.join(sft_data_dir, "cleaned.jsonl")
    augmented_file = os.path.join(sft_data_dir, "augmented.jsonl")
    # Paths below might not be in Config, using constants/defaults
    arc_tasks_base_dir = NET_SCRATCH_TASK_DATA_DIR_PATH
    rearc_data_dir = NET_SCRATCH_RE_ARC_DATA_PATH

    # --- Scan Task Directory ---
    task_name_to_path, directory_structure = load_task_info(arc_tasks_base_dir)
    training_task_names = directory_structure.get('training', set())
    logger.info(f"Identified {len(training_task_names)} training tasks.")

    # --- Load and Group Cleaned Solutions by Task ---
    solutions_by_task: Dict[str, List[Dict]] = defaultdict(list)
    # ... (loading logic remains the same as previous version) ...
    try:
        with open(cleaned_file, 'r', encoding='utf-8') as infile:
            for line_num, line in enumerate(infile):
                if not line.strip():
                    continue
                try:
                    data = json.loads(line)
                    task_name = data.get("task_name")
                    if task_name and task_name in task_name_to_path:
                        if "solution_code" in data:
                            solutions_by_task[task_name].append(data)
                        else:
                            logger.warning(f"Skipping line {line_num + 1} in {cleaned_file}: Missing 'solution_code'.")
                    # Allow non-ARC tasks through, they just won't be curated/augmented if not in training_task_names
                    # elif task_name:
                    #     logger.warning(f"Task '{task_name}' from cleaned file line {line_num+1} not found in task directory scan.")
                except json.JSONDecodeError:
                    logger.warning(f"Skipping invalid JSON in cleaned file line {line_num + 1}: {line.strip()}")
    except FileNotFoundError:
        logger.error(f"Cleaned input file not found: {cleaned_file}")
        sys.exit(1)
    logger.info(f"Loaded solutions for {len(solutions_by_task)} tasks from {cleaned_file}.")

    # --- Curate Solutions for Each Training Task ---
    curated_solutions_per_task: Dict[str, List[Dict]] = {}
    # ... (curation logic remains the same: Q-val -> Length -> Diversity) ...
    for task_name, solutions in solutions_by_task.items():
        if task_name not in training_task_names:
            continue

        logger.debug(f"Curating solutions for task: {task_name} ({len(solutions)} found)")
        solutions_with_q = [(calculate_avg_q(sol), sol) for sol in solutions]
        top_q_solutions = [sol for q, sol in
                           heapq.nlargest(NUM_SOLUTIONS_Q_VALUE, solutions_with_q, key=lambda x: x[0])]

        for sol in top_q_solutions:
            sol['temp_length'] = get_code_length(sol['solution_code'])
        top_q_solutions.sort(key=lambda x: x['temp_length'])
        shortest_solutions = top_q_solutions[:NUM_SOLUTIONS_LENGTH]
        for sol in top_q_solutions: del sol['temp_length']  # Clean up temp key

        # Precompute clean_code for diversity selection
        for sol in shortest_solutions:
            sol['clean_code'] = remove_comments(sol['solution_code'])
        diverse_solutions = select_diverse_subset(shortest_solutions, NUM_SOLUTIONS_DIVERSITY)
        # Clean up temporary key
        for sol in shortest_solutions:
            if 'clean_code' in sol: del sol['clean_code']

        logger.debug(f"Task {task_name}: Selected {len(diverse_solutions)} diverse solutions.")
        if diverse_solutions:
            curated_solutions_per_task[task_name] = diverse_solutions

    logger.info(f"Finished curation for {len(curated_solutions_per_task)} training tasks.")

    # --- Prepare Augmentation Jobs (Task-Based) ---
    augmentation_jobs = []
    for task_name, curated_solutions in curated_solutions_per_task.items():
        job = (task_name, curated_solutions, rearc_data_dir, TARGET_EXAMPLES_PER_TASK, MAX_GENERATION_ATTEMPTS)
        augmentation_jobs.append(job)
    logger.info(f"Prepared {len(augmentation_jobs)} task-based augmentation jobs.")

    # --- Clear Output File ---
    try:
        with open(augmented_file, 'w') as f:
            pass
        logger.info(f"Cleared/Created output file: {augmented_file}")
    except IOError as e:
        logger.error(f"Could not open or clear output file {augmented_file}: {e}")
        sys.exit(1)

    # --- Parallel Augmentation ---
    total_augmented_entries_saved = 0
    if augmentation_jobs:
        num_workers = max(1, config.cpus - 1 if config.cpus > 1 else 1)
        logger.info(f"Starting parallel augmentation using {num_workers} workers...")
        task_timeout = TIMEOUT

        results_batch_to_write = []
        processed_jobs_count = 0
        try:
            with ProcessPool(max_workers=num_workers) as pool:
                future = pool.map(process_task_augmentation, augmentation_jobs, timeout=task_timeout)
                results_iterator = future.result()

                while True:
                    try:
                        task_output_batch = next(results_iterator)
                        processed_jobs_count += 1
                        if task_output_batch:
                            results_batch_to_write.extend(task_output_batch)
                            total_augmented_entries_saved += len(task_output_batch)

                            if len(results_batch_to_write) >= BATCH_SIZE:
                                write_batch_data(augmented_file, results_batch_to_write)
                                logger.info(
                                    f"Written {len(results_batch_to_write)} entries. Progress: {processed_jobs_count}/{len(augmentation_jobs)} tasks.")
                                results_batch_to_write = []

                        logger.info(f"Completed processing task {processed_jobs_count}/{len(augmentation_jobs)}...")

                    except StopIteration:
                        break
                    except TimeoutError as error:
                        logger.error(f"A task job timed out after {error.args[1]}s. Skipping results.")
                        processed_jobs_count += 1  # Assume timeout advances iterator
                    except Exception as error:
                        logger.error(f"A task job failed: {error}", exc_info=False)
                        processed_jobs_count += 1  # Assume failure advances iterator

            if results_batch_to_write:
                write_batch_data(augmented_file, results_batch_to_write)
                logger.info(f"Written final {len(results_batch_to_write)} entries.")

            logger.info(f"Finished parallel processing.")

        except Exception as e:
            logger.error(f"Error during parallel pool execution: {e}", exc_info=True)
    else:
        logger.info("No augmentation jobs to run.")

    logger.info(f"--- Augmentation Summary ---")
    logger.info(f"Total curated solution entries generated & saved: {total_augmented_entries_saved}")
    logger.info(f"Total entries written to {augmented_file}")


if __name__ == "__main__":
    # Instantiate config directly, assuming it reads from environment or defaults
    config_instance = Config()

    main(config_instance)
