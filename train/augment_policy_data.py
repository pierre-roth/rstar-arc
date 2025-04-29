import heapq
import os
import random
import sys
from concurrent.futures import TimeoutError
from typing import Dict, List, Set, Tuple

from pebble import ProcessPool

# --- Project Setup ---
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

from constants import NET_SCRATCH_TASK_DATA_DIR, NET_SCRATCH_RE_ARC_DATA
from rstar_deepthink.config import Config
# remove_markers is specific to rstar_deepthink, keep it if solutions have markers
from utils import setup_logging
from data_utils import *

logger = logging.getLogger(__name__)
# Disable subprocess execution to verify examples a lot faster!
# rstar_deepthink.tools.python_tool.use_subprocess = False  # Disable subprocess for this script

# --- Constants ---
NUM_SOLUTIONS_Q_VALUE = 32
NUM_SOLUTIONS_LENGTH = 16
NUM_SOLUTIONS_DIVERSITY = 4
TARGET_EXAMPLES_PER_TASK = 1000  # M value
MAX_GENERATION_ATTEMPTS = 0  # N value
VERIFY_GENERATED_EXAMPLES = True  # Boolean flag for additional verification
BATCH_SIZE = 100
TIMEOUT = 10800


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
    logger.info("Starting policy data augmentation process with curation...")
    logger.info(f"VERIFY_GENERATED_EXAMPLES set to: {VERIFY_GENERATED_EXAMPLES}")
    logger.info(
        f"Target Examples per task (M): {TARGET_EXAMPLES_PER_TASK}, Max Attempts (N): {MAX_GENERATION_ATTEMPTS}")

    # --- Define Paths (Using Config) ---
    sft_data_dir = os.path.join(config.sft_data_dir, f"round_{config.round_number}")
    os.makedirs(sft_data_dir, exist_ok=True)
    solutions_file_path = os.path.join(sft_data_dir, "solutions_training.jsonl")
    solutions_augmented_file_path = os.path.join(sft_data_dir, "solutions_training_augmented.jsonl")
    # Paths below might not be in Config, using constants/defaults
    arc_tasks_base_dir = NET_SCRATCH_TASK_DATA_DIR
    rearc_data_dir = NET_SCRATCH_RE_ARC_DATA

    # --- Scan Task Directory ---
    task_name_to_path, directory_structure = load_task_info(arc_tasks_base_dir)
    training_task_names = directory_structure.get('training', set())
    logger.info(f"Identified {len(training_task_names)} training tasks.")

    # --- Load and Group Cleaned Solutions by Task ---
    solutions_by_task: Dict[str, List[Dict]] = defaultdict(list)
    # ... (loading logic remains the same as previous version) ...
    try:
        with open(solutions_file_path, 'r', encoding='utf-8') as infile:
            for line_num, line in enumerate(infile):
                if not line.strip():
                    continue
                try:
                    data = json.loads(line)
                    task_name = data.get("task_name")
                    if task_name and task_name in task_name_to_path:
                        solutions_by_task[task_name].append(data)
                except json.JSONDecodeError:
                    logger.warning(f"Skipping invalid JSON in cleaned file line {line_num + 1}: {line.strip()}")
    except FileNotFoundError:
        logger.error(f"Cleaned input file not found: {solutions_file_path}")
        sys.exit(1)
    logger.info(f"Loaded solutions for {len(solutions_by_task)} tasks from {solutions_file_path}.")

    # --- Curate Solutions for Each Training Task ---
    curated_augmentable_solutions_per_task: Dict[str, List[Dict]] = {}
    curated_non_augmentable_solutions_per_task: Dict[str, List[Dict]] = {}
    # ... (curation logic remains the same: Q-val -> Length -> Diversity) ...
    for task_name, solutions in solutions_by_task.items():
        logger.debug(f"Curating solutions for task: {task_name} ({len(solutions)} found)")
        solutions_with_q = [(calculate_avg_q(sol), sol) for sol in solutions]
        top_q_solutions = [sol for q, sol in
                           heapq.nlargest(NUM_SOLUTIONS_Q_VALUE, solutions_with_q, key=lambda x: x[0])]

        for sol in top_q_solutions:
            sol['temp_length'] = get_code_length(sol['solution_code'])
        top_q_solutions.sort(key=lambda x: x['temp_length'])
        shortest_solutions = top_q_solutions[:NUM_SOLUTIONS_LENGTH]
        for sol in top_q_solutions:
            del sol['temp_length']

        # Precompute clean_code for diversity selection
        for sol in shortest_solutions:
            sol['clean_code'] = remove_comments(sol['solution_code'])
        diverse_solutions = select_diverse_subset(shortest_solutions, NUM_SOLUTIONS_DIVERSITY)
        # Clean up temporary key
        for sol in shortest_solutions:
            if 'clean_code' in sol:
                del sol['clean_code']

        logger.debug(f"Task {task_name}: Selected {len(diverse_solutions)} diverse solutions.")
        if diverse_solutions:
            if task_name in training_task_names:
                curated_augmentable_solutions_per_task[task_name] = diverse_solutions
            else:
                curated_non_augmentable_solutions_per_task[task_name] = diverse_solutions

    logger.info(f"Finished curation for {len(curated_augmentable_solutions_per_task)} training tasks.")

    # --- Prepare Augmentation Jobs (Task-Based) ---
    augmentation_jobs = []
    for task_name, curated_solutions in curated_augmentable_solutions_per_task.items():
        job = (task_name, curated_solutions, rearc_data_dir, TARGET_EXAMPLES_PER_TASK, MAX_GENERATION_ATTEMPTS)
        augmentation_jobs.append(job)
    logger.info(f"Prepared {len(augmentation_jobs)} task-based augmentation jobs.")

    # --- Clear Output File ---
    try:
        with open(solutions_augmented_file_path, 'w') as _:
            pass
        logger.info(f"Cleared/Created output file: {solutions_augmented_file_path}")
    except IOError as e:
        logger.error(f"Could not open or clear output file {solutions_augmented_file_path}: {e}")
        sys.exit(1)

    total_augmented_entries_saved = 0
    # --- Write Non-Augmentable Solutions to Output File ---
    results_batch_to_write = []
    for task_name, solutions in curated_non_augmentable_solutions_per_task.items():
        for i, solution in enumerate(solutions):
            output_data = {
                "task_name": f"{task_name}_curated_sol_{i}",
                "original_task_name": task_name,
                "solution_code": solution["solution_code"],
                "examples": [],
                "metadata": solution.get("metadata", {})
            }
            results_batch_to_write.append(output_data)

    write_batch_data(solutions_augmented_file_path, results_batch_to_write)
    logger.info(f"Written {len(results_batch_to_write)} non-augmentable entries to {solutions_augmented_file_path}.")
    total_augmented_entries_saved += len(results_batch_to_write)

    # --- Parallel Augmentation ---
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
                                write_batch_data(solutions_augmented_file_path, results_batch_to_write)
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
                write_batch_data(solutions_augmented_file_path, results_batch_to_write)
                logger.info(f"Written final {len(results_batch_to_write)} entries.")

            logger.info(f"Finished parallel processing.")

        except Exception as e:
            logger.error(f"Error during parallel pool execution: {e}", exc_info=True)
    else:
        logger.info("No augmentation jobs to run.")

    logger.info(f"--- Augmentation Summary ---")
    logger.info(f"Total curated solution entries generated & saved: {total_augmented_entries_saved}")
    logger.info(f"Total entries written to {solutions_augmented_file_path}")


if __name__ == "__main__":
    # Instantiate config directly, assuming it reads from environment or defaults
    config_instance = Config()

    main(config_instance)
