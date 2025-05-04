import os
import random
import sys
from concurrent.futures import TimeoutError

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
GENERATION_ATTEMPTS = 0  # N value
VERIFY_GENERATED_EXAMPLES = True  # Boolean flag for additional verification
BATCH_SIZE = 100
TIMEOUT = 10800


# --- Main Processing Function (for Parallel Execution) ---
def process_task_augmentation(job_data: tuple[str, list[dict], str, int]) -> list[dict]:
    """Processes augmentation for pairs form a single task."""
    task_name, pairs, rearc_data_dir, max_attempts_n = job_data
    logger.debug(
        f"Processing task: {task_name} with {len(pairs)} solutions. N={max_attempts_n}")

    # Flatten all solution dicts from the input pairs (preserve order, allow duplicates)
    all_solutions: set[str] = set()
    for pair in pairs:
        sols = pair.get("solutions", [])
        for sol in sols:
            all_solutions.add(sol["solution_code"])

    assigned_examples: dict[int, list] = defaultdict(list)
    solved_count = 0

    all_solutions_list: list[str] = list(all_solutions)  # Convert to list for indexing

    # 1. Process Existing reARC Examples
    examples = load_rearc_examples(rearc_data_dir, task_name) or []
    if examples:
        logger.debug(f"Task {task_name}: Found {len(examples)} existing reARC examples.")

        for i, example in enumerate(examples):
            for j, solution in enumerate(all_solutions_list):
                if test_solution_on_rearc_example(solution, example):
                    assigned_examples[j].append(i)
                    solved_count += 1
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
            f"Task {task_name}: Current solved: {solved_count}, Max attempts: {max_attempts_n}")
        generation_attempts = 0  # Track attempts separate from existing examples tested

        while generation_attempts < max_attempts_n:
            generation_attempts += 1

            try:
                diff_lb = random.random() * 0.5  # Skew towards lower difficulty
                diff_ub = random.uniform(diff_lb, 1.0)
                new_example = rearc_generator(diff_lb=diff_lb, diff_ub=diff_ub)

                if not new_example or 'input' not in new_example or 'output' not in new_example:
                    logger.warning(
                        f"Task {task_name}: Generator produced invalid structure. Attempt {generation_attempts}")
                    continue

                # Optional: Verify with reARC verifier first
                if VERIFY_GENERATED_EXAMPLES:
                    if not verify_with_rearc_verifier(rearc_verifier, new_example):
                        # logger.debug(f"Task {task_name}: Generated example failed verification.")
                        continue  # Skip example if verification fails

                examples.append(new_example)
                i = len(examples) - 1

                # test against all solutions
                for j, solution in enumerate(all_solutions_list):
                    if test_solution_on_rearc_example(solution, new_example):
                        assigned_examples[j].append(i)
                        solved_count += 1

            except Exception as e:
                logger.error(f"Task {task_name}: Error during generation/testing attempt {generation_attempts}: {e}",
                             exc_info=False)

    logger.debug(f"Task {task_name}: Finished processing examples. Solved count: {solved_count}")

    # 3. Format Output
    output_batch = []
    # for mapping back from solution‐index to actual solution object
    # (all_solutions was created above as list(all_solutions_set))
    for pair_idx, pair in enumerate(pairs):
        # gather all examples solved by any solution in this pair
        # use list of solutions directly (solutions are dicts, not hashable)
        pair_solutions = set()
        for sol in pair.get("solutions", []):
            sol_code = sol["solution_code"]
            pair_solutions.add(sol_code)
        solved_example_idxs: set[int] = set()
        for sol_idx, example_idxs in assigned_examples.items():
            sol = all_solutions_list[sol_idx]
            if sol in pair_solutions:
                solved_example_idxs.update(example_idxs)

        # build the list of example dicts (with 'input' and 'output')
        examples_for_pair = [examples[i] for i in sorted(solved_example_idxs)]

        # emit one entry per original pair
        output_batch.append({
            "task_name": f"{task_name}_pair_{pair_idx}",
            "original_task_name": task_name,
            "prefix": pair["prefix"],
            "chosen": pair["chosen"],
            "rejected": pair["rejected"],
            "examples": examples_for_pair,
            "metadata": pair.get("metadata", {}),
        })

    logger.debug(
        f"Task {task_name}: Finished processing. Generated {len(output_batch)} output entries."
    )
    return output_batch


# --- Main Orchestration ---
def main(config: Config):
    """Main function orchestrating the augmentation process."""
    setup_logging(config.numeric_log_level)
    logger.info("Starting reward data augmentation process...")
    logger.info(f"VERIFY_GENERATED_EXAMPLES set to: {VERIFY_GENERATED_EXAMPLES}")
    logger.info(f"Max Attempts (N): {GENERATION_ATTEMPTS}")

    # --- Define Paths (Using Config) ---
    sft_data_dir = os.path.join(config.sft_data_dir, f"round_{config.round_number}")
    os.makedirs(sft_data_dir, exist_ok=True)
    preference_pairs_file_path = os.path.join(sft_data_dir, "preference_pairs_training.jsonl")
    preference_pairs_augmented_file_path = os.path.join(sft_data_dir, "preference_pairs_training_augmented.jsonl")
    # Paths below might not be in Config, using constants/defaults
    arc_tasks_base_dir = NET_SCRATCH_TASK_DATA_DIR
    rearc_data_dir = NET_SCRATCH_RE_ARC_DATA

    # --- Scan Task Directory ---
    task_name_to_path, directory_structure = load_task_info(arc_tasks_base_dir)
    training_task_names = directory_structure.get('training', set())
    logger.info(f"Identified {len(training_task_names)} training tasks.")

    # --- Load and Group Cleaned Solutions by Task ---
    pairs_per_task: dict[str, list[dict]] = defaultdict(list)
    try:
        with open(preference_pairs_file_path, 'r', encoding='utf-8') as infile:
            for line_num, line in enumerate(infile):
                if not line.strip():
                    continue
                try:
                    data = json.loads(line)
                    task_name = data.get("task_name")
                    if task_name and task_name in task_name_to_path:
                        pairs_per_task[task_name].append(data)
                except json.JSONDecodeError:
                    logger.warning(f"Skipping invalid JSON in cleaned file line {line_num + 1}: {line.strip()}")
    except FileNotFoundError:
        logger.error(f"Input file not found: {preference_pairs_file_path}")
        sys.exit(1)
    logger.info(f"Loaded solutions for {len(pairs_per_task)} tasks from {preference_pairs_file_path}.")

    # --- Curate Solutions for Each Training Task ---
    augmentable_pairs_per_task: dict[str, list[dict]] = {}
    non_augmentable_pairs_per_task: dict[str, list[dict]] = {}

    for task_name, solutions in pairs_per_task.items():
        if task_name in training_task_names:
            augmentable_pairs_per_task[task_name] = solutions
        else:
            non_augmentable_pairs_per_task[task_name] = solutions

    logger.info(f"Found pairs for {len(augmentable_pairs_per_task)} training tasks.")

    # --- Prepare Augmentation Jobs (Task-Based) ---
    augmentation_jobs = []
    for task_name, pairs in augmentable_pairs_per_task.items():
        job = (task_name, pairs, rearc_data_dir, GENERATION_ATTEMPTS)
        augmentation_jobs.append(job)
    logger.info(f"Prepared {len(augmentation_jobs)} task-based augmentation jobs.")

    # --- Clear Output File ---
    try:
        with open(preference_pairs_augmented_file_path, 'w') as _:
            pass
        logger.info(f"Cleared/Created output file: {preference_pairs_augmented_file_path}")
    except IOError as e:
        logger.error(f"Could not open or clear output file {preference_pairs_augmented_file_path}: {e}")
        sys.exit(1)

    total_augmented_entries_saved = 0
    # --- Write Non-Augmentable Solutions to Output File ---
    results_batch_to_write = []
    for task_name, pairs in non_augmentable_pairs_per_task.items():
        for i, pair in enumerate(pairs):
            prefix_code = pair["prefix"]
            chosen = pair["chosen"]
            rejected = pair["rejected"]
            output_data = {
                "task_name": f"{task_name}_pair_{i}",
                "original_task_name": task_name,
                "prefix": prefix_code,  # Prompt + code up to the split point
                "chosen": chosen,  # The chosen step's code
                "rejected": rejected,  # The rejected step's code
                "examples": [],  # No examples for non-augmentable solutions
                "metadata": pair.get("metadata", {})
            }
            results_batch_to_write.append(output_data)

    write_batch_data(preference_pairs_augmented_file_path, results_batch_to_write)
    logger.info(
        f"Written {len(results_batch_to_write)} non-augmentable entries to {preference_pairs_augmented_file_path}.")
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
                                write_batch_data(preference_pairs_augmented_file_path, results_batch_to_write)
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
                write_batch_data(preference_pairs_augmented_file_path, results_batch_to_write)
                logger.info(f"Written final {len(results_batch_to_write)} entries.")

            logger.info(f"Finished parallel processing.")

        except Exception as e:
            logger.error(f"Error during parallel pool execution: {e}", exc_info=True)
    else:
        logger.info("No augmentation jobs to run.")

    logger.info(f"--- Augmentation Summary ---")
    logger.info(f"Total pairs generated & saved: {total_augmented_entries_saved}")


if __name__ == "__main__":
    # Instantiate config directly, assuming it reads from environment or defaults
    config_instance = Config()

    main(config_instance)
