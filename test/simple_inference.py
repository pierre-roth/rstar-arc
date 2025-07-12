import json
import logging
import math
import os
import sys
from concurrent.futures import TimeoutError
from functools import partial
from typing import List

from pebble import ProcessPool
from vllm import LLM, SamplingParams

# ---------------------------------------------------------------------------
# Project setup
# ---------------------------------------------------------------------------
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from constants import (
    SFT_SYSTEM_PROMPT,
    SFT_IN_BETWEEN_PROMPT,
    CODE,
    CODE_END,
    CODE_PREFIX,
    WALL_TIMEOUT_SECONDS,
)
from utils import setup_logging
from rstar_deepthink import Config
from rstar_deepthink.arc_task.task_utils import load_tasks, task_to_prompt
from rstar_deepthink.tools import verify_prefixes_and_code

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _extract_code(text: str) -> str:
    """Extract the code segment between CODE and CODE_END."""
    start = text.find(CODE)
    end = text.find(CODE_END)
    if start == -1 or end == -1:
        return ""
    return text[start: end + len(CODE_END)].strip()


def _prepare_io(task) -> tuple[List[List[List[int]]], List[List[List[int]]]]:
    """Return input and output grids for all examples of a task."""
    inputs = [e.input_grid.grid for e in task.training_examples + task.test_examples]
    outputs = [e.output_grid.grid for e in task.training_examples + task.test_examples]
    return inputs, outputs


def _verify_code_worker(
        code: str, inputs: List[List[List[int]]], outputs: List[List[List[int]]]
) -> bool:
    """
    Wrapper for verify_prefixes_and_code to be used with a process pool.
    Returns True if the code is correct, False otherwise.
    """

    success, _, err_full, passed_full, _ = verify_prefixes_and_code(
        code, inputs, outputs
    )
    return success and not err_full and passed_full


def _compute_pass_at_m(c: int, n: int, m: int) -> float:
    """Expected pass@M for a single task given c successes out of n samples."""

    if m > n or m <= 0:
        return 0.0
    if c == 0:
        return 0.0
    return 1.0 - math.comb(n - c, m) / math.comb(n, m)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    config = Config()
    config.sort_by_length = True  # Ensure tasks are sorted by length for batch skipping
    setup_logging(config.numeric_log_level)

    tasks = load_tasks(config)

    logger.info("Initializing policy model via vLLM…")
    model_name = (
        os.path.join(config.policy_model_dir, config.policy_model)
        if config.fine_tuned
        else config.policy_model
    )

    llm = LLM(
        model=model_name,
        trust_remote_code=True,
        download_dir=config.policy_model_dir,
        dtype=config.dtype,
        max_model_len=config.max_seq_len,
        gpu_memory_utilization=0.8,
    )

    tokenizer = llm.get_tokenizer()
    stop_ids = [tokenizer.convert_tokens_to_ids(CODE_END)]

    if config.length_pre_filtering:
        original_task_count = len(tasks)
        filtered_tasks = []
        for task in tasks:
            prompt = (
                    SFT_SYSTEM_PROMPT
                    + task_to_prompt(task)
                    + SFT_IN_BETWEEN_PROMPT
                    + CODE_PREFIX
            )
            prompt_len = len(tokenizer.encode(prompt))
            total_len = prompt_len + config.max_tokens
            if total_len <= config.max_seq_len:
                filtered_tasks.append(task)
            else:
                logger.info(
                    f"Skipping task {task.name} as prompt length ({prompt_len}) + max_tokens ({config.max_tokens}) = {total_len} exceeds context {config.max_seq_len}"
                )

        if original_task_count > 0 and not filtered_tasks:
            logger.error("No tasks left after filtering by length. Exiting.")
            sys.exit(1)

        logger.info(
            f"Filtered tasks: {len(filtered_tasks)}/{original_task_count} remaining."
        )
        tasks = filtered_tasks

    # -------------------------------------------------------------------
    # Generation / verification parameters
    # -------------------------------------------------------------------
    n = config.num_rollouts  # assumed to be a power of two
    batch_size = config.batch_size if config.batch_size > 0 else len(tasks)

    # All powers of two < n: 1, 2, 4, …
    m_values: List[int] = [1]
    while m_values[-1] * 2 < n:
        m_values.append(m_values[-1] * 2)

    sampling_params = SamplingParams(
        temperature=config.policy_temperature,
        top_p=config.top_p,
        repetition_penalty=config.repetition_penalty,
        max_tokens=config.max_tokens,
        n=n,
        stop_token_ids=stop_ids,
        include_stop_str_in_output=True,
        skip_special_tokens=False,
    )

    # -------------------------------------------------------------------
    # Output paths
    # -------------------------------------------------------------------
    output_dir = os.path.join(config.sft_data_dir, f"round_{config.round_number}")
    os.makedirs(output_dir, exist_ok=True)
    solutions_file_path = os.path.join(
        output_dir,
        "solutions_training.jsonl" if not config.evaluation else "solutions_evaluation.jsonl",
    )

    os.makedirs(config.local_job_dir, exist_ok=True)

    stats_file_path = os.path.join(config.local_job_dir, "pass_stats.json")

    batch_stats_file_path = os.path.join(
        config.local_job_dir, "pass_stats_batches.jsonl"
    )

    # -------------------------------------------------------------------
    # Statistics containers
    # -------------------------------------------------------------------
    task_stats: List[dict] = []
    overall_pass_flags: List[bool] = []

    workers = max(1, config.cpus - 1)
    with ProcessPool(max_workers=workers) as pool:
        for i, batch_start in enumerate(range(0, len(tasks), batch_size)):
            if i < config.skip_batches:
                logger.info(f"Skipping batch index {i}")
                continue

            if config.num_batches > 0:
                config.num_batches -= 1
            elif config.num_batches == 0:
                logger.info("Reached the limit of batches to process. Stopping.")
                break

            batch_tasks = tasks[batch_start:batch_start + batch_size]

            # Build prompts on the fly to save memory
            batch_prompts = [
                SFT_SYSTEM_PROMPT
                + task_to_prompt(t)
                + SFT_IN_BETWEEN_PROMPT
                + CODE_PREFIX
                for t in batch_tasks
            ]

            logger.info(
                f"Running inference on batch {batch_start // batch_size + 1} with {len(batch_tasks)} tasks…"
            )

            request_outputs = llm.generate(batch_prompts, sampling_params)

            batch_task_stats: List[dict] = []
            batch_pass_flags: List[bool] = []

            for task, output in zip(batch_tasks, request_outputs):
                generations = [o.text for o in output.outputs]
                texts = [CODE_PREFIX + gen for gen in generations]
                codes = [_extract_code(text) for text in texts]
                inputs, outputs_ = _prepare_io(task)

                check = partial(_verify_code_worker, inputs=inputs, outputs=outputs_)
                future = pool.map(
                    check,
                    codes,
                    timeout=(n // workers + 1) * WALL_TIMEOUT_SECONDS,
                )

                successes = 0
                results = []
                try:
                    results = list(future.result())
                    successes = sum(1 for r in results if r)
                except TimeoutError:
                    logger.warning(f"Verification for task {task.name} timed out.")
                except Exception as e:
                    logger.error(
                        f"An error occurred during verification for task {task.name}: {e}"
                    )

                passed = successes > 0
                overall_pass_flags.append(passed)
                batch_pass_flags.append(passed)
                logger.info(
                    f"Task {task.name}: {successes}/{n} correct → pass@{n}={passed}"
                )

                # Persist individual correct solutions (unchanged logic)
                if passed:
                    with open(solutions_file_path, "a", encoding="utf-8") as f:
                        for ok, code in zip(results, codes):
                            if ok:
                                json.dump({"task_name": task.name, "solution_code": code}, f)
                                f.write("\n")

                # Compute expected pass@M values for this task
                pass_at_m = {m: _compute_pass_at_m(successes, n, m) for m in m_values}
                task_stats.append(
                    {
                        "task_name": task.name,
                        "successes": successes,
                        "pass_at_m": pass_at_m,
                    }
                )

                batch_task_stats.append(
                    {
                        "task_name": task.name,
                        "successes": successes,
                        "pass_at_m": pass_at_m,
                    }
                )

            batch_overall_pass_at_n = (
                    sum(batch_pass_flags) / len(batch_pass_flags)
            )
            batch_overall_pass_at_m = {
                m: sum(t["pass_at_m"][m] for t in batch_task_stats) / len(batch_task_stats)
                for m in m_values
            }

            batch_stats_payload = {
                "batch_index": i,  # zero-based
                "num_tasks": len(batch_task_stats),
                "overall_pass_at_m": batch_overall_pass_at_m,
                "overall_pass_at_n": batch_overall_pass_at_n,
                "tasks": batch_task_stats,
            }

            # append **one line** to the JSONL file
            with open(batch_stats_file_path, "a", encoding="utf-8") as f:
                json.dump(batch_stats_payload, f)
                f.write("\n")

    # -------------------------------------------------------------------
    # Aggregate statistics
    # -------------------------------------------------------------------
    overall_pass_at_n = sum(overall_pass_flags) / len(overall_pass_flags)
    overall_pass_at_m = {
        m: sum(t["pass_at_m"][m] for t in task_stats) / len(task_stats) for m in m_values
    }

    logger.info("Overall pass rates:")
    for m in m_values:
        logger.info(f"  expected pass@{m}: {overall_pass_at_m[m]:.4f}")
    logger.info(f"  empirical pass@{n}: {overall_pass_at_n:.4f}")

    # -------------------------------------------------------------------
    # Save statistics for later visualisation
    # -------------------------------------------------------------------
    stats_payload = {
        "n": n,
        "m_values": m_values,
        "num_tasks": len(task_stats),
        "overall_pass_at_m": overall_pass_at_m,
        "overall_pass_at_n": overall_pass_at_n,
        "tasks": task_stats,
    }

    with open(stats_file_path, "w", encoding="utf-8") as f:
        json.dump(stats_payload, f, indent=2)

    logger.info(f"Wrote detailed statistics to {stats_file_path}")


if __name__ == "__main__":
    main()
