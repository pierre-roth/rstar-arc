import logging
import os
import sys
from typing import List
from functools import partial
from concurrent.futures import TimeoutError
from pebble import ProcessPool

from vllm import LLM, SamplingParams

# ---------------------------------------------------------------------------
# Project setup
# ---------------------------------------------------------------------------
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from constants import SFT_SYSTEM_PROMPT, SFT_IN_BETWEEN_PROMPT, CODE, CODE_END, CODE_PREFIX, WALL_TIMEOUT_SECONDS
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


def _verify_code_worker(code: str, inputs: List[List[List[int]]], outputs: List[List[List[int]]]) -> bool:
    """
    Wrapper for verify_prefixes_and_code to be used with a process pool.
    Returns True if the code is correct, False otherwise.
    """
    success, _, err_full, passed_full, _ = verify_prefixes_and_code(
        code, inputs, outputs
    )
    return success and not err_full and passed_full


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    config = Config()
    setup_logging(config.numeric_log_level)

    tasks = load_tasks(config)

    logger.info("Initializing policy model via vLLM...")
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
        gpu_memory_utilization=config.policy_vram_percentage,
    )

    tokenizer = llm.get_tokenizer()
    stop_ids = [
        tokenizer.convert_tokens_to_ids(CODE_END)
    ]

    if config.length_pre_filtering:
        original_task_count = len(tasks)
        filtered_tasks = []
        for task in tasks:
            prompt = SFT_SYSTEM_PROMPT + task_to_prompt(task) + SFT_IN_BETWEEN_PROMPT + CODE_PREFIX
            prompt_len = len(tokenizer.encode(prompt))
            total_len = prompt_len + config.max_tokens
            if total_len <= config.max_seq_len:
                filtered_tasks.append(task)
            else:
                logger.info(
                    f"Skipping task {task.name} as prompt length ({prompt_len}) + max_tokens ({config.max_tokens}) = {total_len} would exceed context size {config.max_seq_len}"
                )

        if original_task_count > 0 and not filtered_tasks:
            logger.error("No tasks left after filtering by length. Exiting.")
            sys.exit(1)

        logger.info(f"Filtered tasks: {len(filtered_tasks)}/{original_task_count} remaining.")
        tasks = filtered_tasks

    prompts = [
        SFT_SYSTEM_PROMPT + task_to_prompt(task) + SFT_IN_BETWEEN_PROMPT + CODE_PREFIX
        for task in tasks
    ]

    n = config.num_rollouts
    sampling_params = SamplingParams(
        temperature=config.policy_temperature,
        top_p=config.top_p,
        repetition_penalty=config.repetition_penalty,
        max_tokens=config.max_tokens,
        n=n,
        stop_token_ids=stop_ids,
        include_stop_str_in_output=True,
        skip_special_tokens=False
    )

    logger.info("Starting inference...")
    request_outputs = llm.generate(prompts, sampling_params)

    workers = max(1, config.cpus - 1)

    overall_pass = []
    with ProcessPool(max_workers=workers) as pool:
        for task, output in zip(tasks, request_outputs):
            partial_prompt = CODE_PREFIX
            generations = [o.text for o in output.outputs]
            texts = [partial_prompt + generation for generation in generations]
            codes = [_extract_code(text) for text in texts]
            inputs, outputs_ = _prepare_io(task)

            check_code_with_context = partial(_verify_code_worker, inputs=inputs, outputs=outputs_)
            future = pool.map(check_code_with_context, codes, timeout=(n // workers + 1) * WALL_TIMEOUT_SECONDS)

            successes = 0
            try:
                iterator = future.result()
                successes = sum(1 for r in iterator if r)
            except TimeoutError:
                logger.warning(
                    f"Code verification for task {task.name} timed out."
                )
            except Exception as e:
                logger.error(f"An error occurred during verification for task {task.name}: {e}")

            passed = successes > 0
            overall_pass.append(passed)
            logger.info(
                f"Task {task.name}: pass@{n} {passed} ({successes}/{n} correct)"
            )

    overall_rate = sum(overall_pass) / len(overall_pass)
    logger.info(f"Overall pass@{n}: {overall_rate:.2f}")


if __name__ == "__main__":
    main()
