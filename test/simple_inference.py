import logging
import os
import sys
from typing import List

from vllm import LLM, SamplingParams

# ---------------------------------------------------------------------------
# Project setup
# ---------------------------------------------------------------------------
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from constants import SFT_SYSTEM_PROMPT, SFT_IN_BETWEEN_PROMPT
from utils import setup_logging
from rstar_deepthink import Config
from rstar_deepthink.arc_task.task_utils import load_tasks, task_to_prompt
from rstar_deepthink.tools import verify_prefixes_and_code

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _extract_code(text: str) -> str:
    """Extract the code segment between <beginning_of_code> and <end_of_code>."""
    start = text.find("<beginning_of_code>")
    end = text.find("<end_of_code>")
    if start == -1 or end == -1:
        return ""
    return text[start: end + len("<end_of_code>")].strip()


def _prepare_io(task) -> tuple[List[List[List[int]]], List[List[List[int]]]]:
    """Return input and output grids for all examples of a task."""
    inputs = [e.input_grid.grid for e in task.training_examples + task.test_examples]
    outputs = [e.output_grid.grid for e in task.training_examples + task.test_examples]
    return inputs, outputs


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    config = Config()
    setup_logging(config.numeric_log_level)

    tasks = load_tasks(config)
    prompts = [
        SFT_SYSTEM_PROMPT + task_to_prompt(task) + SFT_IN_BETWEEN_PROMPT
        for task in tasks
    ]

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
        tokenizer.convert_tokens_to_ids("<end_of_code>")
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
    )

    logger.info("Starting inference...")
    request_outputs = llm.generate(prompts, sampling_params)

    overall_pass = []
    for task, output in zip(tasks, request_outputs):
        texts = [o.text for o in output.outputs]
        codes = [_extract_code(text) for text in texts]
        logger.info(f"Task {task.name}")
        logger.info(f"Task prompt: {task_to_prompt(task)}")
        logger.info(f"Generations: {texts}")
        inputs, outputs_ = _prepare_io(task)
        successes = 0
        for code in codes:
            success, _, err_full, passed_full, _ = verify_prefixes_and_code(
                code, inputs, outputs_
            )
            if success and not err_full and passed_full:
                successes += 1
        passed = successes > 0
        overall_pass.append(passed)
        logger.info(
            f"Task {task.name}: pass@{n} {passed} ({successes}/{n} correct)"
        )

    overall_rate = sum(overall_pass) / len(overall_pass)
    logger.info(f"Overall pass@{n}: {overall_rate:.2f}")


if __name__ == "__main__":
    main()
