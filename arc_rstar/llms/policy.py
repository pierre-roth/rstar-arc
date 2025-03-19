import logging

from vllm import LLM, SamplingParams
import os
import random
import time
from vllm.utils import get_open_port  # Using vLLM's built-in utility

from config import Config, STEP_END, CODE_END

logger = logging.getLogger(__name__)


class PolicyModel:
    def __init__(self, config: Config):
        self.config = config
        self.llm = None
        self.is_initialized = False
        self.engine_id = f"vllm_engine_{config.job_id}_{int(time.time())}"

    def init_llm(self):
        """Initialize the language model."""
        if self.is_initialized:
            return

        logger.info("Initializing policy model ...")

        self.llm = LLM(
            model=self.config.policy_model,
            download_dir=self.config.policy_model_dir,
            tensor_parallel_size=self.config.gpus,
            dtype=self.config.dtype,
            max_model_len=self.config.max_model_len,
        )

        self.is_initialized = True

        logger.info("Policy model initialized.")

    def generate(self, prompt: str) -> list[str]:
        """
        Generate completions for the given prompt.

        Args:
            prompt: The prompt to generate completions for

        Returns:
            List of completions
        """
        if not self.is_initialized:
            self.init_llm()

        # Generate completions with default params
        sampling_params = SamplingParams(
            temperature=self.config.policy_temperature,
            top_p=self.config.top_p,
            max_tokens=self.config.max_tokens,
            n=self.config.branching_factor,  # Number of candidates to generate
            stop=[STEP_END, CODE_END],
            include_stop_str_in_output=True
        )

        request_outputs = self.llm.generate([prompt], sampling_params=sampling_params)

        request_output = request_outputs[0]
        completion_outputs = request_output.outputs

        outputs = [completion_output.text for completion_output in completion_outputs]

        # Return the completions
        return outputs

