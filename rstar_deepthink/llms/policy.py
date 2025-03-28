import logging
from datetime import datetime

from vllm import LLM, SamplingParams, RequestOutput

from rstar_deepthink.config import Config, STEP_END, CODE_END

logger = logging.getLogger(__name__)


class PolicyModel:
    def __init__(self, config: Config):
        self.config = config
        self.llm = None
        self.sampling_params = None

    def init(self):
        """Initialize the language model."""

        start = datetime.now()

        self.llm = LLM(
            model=self.config.policy_model,
            download_dir=self.config.policy_model_dir,
            tensor_parallel_size=self.config.gpus,
            dtype=self.config.dtype,
            max_model_len=self.config.max_model_len,
        )

        self.sampling_params = SamplingParams(
            temperature=self.config.policy_temperature,
            top_p=self.config.top_p,
            max_tokens=self.config.max_tokens,
            n=self.config.branching_factor,
            stop=[STEP_END, CODE_END],
            include_stop_str_in_output=True
        )

        end = datetime.now()
        self.config.model_initialization_times["policy"] = end - start

    def generate(self, prompts: list[str]) -> list[RequestOutput]:
        """
        Generate completions for a given list of prompts

        Args:
            prompts: the list of prompts to generate completions for

        Returns:
            List of RequestOutput objects
        """

        request_outputs = self.llm.generate(prompts, sampling_params=self.sampling_params)

        return request_outputs
