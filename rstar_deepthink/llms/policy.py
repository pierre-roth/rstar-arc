import logging
import os.path
from datetime import datetime

from vllm import LLM, SamplingParams, RequestOutput

from constants import STEP_END, CODE_END
from rstar_deepthink.config import Config

logger = logging.getLogger(__name__)


class PolicyModel:
    def __init__(self, config: Config):
        self.config = config
        self.llm = None

    def init(self):
        """Initialize the language model."""

        start = datetime.now()

        self.llm = LLM(
            trust_remote_code=True,
            model=self.config.policy_model if not self.config.fine_tuned else os.path.join(self.config.policy_model_dir,
                                                                                           self.config.policy_model),
            download_dir=self.config.policy_model_dir,
            tensor_parallel_size=self.config.gpus,
            dtype=self.config.dtype,
            max_model_len=self.config.max_model_len,
            enforce_eager=self.config.max_model_len > 32768,
            # max_num_seqs=self.config.max_num_seqs,
            # max_num_batched_tokens=self.config.max_num_batched_tokens,
        )

        end = datetime.now()
        self.config.model_initialization_times["policy"] = end - start

    def generate(self, prompts: list[str], temperature: float) -> list[RequestOutput]:
        """
        Generate completions for a given list of prompts

        Args:
            prompts: the list of prompts to generate completions for
            temperature: the temperature to use for sampling (optional)

        Returns:
            List of RequestOutput objects
        """

        sampling_parameters = SamplingParams(
            temperature=temperature,
            top_p=self.config.top_p,
            repetition_penalty=1.05,
            max_tokens=self.config.max_tokens,
            n=self.config.branching_factor,
            stop=[STEP_END, CODE_END],
            include_stop_str_in_output=True
        )

        request_outputs = self.llm.generate(prompts, sampling_params=sampling_parameters)

        return request_outputs
