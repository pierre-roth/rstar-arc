import logging
import os.path
from datetime import datetime

"""Removed static import of vllm. Import LLM and SamplingParams locally when needed."""

from constants import STEP_END, CODE_END
from rstar_deepthink.config import Config

logger = logging.getLogger(__name__)


class PolicyModel:
    def __init__(self, config: Config):
        self.config = config
        self.llm = None
        self.stop_token_ids = []

    def init(self):
        """Initialize the language model."""

        start = datetime.now()
        # Import heavy vllm LLM class only when initializing model
        from vllm import LLM

        model = self.config.policy_model if not self.config.fine_tuned else os.path.join(self.config.policy_model_dir,
                                                                                         self.config.policy_model)

        self.llm = LLM(
            trust_remote_code=True,
            model=model,
            download_dir=self.config.policy_model_dir,
            tensor_parallel_size=self.config.gpus,
            dtype=self.config.dtype,
            max_model_len=self.config.max_seq_len,
            enforce_eager=self.config.enforce_eager,
            # max_num_seqs=self.config.max_num_seqs,
            # max_num_batched_tokens=self.config.max_num_batched_tokens,
            gpu_memory_utilization=self.config.policy_vram_percentage,
        )

        tokenizer = self.llm.get_tokenizer()
        self.stop_token_ids = [tokenizer.convert_tokens_to_ids(STEP_END), tokenizer.convert_tokens_to_ids(CODE_END)]

        end = datetime.now()
        self.config.model_initialization_times["policy"] = end - start

    def generate(self, prompts: list[str], temperature: float):
        """
        Generate completions for a given list of prompts

        Args:
            prompts: the list of prompts to generate completions for
            temperature: the temperature to use for sampling (optional)

        Returns:
            List of RequestOutput objects
        """

        # Import heavy vllm SamplingParams only when generating
        from vllm import SamplingParams
        sampling_parameters = SamplingParams(
            temperature=temperature,
            top_p=self.config.top_p,
            repetition_penalty=self.config.repetition_penalty,
            max_tokens=self.config.max_tokens,
            n=self.config.branching_factor,
            stop_token_ids=self.stop_token_ids,
            include_stop_str_in_output=True,
            skip_special_tokens=False
        )

        request_outputs = self.llm.generate(prompts, sampling_params=sampling_parameters)

        return request_outputs
