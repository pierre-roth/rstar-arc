import logging
from vllm import LLM, SamplingParams, RequestOutput

from config import Config, STEP_END, CODE_END

logger = logging.getLogger(__name__)


class RewardModel:
    def __init__(self, config: Config):
        self.config = config

    def init(self):
        """Initialize the language model."""

        logger.info("Initializing reward model ...")

        logger.info("Reward model initialized.")

    def score(self, prompts: list[str]) -> list[float]:
        """
        Generate completions for a given list of prompts

        Args:
            prompts: the list of prompts to generate completions for

        Returns:
            List of floats
        """

        # TODO: add if value_func block if reward model code is implemented

        return [0.0] * len(prompts)
