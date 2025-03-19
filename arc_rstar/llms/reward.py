import logging
from random import random

from vllm import LLM

from arc_rstar.agents.node import Node
from config import Config


class RewardModel:
    def __init__(self, config: Config, terminal_guided=False):
        self.config = config
        self.llm = None
        self.tg = terminal_guided
        self.is_initialized = False

    def init_llm(self):
        if self.is_initialized or self.tg:
            return

        self.llm = LLM(
            model=self.config.reward_model,
            download_dir=self.config.reward_model_dir,
            tensor_parallel_size=self.config.gpus,
            dtype=self.config.dtype
        )

        self.is_initialized = True

    # the score function is used to evaluate the quality of a node
    # the value is a float between -1 and 1
    def score(self, node: Node):
        if self.tg:
            # In terminal-guided mode, just return random score
            score = random() * 2 - 1

            logging.debug(f"PPM (terminal-guided): generated score for node {node.tag}: {score:.4f}")

            return score

        raise NotImplementedError

        # sampling_params = SamplingParams(temperature=self.config.temperature)
        # return self.llm.generate(node, sampling_params=sampling_params)

