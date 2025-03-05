from vllm import LLM, SamplingParams
from arc_rstar.config import Config


class PolicyModel:
    def __init__(self, config: Config):
        self.config = config



