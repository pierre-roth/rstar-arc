from vllm import LLM, SamplingParams
from arc_rstar.config import Config


class ProcessPreferenceModel:
    def __init__(self, config: Config):
        self.config = config



