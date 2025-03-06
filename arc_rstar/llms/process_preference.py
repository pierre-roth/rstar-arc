from config import Config
from vllm import LLM, SamplingParams


class ProcessPreferenceModel:
    def __init__(self, config: Config, terminal_guided=False):
        self.config = config
        self.llm = None
        self.tg = terminal_guided

    def init_llm(self):
        if not self.tg:
            self.llm = LLM(
                model=self.config.pp_model,
                download_dir=self.config.pp_model_dir,
                tensor_parallel_size=self.config.gpus,
                dtype=self.config.dtype
            )
        else:
            pass

    def score(self, node):
        sampling_params = SamplingParams(temperature=self.config.temperature)
        return self.llm.generate(node, sampling_params)


