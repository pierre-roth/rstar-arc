from config import Config
from vllm import LLM, SamplingParams


class ProcessPreferenceModel:
    def __init__(self, config: Config):
        self.config = config
        self.llm = LLM(
            model=config.pp_model,
            download_dir=config.pp_model_dir,
            tensor_parallel_size=config.gpus,
            dtype=config.dtype
        )

    def generate(self, prompt):
        sampling_params = SamplingParams(temperature=self.config.temperature)
        return self.llm.generate(prompt, sampling_params)

