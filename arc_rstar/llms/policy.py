from typing import List, Dict, Any, Optional
from config import Config
from vllm import LLM, SamplingParams


class PolicyModel:
    def __init__(self, config: Config):
        self.config = config
        self.sampling_params = None
        self.llm = None
        self.is_initialized = False

    def init_llm(self):
        """Initialize the language model."""
        if self.is_initialized:
            return
            
        # For beam search, we need to generate multiple completions
        
        # When temperature is 0, this is greedy sampling and n must be 1
        # Otherwise we can use beam_width for n
        if self.config.temperature == 0:
            actual_n = 1
        else:
            actual_n = self.config.beam_width
            
        self.sampling_params = SamplingParams(
            temperature=self.config.temperature,
            top_p=0.95,
            max_tokens=self.config.max_tokens,
            n=actual_n  # Number of candidates to generate
        )
        
        self.llm = LLM(
            model=self.config.policy_model,
            download_dir=self.config.policy_model_dir,
            tensor_parallel_size=self.config.gpus,
            dtype=self.config.dtype
        )
        
        self.is_initialized = True

    def generate(self, prompt, n=None):
        """
        Generate completions for the given prompt.
        
        Args:
            prompt: The prompt to generate completions for
            n: Optional number of completions to generate
            
        Returns:
            List of completions
        """
        if not self.is_initialized:
            self.init_llm()
            
        # Generate completions
        outputs = self.llm.generate(prompt, self.sampling_params)
        
        # Extract and return the completions
        return outputs

