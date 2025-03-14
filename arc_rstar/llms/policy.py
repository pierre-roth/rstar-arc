from typing import Any, Optional
from config import Config
from vllm import LLM, SamplingParams


class PolicyModel:
    def __init__(self, config: Config):
        self.config = config
        self.llm = None
        self.is_initialized = False

    def init_llm(self):
        """Initialize the language model."""
        if self.is_initialized:
            return

        self.llm = LLM(
            model=self.config.policy_model,
            download_dir=self.config.policy_model_dir,
            tensor_parallel_size=self.config.gpus,
            dtype=self.config.dtype
        )

        self.is_initialized = True

    def generate(self, prompt: str) -> list[str]:
        """
        Generate completions for the given prompt.
        
        Args:
            prompt: The prompt to generate completions for
            
        Returns:
            List of completions
        """
        if not self.is_initialized:
            self.init_llm()

        # Generate completions with default params

        sampling_params = SamplingParams(
            temperature=self.config.temperature,
            top_p=0.95,
            max_tokens=self.config.max_tokens,
            n=self.config.branching_factor  # Number of candidates to generate
        )

        request_outputs = self.llm.generate([prompt], sampling_params=sampling_params)

        request_output = request_outputs[0]
        completion_outputs = request_output.outputs

        outputs = [completion_output.text for completion_output in completion_outputs]

        # Log the completions if verbose
        if self.config.verbose:
            print(f"\nGenerated {len(outputs)} completions from policy model:")
            for i, output in enumerate(outputs):
                # Truncate long outputs for logging
                preview = output[:100] + "..." if len(output) > 100 else output
                preview = preview.replace('\n', ' ')
                print(f"Completion {i+1}: {preview}")
        
        # Return the completions
        return outputs
