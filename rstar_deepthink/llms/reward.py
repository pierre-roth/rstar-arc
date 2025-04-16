import logging
from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional, Union, Tuple

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PretrainedConfig
from transformers.modeling_outputs import ModelOutput

from rstar_deepthink.config import Config

logger = logging.getLogger(__name__)


# Define a custom output class for the reward model to potentially include hidden states if needed later
@dataclass
class RewardModelOutput(ModelOutput):
    """
    Output class for the RewardModelWithValueHead.
    """
    logits: torch.FloatTensor = None  # The scalar reward score(s)
    # Add other outputs like hidden_states if needed for other purposes


class ValueHead(nn.Module):
    """
    Value Head for the Reward Model.
    Takes hidden states from the base model and outputs a scalar reward value.
    """

    def __init__(self, config: PretrainedConfig, **kwargs):
        super().__init__()
        # Use dropout probability from config or default
        dropout_prob = getattr(config, "summary_dropout_prob", kwargs.pop("summary_dropout_prob", 0.1))
        self.dropout = nn.Dropout(dropout_prob) if dropout_prob > 0 else nn.Identity()

        # Determine hidden size - prioritize config attribute, fallback to common sizes or default
        hidden_size = getattr(config, "hidden_size",
                              getattr(config, "word_embed_proj_dim", 4096))  # Example fallback logic

        # Linear layer to project hidden state to a single scalar value
        self.summary = nn.Linear(hidden_size, 1)

        # Initialize weights (optional, but can sometimes help stability)
        # nn.init.xavier_uniform_(self.summary.weight) # Example initialization
        # nn.init.zeros_(self.summary.bias)
        # rStar-Math paper used specific small init values:
        nn.init.normal_(self.summary.weight, mean=5e-7, std=1e-6)
        nn.init.constant_(self.summary.bias, 1e-6)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the ValueHead.

        Args:
            hidden_states: Hidden states from the base language model.

        Returns:
            A tensor containing the scalar reward value(s).
        """
        hidden_states = self.dropout(hidden_states)

        # Ensure correct dtype for the linear layer
        if hasattr(self.summary, "weight") and hidden_states.dtype != self.summary.weight.dtype:
            output = hidden_states.to(self.summary.weight.dtype)
        else:
            output = hidden_states

        # Project to scalar value
        output = self.summary(output)

        # Optional: Apply activation like tanh as mentioned in rStar-Math paper
        # output = torch.tanh(output)

        return output


class RewardModelWithValueHead(PreTrainedModel):
    """
    A wrapper model that combines a pre-trained transformer base
    with a ValueHead to predict reward scores.
    Inherits from PreTrainedModel for easier saving/loading.
    """
    # Add the config_class for compatibility with from_pretrained etc.
    config_class = PretrainedConfig  # Use base config class or specific one if needed

    def __init__(self, pretrained_model: PreTrainedModel):
        # Initialize using the base model's config
        super().__init__(pretrained_model.config)
        self.pretrained_model = pretrained_model
        self.v_head = ValueHead(self.config)

        # Copy gradient checkpointing attributes if they exist
        if hasattr(self.pretrained_model, "gradient_checkpointing_disable"):
            self.gradient_checkpointing_disable = self.pretrained_model.gradient_checkpointing_disable
        if hasattr(self.pretrained_model, "gradient_checkpointing_enable"):
            self.gradient_checkpointing_enable = self.pretrained_model.gradient_checkpointing_enable

    def forward(
            self,
            input_ids: Optional[torch.LongTensor] = None,
            attention_mask: Optional[torch.FloatTensor] = None,
            return_dict: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            use_cache: Optional[bool] = None,  # Added use_cache
            past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,  # Added past_key_values
            **kwargs,  # Allow other base model args
    ) -> Union[torch.Tensor, RewardModelOutput]:
        """
        Forward pass of the combined reward model.

        Args:
            input_ids: Input token IDs.
            attention_mask: Attention mask.
            return_dict: Whether to return a ModelOutput object.
            output_attentions: Whether to output attention weights.
            output_hidden_states: Whether to output hidden states.
            use_cache: Whether to use cached outputs.
            past_key_values: Cached key/value pairs for faster decoding.
            kwargs: Additional arguments passed to the base model.


        Returns:
            Scalar reward score tensor or RewardModelOutput object.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        output_hidden_states = True  # Always require hidden states for the value head

        # Pass inputs to the base transformer model
        outputs = self.pretrained_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,  # Force return_dict from base model
            use_cache=use_cache,
            past_key_values=past_key_values,
            **kwargs,
        )

        # Get the last hidden state
        # Indexing might vary slightly based on model architecture, [-1] is common
        last_hidden_state = outputs.hidden_states[-1]

        # Pass the last hidden state through the value head
        # Shape: (batch_size, sequence_length, hidden_size) -> (batch_size, sequence_length, 1)
        values = self.v_head(last_hidden_state)

        # --- Extract Score for the Last Token ---
        # Find the index of the last non-padding token for each sequence
        if attention_mask is None:
            # If no mask, assume all tokens are valid (might be incorrect for padding)
            sequence_lengths = -1  # Take the last token
        else:
            # Find the last token that is not padding (mask == 1)
            sequence_lengths = torch.sum(attention_mask, dim=1) - 1

        # Gather the value corresponding to the last token of each sequence
        # values shape: (batch_size, sequence_length, 1)
        # sequence_lengths shape: (batch_size,) -> need (batch_size, 1, 1) for gather
        # Un-squeeze twice to match the dimensions for gather
        last_token_indices = sequence_lengths.unsqueeze(-1).unsqueeze(-1)
        # Expand the index tensor to match the last dimension of the values tensor (which is 1)
        last_token_indices = last_token_indices.expand(-1, -1, values.shape[-1])

        # Gather the scores
        # Shape: (batch_size, 1, 1)
        final_scores = torch.gather(values, dim=1, index=last_token_indices)

        # Squeeze the result to get shape (batch_size,)
        final_scores = final_scores.squeeze(-1).squeeze(-1)

        if not return_dict:
            return final_scores
        else:
            return RewardModelOutput(
                logits=torch.FloatTensor(final_scores)
                # Optionally add other outputs from the base model if needed:
                # hidden_states=outputs.hidden_states,
                # attentions=outputs.attentions,
            )


class RewardModel:
    """
    Main Reward Model class using the Transformers library implementation.
    Loads a pre-trained model and value head, provides scoring.
    """

    def __init__(self, config: Config):
        """
        Initializes the RewardModel.

        Args:
            config: The configuration object.
        """
        self.config = config
        self.model: Optional[RewardModelWithValueHead] = None  # Will hold the combined model
        self.tokenizer: Optional[AutoTokenizer] = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"RewardModel initialized (Transformers). Device: {self.device}")

    def init(self):
        """
        Initialize the language model and tokenizer using Transformers.
        """
        start_time = datetime.now()
        logger.info(f"Initializing Reward Model from: {self.config.reward_model}")
        logger.info(f"Using device: {self.device}")

        try:
            # 1. Load Tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config.reward_model,  # Should point to the base model ID/path
                trust_remote_code=True
            )
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                logger.info(f"Set RewardModel pad token to EOS token: {self.tokenizer.pad_token}")

            # 2. Load Base Model (e.g., the fine-tuned policy model)
            # Use appropriate dtype from config
            model_dtype = getattr(torch, self.config.dtype, torch.float32)
            base_model = AutoModelForCausalLM.from_pretrained(
                self.config.reward_model,
                trust_remote_code=True,
                torch_dtype=model_dtype,
                # Add quantization config here if needed for the base model
                # quantization_config=bnb_config, # Example
            )

            # 3. Wrap with Value Head
            self.model = RewardModelWithValueHead(base_model)

            # 4. Move to device and set to eval mode
            self.model.to(self.device)
            self.model.eval()

            end_time = datetime.now()
            self.config.model_initialization_times["reward"] = end_time - start_time
            logger.info(f"Reward Model initialized successfully in {end_time - start_time}.")

        except Exception as e:
            logger.error(f"Failed to initialize Reward Model: {e}", exc_info=True)
            self.model = None
            self.tokenizer = None
            self.config.model_initialization_times[
                "reward"] = datetime.now() - start_time  # Record time even on failure

    @torch.no_grad()  # Ensure no gradients are computed during scoring
    def score(self, prompts: List[str], batch_size: int = 8) -> List[float]:
        """
        Scores a list of prompts (reasoning sequences) using the loaded model.

        Args:
            prompts: A list of strings, where each string is a complete
                     sequence (e.g., question + reasoning steps) to be scored.
            batch_size: The batch size to use for inference.

        Returns:
            A list of float scores, one for each prompt. Returns empty list on error.
        """
        if self.model is None or self.tokenizer is None:
            logger.error("Reward Model not initialized. Cannot score.")
            return [0.0] * len(prompts)  # Return dummy scores

        self.model.eval()  # Ensure model is in eval mode

        all_scores = []
        try:
            for i in range(0, len(prompts), batch_size):
                batch_prompts = prompts[i: i + batch_size]

                inputs = self.tokenizer(
                    batch_prompts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=self.config.max_model_len,  # Use max length from config
                ).to(self.device)

                # Get scores from the RewardModelWithValueHead forward pass
                outputs = self.model(**inputs, return_dict=True)
                scores = outputs.logits  # Logits are the final scalar scores

                all_scores.extend(scores.cpu().float().tolist())

            if len(all_scores) != len(prompts):
                logger.error(f"Scoring mismatch: Expected {len(prompts)} scores, got {len(all_scores)}")
                # Handle mismatch, maybe return dummy scores for consistency
                return [0.0] * len(prompts)

            return all_scores

        except Exception as e:
            logger.error(f"Error during reward model scoring: {e}", exc_info=True)
            # Return dummy scores in case of any exception during batch processing
            return [0.0] * len(prompts)
