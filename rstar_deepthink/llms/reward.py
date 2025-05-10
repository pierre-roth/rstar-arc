import logging
import os
from datetime import datetime

# 'torch' and 'nn' are used for tensor and module definitions
import torch
from torch import nn

# Heavy frameworks (PEFT, Transformers) are imported inside methods to keep module import light
from rstar_deepthink.config import Config

logger = logging.getLogger(__name__)


class ValueHead(nn.Sequential):
    """
    Tiny head: (hidden_size) → (1)
    """

    def __init__(self, hidden_size: int, dropout: float = 0.1):
        super().__init__(
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(hidden_size, 1),
        )
        # Very small normal init (helps for RLHF style training)
        # Accessing the Linear layer directly via self[1] assuming it's the second element.
        nn.init.normal_(self[1].weight, mean=5e-7, std=1e-6)
        nn.init.constant_(self[1].bias, 1e-6)


class RewardModelModule(nn.Module):
    """
    Wraps a decoder-only LM with a scalar reward head.
    """

    def __init__(
            self,
            model_or_name,
            *,
            dtype: torch.dtype = torch.bfloat16,
            device=None,
            dropout: float = 0.1,
    ):
        super().__init__()

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Import heavy ML libs only when initializing
        from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel
        from peft import PeftModel

        model_name_for_tokenizer: str

        if isinstance(model_or_name, (PreTrainedModel, PeftModel)):
            self.backbone = model_or_name.to(self.device)
            # Attempt to infer tokenizer name from the model object
            if hasattr(self.backbone, "name_or_path") and self.backbone.name_or_path:
                model_name_for_tokenizer = self.backbone.name_or_path
            elif hasattr(self.backbone, "base_model") and \
                    hasattr(self.backbone.base_model, "model") and \
                    hasattr(self.backbone.base_model.model, "name_or_path") and \
                    self.backbone.base_model.model.name_or_path:  # For PeftModel
                model_name_for_tokenizer = self.backbone.base_model.model.name_or_path
            else:
                model_name_for_tokenizer = str(model_or_name)
        else:
            self.backbone = AutoModelForCausalLM.from_pretrained(
                model_or_name,
                torch_dtype=dtype,
                trust_remote_code=True,
            ).to(self.device)
            model_name_for_tokenizer = model_or_name

        hidden_size = getattr(self.backbone.config, "hidden_size", None)
        if hidden_size is None:
            raise ValueError("Could not infer hidden_size from backbone config.")

        self.v_head = ValueHead(hidden_size, dropout).to(self.device, dtype=dtype)

        self.tokenizer = AutoTokenizer.from_pretrained(model_name_for_tokenizer, use_fast=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            logger.info(f"Tokenizer pad_token was None, set to eos_token: {self.tokenizer.eos_token}")

        self.eval()  # Set model to evaluation mode by default

    def forward(
            self,
            input_ids: torch.LongTensor,
            attention_mask: torch.Tensor,
            labels: torch.Tensor | None = None,  # Unused, but kept for API compatibility (e.g. with Trainer)
            **kwargs,
    ):
        """
        Forward pass to compute rewards.

        Args:
            input_ids (torch.LongTensor): Batch of input token IDs. Shape: (B, L)
            attention_mask (torch.Tensor): Batch of attention masks. Shape: (B, L)
            labels (torch.Tensor | None, optional): Unused in this model, present for API compatibility.

        Returns:
            torch.Tensor: Scalar reward for each sequence in the batch. Shape: (B,)
        """
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )

        # Get the hidden states of the last layer
        last_hidden = outputs.hidden_states[-1]  # Shape: (B, L, H)

        # Find the sequence lengths from the attention mask
        # (sum of non-padding tokens, then subtract 1 for 0-based indexing)
        seq_lens = attention_mask.long().sum(dim=1) - 1

        # Get the hidden state of the last non-padding token for each sequence
        # This uses advanced indexing: for each batch entry i, it selects hidden_states[i, seq_lens[i], :]
        last_token_h = last_hidden[
            torch.arange(last_hidden.size(0), device=last_hidden.device), seq_lens]  # Shape: (B, H)

        # Pass the last token's hidden state through the value head
        reward = self.v_head(last_token_h).squeeze(-1)  # Shape: (B,)

        return reward

    @torch.no_grad()
    def score(self, texts: list[str], max_length: int = 1024, batch_size: int = 8) -> list[float]:
        """
        Scores a list of texts and returns a scalar reward per text.

        Args:
            texts (list[str]): A list of plain text strings to score.
            max_length (int, optional): Maximum sequence length for tokenization. Defaults to 1024.
            batch_size (int, optional): Batch size for processing texts. Defaults to 8.

        Returns:
            list[float]: A list of float scores, one for each input text.
        """
        all_scores: list[float] = []
        self.eval()  # Ensure model is in eval mode

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i: i + batch_size]
            enc = self.tokenizer(
                batch_texts,
                padding=True,  # Pad to the longest sequence in the batch
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
                # padding_side is implicitly determined by the loaded tokenizer's config.
                # If trained with left padding, the saved tokenizer should reflect that.
            ).to(self.device)

            # The forward pass computes rewards based on the last non-padding token.
            # torch.tanh is applied here to scale the reward, typically to [-1, 1].
            scores = torch.tanh(self.forward(enc["input_ids"], enc["attention_mask"]))
            all_scores.extend(scores.cpu().tolist())

        return all_scores

    def save_pretrained(self, output_dir: str):
        """
        Save the backbone model, value head, and tokenizer to a directory.
        This method tries to save the backbone in a way that's compatible with
        Hugging Face's `from_pretrained` (if it's a PreTrainedModel).

        Args:
            output_dir (str): The directory where the model components will be saved.
        """
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Saving reward model components to {output_dir}")

        from transformers import PreTrainedModel
        from peft import PeftModel

        # Save backbone model
        if isinstance(self.backbone, (PreTrainedModel, PeftModel)):
            self.backbone.save_pretrained(output_dir)
            logger.info(f"Backbone ({type(self.backbone).__name__}) saved using save_pretrained.")
        else:
            # Fallback for models not supporting save_pretrained (e.g., plain nn.Module)
            backbone_path = os.path.join(output_dir, "pytorch_model.bin")  # Standard name
            torch.save(self.backbone.state_dict(), backbone_path)
            logger.info(f"Backbone ({type(self.backbone).__name__}) state_dict saved to {backbone_path}.")
            # Note: If the backbone isn't a PreTrainedModel, its config might need manual saving/handling.

        # Save value head weights
        v_head_path = os.path.join(output_dir, "v_head.bin")
        torch.save(self.v_head.state_dict(), v_head_path)
        logger.info(f"Value head saved to {v_head_path}.")

        # Save tokenizer
        self.tokenizer.save_pretrained(output_dir)
        logger.info(f"Tokenizer saved to {output_dir}.")

    @classmethod
    def from_pretrained(
            cls,
            model_dir: str,
            *,
            dtype: torch.dtype = torch.bfloat16,
            device=None,
            dropout: float = 0.1,
    ) -> "RewardModelModule":
        """
        Load a RewardModelModule from a directory previously saved by `save_pretrained`.

        Args:
            model_dir (str): Directory containing the saved model components.
            dtype (torch.dtype, optional): Desired torch dtype for the model. Defaults to torch.bfloat16.
            device (str | torch.device | None, optional): Device to load the model onto. Auto-detected if None.
            dropout (float, optional): Dropout rate for the value head. Defaults to 0.1.

        Returns:
            RewardModelModule: An instance of the loaded reward model.
        """
        dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Loading RewardModelModule from {model_dir} onto device {dev} with dtype {dtype}")

        from transformers import AutoModelForCausalLM  # Heavy import

        # Load backbone model
        # This assumes the backbone was saved in a way AutoModelForCausalLM can load it
        # (e.g., via save_pretrained or as a compatible pytorch_model.bin with a config.json)
        backbone = AutoModelForCausalLM.from_pretrained(
            model_dir,
            torch_dtype=dtype,
            trust_remote_code=True,  # Necessary for some custom model architectures
        ).to(dev)
        logger.info("Backbone loaded successfully.")

        # Instantiate the RewardModelModule with the loaded backbone
        # The tokenizer will be loaded from model_dir by the __init__ method
        reward_model_instance = cls(backbone, dtype=dtype, device=dev, dropout=dropout)

        # Load value head weights
        v_head_path = os.path.join(model_dir, "v_head.bin")
        if os.path.isfile(v_head_path):
            state_dict = torch.load(v_head_path, map_location=reward_model_instance.device)
            reward_model_instance.v_head.load_state_dict(state_dict)
            logger.info("Value head weights loaded successfully.")
        else:
            logger.warning(
                f"Value head file (v_head.bin) not found at {v_head_path}. "
                "The value head will have randomly initialized weights. "
                "This is expected if you are loading a base model before reward training, "
                "but an issue if loading a fully trained reward model."
            )
        return reward_model_instance


class RewardModel:
    """
    High-level wrapper for RewardModelModule, handling configuration-driven
    model loading and providing a simple scoring interface.
    """

    def __init__(self, config: Config):
        self.config = config
        self.llm: RewardModelModule | None = None
        self._dtype_map = {
            "bfloat16": torch.bfloat16,
            "bf16": torch.bfloat16,
            "float16": torch.float16,
            "fp16": torch.float16,
            "float32": torch.float32,
            "fp32": torch.float32,
        }

    def _get_torch_dtype(self) -> torch.dtype:
        """Maps string dtype from config to torch.dtype object."""
        dtype_str = str(self.config.dtype).lower()
        if dtype_str in self._dtype_map:
            return self._dtype_map[dtype_str]
        raise ValueError(f"Unsupported dtype '{self.config.dtype}' specified in config. "
                         f"Supported: {list(self._dtype_map.keys())}")

    def init(self):
        """
        Load the reward model for inference, using base or fine-tuned weights
        as specified in the configuration.
        """

        if not self.config.use_reward_model:
            logger.info("Reward model usage is disabled in the configuration. Skipping initialization.")
            self.config.model_initialization_times["reward"] = datetime.now() - datetime.now()
            return

        logger.info("Initializing reward model...")
        start_time = datetime.now()

        torch_dtype = self._get_torch_dtype()
        model_path = os.path.join(self.config.reward_model_dir, self.config.reward_model)

        logger.info(f"Loading fine-tuned reward model from: {model_path} with dtype: {torch_dtype}")
        self.llm = RewardModelModule.from_pretrained(
            model_dir=model_path,
            dtype=torch_dtype,
            # device will be auto-detected by from_pretrained if not specified,
            # or can be passed explicitly: device=self.config.device
        )

        end_time = datetime.now()
        initialization_time = end_time - start_time

        self.config.model_initialization_times["reward"] = initialization_time

        logger.info(f"Reward model initialized in {initialization_time}.")

    def score(self, texts: list[str]) -> list[float]:
        """
        Scores a list of texts using the initialized reward model.
        Returns a list of 0.0 if reward model usage is disabled.

        Args:
            texts (list[str]): A list of text strings to score.

        Returns:
            list[float]: A list of scalar rewards, one for each input text.
        """
        if not self.config.use_reward_model:
            return [0.0] * len(texts)

        logger.debug(f"Scoring {len(texts)} texts with reward model.")
        return self.llm.score(
            texts,
            max_length=self.config.max_seq_len,
            batch_size=self.config.reward_batch_size
        )
