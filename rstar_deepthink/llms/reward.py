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
            dtype: torch.dtype = torch.float16,
            device=None,
            dropout: float = 0.1,
    ):
        super().__init__()

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Import heavy ML libs only when initializing
        from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel
        from peft import PeftModel

        # If provided a loaded PreTrainedModel or PeftModel, use directly; else load from name/path
        if isinstance(model_or_name, (PreTrainedModel, PeftModel)):
            self.backbone = model_or_name.to(self.device)
            # Attempt to infer tokenizer name
            model_name_for_tok = getattr(self.backbone, "name_or_path", None)
            if model_name_for_tok is None and hasattr(self.backbone, "base_model"):
                model_name_for_tok = getattr(self.backbone.base_model, "name_or_path", None)
            if model_name_for_tok is None:
                model_name_for_tok = str(model_or_name)
        else:
            self.backbone = AutoModelForCausalLM.from_pretrained(
                model_or_name,
                torch_dtype=dtype,
                trust_remote_code=True,
            ).to(self.device)
            model_name_for_tok = model_or_name
        # -------------------------------------------------------------------

        hidden_size = getattr(self.backbone.config, "hidden_size", None)
        if hidden_size is None:
            raise ValueError("Could not infer hidden_size from backbone config")

        self.v_head = ValueHead(hidden_size, dropout).to(self.device)

        self.tokenizer = AutoTokenizer.from_pretrained(model_name_for_tok, use_fast=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.eval()

    # ----------------------------------------------------
    # Inference helpers
    # ----------------------------------------------------
    @torch.no_grad()
    def forward(self, input_ids: torch.LongTensor, attention_mask: torch.Tensor):
        """
        Args
        ----
        input_ids      : (B, L)
        attention_mask : (B, L)  – 1 for real tokens, 0 for padding
        Returns
        -------
        reward         : (B,)    – scalar per sequence
        """
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )

        # Last hidden state: (B, L, H)
        last_hidden = outputs.hidden_states[-1]

        # Index of last non-pad token for each sequence
        seq_lens = attention_mask.long().sum(dim=1) - 1  # (B,)

        # Fancy indexing instead of gather:
        batch_idx = torch.arange(last_hidden.size(0), device=last_hidden.device)
        last_token_h = last_hidden[batch_idx, seq_lens]  # (B, H)

        reward = self.v_head(last_token_h).squeeze(-1)  # (B,)
        return reward

    @torch.no_grad()
    def score(self, texts: list[str], max_length: int = 1024, batch_size: int = 8):
        """
        Plain-text → float score.
        """
        all_scores: list[float] = []
        for i in range(0, len(texts), batch_size):
            enc = self.tokenizer(
                texts[i: i + batch_size],
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            ).to(self.device)

            scores = self.forward(enc["input_ids"], enc["attention_mask"])
            all_scores.extend(scores.cpu().tolist())

        return all_scores

    def save_pretrained(self, output_dir: str):
        """
        Save the merged backbone model and reward head to a directory.
        """

        # Save backbone (PreTrainedModel supports save_pretrained)
        os.makedirs(output_dir, exist_ok=True)

        from transformers import PreTrainedModel

        if isinstance(self.backbone, PreTrainedModel):
            self.backbone.save_pretrained(output_dir)
        else:
            torch.save(self.backbone.state_dict(), os.path.join(output_dir, "pytorch_model.bin"))
        # Save value head weights
        torch.save(self.v_head.state_dict(), os.path.join(output_dir, "v_head.bin"))
        # Save tokenizer
        self.tokenizer.save_pretrained(output_dir)

    @classmethod
    def from_pretrained(
            cls,
            model_dir: str,
            *,
            dtype: torch.dtype = torch.float16,
            device=None,
            dropout: float = 0.1,
    ) -> "RewardModelModule":
        """
        Load a RewardModel from a directory saved by `save_pretrained`.
        """
        # Determine device
        dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
        # Import heavy transformers only when loading pretrained
        from transformers import AutoModelForCausalLM

        # Load backbone model
        backbone = AutoModelForCausalLM.from_pretrained(
            model_dir,
            torch_dtype=dtype,
            trust_remote_code=True,
        ).to(dev)
        # Instantiate reward model
        rm = cls(backbone, dtype=dtype, device=dev, dropout=dropout)
        # Load value head
        v_head_path = os.path.join(model_dir, "v_head.bin")
        if os.path.isfile(v_head_path):
            state = torch.load(v_head_path, map_location=rm.device)
            rm.v_head.load_state_dict(state)
        else:
            logger.warning(
                f"Value head file not found at {v_head_path}! THIS SHOULD NOT HAPPEN AND THE MODEL IS BROKEN.")
        return rm


class RewardModel:
    """
    High-level wrapper for RewardModelModule handling model loading and scoring.
    """

    def __init__(self, config: Config):
        self.config = config
        self.llm: RewardModelModule | None = None

    def init(self):
        """Load the reward model for inference, using base or fine-tuned weights."""
        # Map config.dtype (str or torch.dtype) to torch.dtype
        dt = self.config.dtype.lower()
        if dt in ("bfloat16", "bf16"):
            torch_dtype = torch.bfloat16
        elif dt in ("float16", "fp16"):
            torch_dtype = torch.float16
        elif dt in ("float32", "fp32"):
            torch_dtype = torch.float32
        else:
            raise ValueError(f"Unsupported dtype '{self.config.dtype}' for RewardModel")

        start = datetime.now()
        # Load fine-tuned model if necessary
        if self.config.use_reward_model:
            model_path = os.path.join(self.config.reward_model_dir, self.config.reward_model)
            self.llm = RewardModelModule.from_pretrained(
                model_dir=model_path,
                dtype=torch_dtype,
            )

        end = datetime.now()
        self.config.model_initialization_times["reward"] = end - start

    def score(self, texts: list[str]) -> list[float]:
        """
        Score a list of texts, returning a scalar reward per text.
        Uses config.batch_size (-1 for all-at-once) and config.max_seq_len.
        """

        if not self.config.use_reward_model:
            return [0.0] * len(texts)

        return self.llm.score(texts, max_length=self.config.max_seq_len, batch_size=self.config.reward_batch_size)
