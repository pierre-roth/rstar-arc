"""
Train a reward model on positive–negative preference pairs.

Everything is driven by rstar_deepthink.Config:
  • model and dataset paths
  • LoRA hyper-params
  • batch-size, LR, logging, etc.
Multi-GPU training is handled via Accelerate.

Launch with:
  # Example: 4 GPUs with torchrun
  torchrun --nproc_per_node 4 train/train_reward.py --config-file configs/train_reward.yaml
or
  # Example: via Accelerate
  accelerate launch train/train_reward.py --config-file configs/train_reward.yaml
"""

from __future__ import annotations

import logging
import os
import sys
from functools import partial
from typing import Any, Sequence, Dict, List

import torch
import torch.nn.functional as F  # Renamed for conventional alias
import wandb
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    set_seed,
    PreTrainedTokenizerBase,
)

# Ensure project root is in path to import custom modules
# This assumes the script is in a subdirectory like 'train/'
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from train_utils import maybe_peft_wrap  # Assuming train_utils.py is in the same directory or accessible
from constants import NET_SCRATCH_PATH, SFT_SYSTEM_PROMPT, SFT_IN_BETWEEN_PROMPT  # Project-specific constants
from rstar_deepthink import Config  # Project-specific Config class
from utils import setup_logging  # Project-specific logging setup
from rstar_deepthink.llms.reward import RewardModelModule  # The model being trained
from rstar_deepthink.arc_task import ARCTask  # Project-specific task representation
from rstar_deepthink.arc_task.task_utils import task_to_prompt  # Utility for task formatting

logger = logging.getLogger(__name__)

# ------------------- configuration and setup -------------------
config = Config()  # Load configuration (e.g., from YAML via hydra or argparse)
set_seed(config.seed or 42)  # Set seed for reproducibility
setup_logging(config.numeric_log_level)  # Configure logging verbosity

base_model_name = config.reward_model.split('/')[-1]

if not config.full_finetune:
    dir_name_parts = [
        f"ft-{base_model_name}",
        str(config.max_seq_len),
        str(config.learning_rate),
        f"lora_r{config.lora_rank}",
        f"lora_a{config.lora_alpha}",
    ]
else:
    dir_name_parts = [
        f"ft-{base_model_name}",
        str(config.max_seq_len),
        str(config.learning_rate),
    ]

OUT_DIR = os.path.join(
    NET_SCRATCH_PATH, "models", "fine_tuned", "reward", "-".join(dir_name_parts)
)
os.makedirs(OUT_DIR, exist_ok=True)
logger.info(f"Reward model output directory: {OUT_DIR}")

# A more explicit run name for logging / experiment tracking.
RUN_NAME = f"reward-{'-'.join(dir_name_parts)}"

# Log hardware overview
num_gpus = torch.cuda.device_count()
logger.info(f"GPUs visible: {num_gpus}")
if num_gpus > 0:
    logger.info(f"Current CUDA device: {torch.cuda.current_device()}")

# ------------------- tokenizer -------------------
logger.info(f"Loading tokenizer for reward model: {config.reward_model}")
tok: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(
    config.reward_model,
    trust_remote_code=True
)
# Ensure pad_token is set
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
    logger.info(f"Tokenizer pad_token was None, set to eos_token: {tok.eos_token}")
# Use right-padding so that the final token in each sequence is the last real token
tok.padding_side = "right"
logger.info(f"Tokenizer padding side set to: {tok.padding_side}")

# ------------------- model initialization -------------------
logger.info(f"Initializing base model for reward training: {config.reward_model}")
dtype = torch.bfloat16 if config.use_bf16 else torch.float16
base_model = AutoModelForCausalLM.from_pretrained(
    config.reward_model,
    torch_dtype=dtype,
    trust_remote_code=True,
)

# Configure model for training
base_model.config.use_cache = False  # Disable cache for gradient checkpointing compatibility
base_model.gradient_checkpointing_enable()  # Enable gradient checkpointing for memory efficiency
base_model.enable_input_require_grads()  # Enable input gradients, necessary for some PEFT methods

# Wrap base model with LoRA adapters if not full fine-tuning
peft_wrapped_model = maybe_peft_wrap(base_model, config)  # maybe_peft_wrap handles the config.full_finetune check

# Instantiate the RewardModelModule with the (potentially PEFT-wrapped) backbone
model = RewardModelModule(
    peft_wrapped_model,
    dtype=dtype,  # Pass dtype explicitly
    dropout=config.reward_value_head_dropout
)
logger.info("RewardModelModule initialized with backbone and value head.")


# ------------------- preprocessing -------------------


def preprocess_for_pairwise_pref(
        examples: Dict[str, Sequence[Any]], *, tokenizer: PreTrainedTokenizerBase, max_len: int
) -> Dict[str, List[Any]]:
    """
    Tokenizes a batch of preference pairs (chosen and rejected responses).
    The prompt is constructed from system prompt, task details, and prefix.
    EOS token is NOT appended here as padding is handled by the collator.
    """
    processed_examples: Dict[str, List[Any]] = {
        "chosen_input_ids": [], "chosen_attention_mask": [],
        "rejected_input_ids": [], "rejected_attention_mask": [],
        "weight": [],
    }

    for task_json, prefix, chosen_completion, rejected_completion, weight_val in zip(
            examples["task_json"], examples["prefix"], examples["chosen"], examples["rejected"], examples["weight"]
    ):
        task = ARCTask.from_dict(task_json)  # Assuming task_json is a string needing parsing
        task_prompt_segment = task_to_prompt(task)

        # Construct full prompts
        # Note: SFT_SYSTEM_PROMPT, SFT_IN_BETWEEN_PROMPT are global constants
        prompt_base = SFT_SYSTEM_PROMPT + task_prompt_segment + SFT_IN_BETWEEN_PROMPT + prefix

        chosen_text = prompt_base + chosen_completion
        rejected_text = prompt_base + rejected_completion

        # Tokenize without padding; collator will handle padding
        chosen_tokens = tokenizer(chosen_text, max_length=max_len, truncation=True, padding=False,
                                  add_special_tokens=False)
        rejected_tokens = tokenizer(rejected_text, max_length=max_len, truncation=True, padding=False,
                                    add_special_tokens=False)

        processed_examples["chosen_input_ids"].append(chosen_tokens["input_ids"])
        processed_examples["chosen_attention_mask"].append(chosen_tokens["attention_mask"])
        processed_examples["rejected_input_ids"].append(rejected_tokens["input_ids"])
        processed_examples["rejected_attention_mask"].append(rejected_tokens["attention_mask"])
        processed_examples["weight"].append(float(weight_val))

    return processed_examples


class PairwiseCollator(DataCollatorWithPadding):
    """
    Custom data collator for pairwise preference data.
    It pads chosen and rejected sequences separately before concatenating them
    into a single batch for the model. It also handles the 'weight' vector.
    The tokenizer used should have `padding_side="left"`.
    """

    def __call__(self, features: Sequence[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        # Separate chosen and rejected items to maintain their order for loss calculation
        chosen_items = [
            {"input_ids": f["chosen_input_ids"], "attention_mask": f["chosen_attention_mask"]}
            for f in features
        ]
        rejected_items = [
            {"input_ids": f["rejected_input_ids"], "attention_mask": f["rejected_attention_mask"]}
            for f in features
        ]
        weights = [f["weight"] for f in features]

        # Concatenate chosen and rejected items. The first half of the batch will be 'chosen',
        # the second half will be 'rejected'. This structure is expected by PairwiseTrainer.
        all_items_to_pad = chosen_items + rejected_items

        # Pad all items together using the tokenizer's padding settings
        batch = self.tokenizer.pad(
            all_items_to_pad,
            return_tensors="pt",
            padding=self.padding,  # e.g., "longest"
            # max_length=self.max_length, # Optionally enforce max_length at collate time too
            # truncation=self.truncation, # Optionally truncate at collate time
        )

        # Add sample-wise weights for the pairwise loss (shape: B)
        batch["weight"] = torch.tensor(weights, dtype=torch.float32)

        # Add dummy labels. These are required by the Hugging Face Trainer's infrastructure
        # for running evaluation loops and computing metrics, even if the custom loss
        # function (compute_loss) doesn't use them directly for gradient calculation.
        # The content of labels doesn't matter here as long as they have the correct shape.
        batch["labels"] = batch["input_ids"].clone()  # Or torch.zeros_like(batch["input_ids"])

        return dict(batch)


class PairwiseTrainer(Trainer):
    """
    Custom Trainer for pairwise preference loss: L = −log σ(r_chosen − r_rejected) * sample_weight
    """

    def compute_loss(self, model, inputs, return_outputs: bool = False, **kwargs) -> torch.Tensor | tuple:
        """
        Computes the pairwise preference loss.
        The input batch is expected to have chosen sequences followed by rejected sequences.
        """
        # Extract weights and model inputs
        weights = inputs.pop("weight")  # Shape: (B_train,) where B_train is per-device train batch size
        # `labels` are also in inputs but ignored by model.forward and this loss.
        # The model's forward pass will receive input_ids and attention_mask.

        # The `inputs` dict passed to model() should contain 'input_ids' and 'attention_mask'.
        # `labels` are popped by Trainer if not used by model.forward signature.
        # If `labels` is still in `inputs` and model.forward doesn't accept it, it might cause an error.
        # RewardModelModule.forward explicitly accepts `labels=None`.

        # input_ids and attention_mask have shape (2 * B_train, L)
        # The first B_train are chosen, the next B_train are rejected.
        rewards = model(**inputs)  # Shape: (2 * B_train,)

        num_pairs = rewards.size(0) // 2
        chosen_rewards = rewards[:num_pairs]  # Shape: (B_train,)
        rejected_rewards = rewards[num_pairs:]  # Shape: (B_train,)

        # Compute pairwise loss
        # Ensure weights are on the same device as the loss terms
        loss_per_pair = -F.logsigmoid(chosen_rewards - rejected_rewards)  # Shape: (B_train,)

        # Apply sample weights
        weighted_loss = loss_per_pair * weights.to(loss_per_pair.device)

        # Normalize by sum of weights (or count of non-zero weights) to make loss independent of weight scale
        # Using sum of weights is common for weighted losses.
        # If all weights are 1, this is equivalent to .mean().
        loss = weighted_loss.sum() / weights.sum().clamp(min=1e-6)  # Avoid division by zero if all weights are zero

        if return_outputs:
            # For metric computation, return the difference in rewards (logits for accuracy)
            # Detach to prevent gradients from flowing back from metric computation.
            predictions = (chosen_rewards.detach() - rejected_rewards.detach())  # Shape: (B_train,)
            return loss, predictions  # Trainer expects (loss, outputs) where outputs can be (logits, labels)

        return loss


def compute_accuracy(eval_preds):
    """Computes accuracy: percentage of pairs where chosen > rejected."""
    # eval_preds.predictions are the (chosen_reward - rejected_reward) differences from PairwiseTrainer
    reward_differences = torch.tensor(eval_preds.predictions)  # Shape: (N_eval_pairs,)
    accuracy = (reward_differences > 0).float().mean().item()
    return {"accuracy": accuracy}


# ------------------- dataset loading and tokenization -------------------
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
total_params = sum(p.numel() for p in model.parameters())
logger.info(
    f"Trainable params: {trainable_params / 1e6:.2f} M / {total_params / 1e6:.2f} M "
    f"({trainable_params / total_params * 100:.2f}%)"
)

# Define paths to training and validation data
# These paths are assumed to be configured via `config.round_number`
TRAIN_PATH = os.path.join(NET_SCRATCH_PATH, "sft_data", f"round_{config.round_number}", "reward_dataset_training.jsonl")
VAL_PATH = os.path.join(NET_SCRATCH_PATH, "sft_data", f"round_{config.round_number}", "reward_dataset_validation.jsonl")
logger.info(f"Loading preference pairs from TRAIN: {TRAIN_PATH}, VALIDATION: {VAL_PATH}")

raw_datasets = load_dataset("json", data_files={"train": TRAIN_PATH, "validation": VAL_PATH})

# Columns to remove after tokenization (original text columns)
columns_to_remove = [c for c in ("task_json", "prefix", "chosen", "rejected") if
                     c in raw_datasets["train"].column_names]

# Partial function for preprocessing with fixed tokenizer and max_len
# Note: `tok` here is the globally defined tokenizer
preprocess_fn = partial(preprocess_for_pairwise_pref, tokenizer=tok, max_len=config.max_seq_len)

logger.info("Tokenizing datasets...")
tokenized_datasets = raw_datasets.map(
    preprocess_fn,
    batched=True,
    remove_columns=columns_to_remove,
    num_proc=max(1, os.cpu_count() // 2 if os.cpu_count() else 1),  # Ensure at least 1 proc
    desc="Running tokenizer on dataset",
)
logger.info(f"Dataset tokenization complete. Train examples: {len(tokenized_datasets['train'])}, "
            f"Validation examples: {len(tokenized_datasets['validation'])}")

# ------------------- training args -------------------
training_args = TrainingArguments(
    output_dir=OUT_DIR,
    # Batching
    per_device_train_batch_size=config.per_device_train_batch_size,
    per_device_eval_batch_size=config.per_device_eval_batch_size,
    gradient_accumulation_steps=config.gradient_accumulation_steps,
    # Optimization
    num_train_epochs=config.num_train_epochs,
    learning_rate=config.learning_rate,
    lr_scheduler_type=config.lr_scheduler_type,
    warmup_ratio=config.warmup_ratio,
    # Logging, Evaluation, Saving
    logging_dir=os.path.join(OUT_DIR, "logs"),  # Specific logging directory
    logging_strategy="steps",
    logging_steps=config.logging_steps,
    eval_strategy="steps",
    eval_steps=config.eval_steps,
    save_strategy="steps",  # Changed from "save_strategy"
    save_steps=config.save_steps,
    save_total_limit=config.save_total_limit,
    save_safetensors=False,  # As per original, can be True for .safetensors format
    # Precision
    bf16=config.use_bf16,
    fp16=not config.use_bf16,  # Only use fp16 if not bf16 and CUDA available
    # Distributed Training (handled by Accelerate launcher)
    # Miscellaneous
    seed=config.seed,
    run_name=RUN_NAME,  # Explicit run name with 'reward-' prefix
    report_to=config.report_to,  # Default to "none" if not set
    remove_unused_columns=False,  # Important: 'weight' column is used by collator/trainer
    load_best_model_at_end=True,
    metric_for_best_model="eval_accuracy",  # Ensure this metric is returned by compute_metrics
    greater_is_better=True,
    # ddp_find_unused_parameters=False, # Optionally set if facing DDP issues with PEFT
    weight_decay=config.weight_decay
)

# ------------------- weights & biases integration -------------------
if config.report_to == "wandb":
    logger.info("Initializing Weights & Biases for experiment tracking.")
    os.environ["WANDB_SILENT"] = "true"  # Suppress W&B informational messages
    os.environ["WANDB_CONSOLE"] = "off"   # Hide live updating console to reduce clutter
    wandb.init(
        project=config.wandb_project,
        entity=config.wandb_entity,
        name=training_args.run_name,  # Use run_name from TrainingArguments
        config={  # Log key hyperparameters
            "learning_rate": training_args.learning_rate,
            "train_batch_size": training_args.per_device_train_batch_size * \
                                training_args.gradient_accumulation_steps * \
                                (num_gpus if num_gpus > 0 else 1),
            "eval_batch_size": training_args.per_device_eval_batch_size * (num_gpus if num_gpus > 0 else 1),
            "num_epochs": training_args.num_train_epochs,
            "lora_rank": config.lora_rank if not config.full_finetune else "N/A",
            "lora_alpha": config.lora_alpha if not config.full_finetune else "N/A",
            "max_seq_len": config.max_seq_len,
            "base_model": config.reward_model,
            "output_dir": OUT_DIR,
            "train_samples": len(tokenized_datasets["train"]),
            "val_samples": len(tokenized_datasets["validation"]),
        }
    )

# ------------------- trainer initialization and execution -------------------
trainer = PairwiseTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    # tokenizer=tok, # Pass tokenizer if needed by Trainer internal (e.g. for generation during eval)
    # Not strictly needed if only using compute_metrics and custom collator handles all tokenization.
    data_collator=PairwiseCollator(tokenizer=tok, padding="longest"),  # Use the custom collator
    compute_metrics=compute_accuracy,  # Function to compute metrics during evaluation
)

logger.info(
    f"Starting training. Evaluation every {training_args.eval_steps} steps. Logging every {training_args.logging_steps} steps.")
train_result = trainer.train()
trainer.save_metrics("train", train_result.metrics)  # Save final training metrics

if config.report_to == "wandb":
    wandb.log({f"train/{k}": v for k, v in train_result.metrics.items()})  # Log final train metrics to W&B

logger.info(f"Training complete. Saving model components to {OUT_DIR}")

# Save the final model (best model if load_best_model_at_end=True)
# Trainer.save_model() saves the full state, including adapter if PEFT is used.
# For RewardModelModule, we need to save backbone (potentially PEFT) and v_head separately.

# If using PEFT, backbone is a PeftModel. save_pretrained saves adapter config and weights.
# If not PEFT, backbone is a PreTrainedModel. save_pretrained saves the full model.
model.backbone.save_pretrained(OUT_DIR)
logger.info(f"Backbone (adapter or full model) saved to {OUT_DIR}")

# Save the value head state dictionary
v_head_path = os.path.join(OUT_DIR, "v_head.bin")
torch.save(model.v_head.state_dict(), v_head_path)
logger.info(f"Value head saved to {v_head_path}")

# Also save the tokenizer for convenience during inference
model.tokenizer.save_pretrained(OUT_DIR)
logger.info(f"Tokenizer saved to {OUT_DIR}")

# ------------------- final evaluation -------------------
logger.info("Performing final evaluation on the validation set...")
eval_metrics = trainer.evaluate(eval_dataset=tokenized_datasets["validation"])
trainer.save_metrics("eval", eval_metrics)  # Save final evaluation metrics

final_accuracy = eval_metrics.get("eval_accuracy", 0.0)
logger.info(f"Final evaluation complete. Accuracy: {final_accuracy:.4f}")

if config.report_to == "wandb":
    wandb.log({"final_eval/accuracy": final_accuracy})  # Log final eval accuracy
    wandb.finish()  # Close W&B run

logger.info(f"Script finished. Model and metrics saved in {OUT_DIR}")
