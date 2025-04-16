import argparse
import logging
import os
import sys
from dataclasses import dataclass
from functools import partial
from typing import Dict, List, Sequence, Tuple, Union, Any

import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,  # Can use this or custom collator
    set_seed
)

# --- Add Project Root to Path ---
# This allows importing modules from rstar_deepthink
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import the reward model components directly (or place them in a shared utils file)
from rstar_deepthink.llms.reward import RewardModelWithValueHead

# --- Basic Setup ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)]
)

logger = logging.getLogger(__name__)

IGNORE_INDEX = -100  # Used for labels, less relevant for RM but good practice


# --- Data Processing ---

def preprocess_preference_dataset(
        examples: Dict[str, List[Any]],
        tokenizer: AutoTokenizer,
        max_length: int = 2048,
) -> Dict[str, List]:
    """
    Preprocesses the preference dataset.
    Takes examples with 'prefix', 'chosen', 'rejected'.
    Concatenates prefix + chosen and prefix + rejected, then tokenizes.
    """
    model_inputs = {
        "chosen_input_ids": [], "chosen_attention_mask": [],
        "rejected_input_ids": [], "rejected_attention_mask": [],
        # Add factor if needed, requires pos_count/neg_count in dataset
        # "factor": []
    }

    for i in range(len(examples["prefix"])):
        prefix = examples["prefix"][i]
        chosen_completion = examples["chosen"][i]
        rejected_completion = examples["rejected"][i]

        # Combine prefix with chosen/rejected completions
        # Add EOS token to signal end of sequence for the RM
        chosen_text = prefix + chosen_completion + tokenizer.eos_token
        rejected_text = prefix + rejected_completion + tokenizer.eos_token

        # Tokenize chosen sequence
        chosen_tokenized = tokenizer(
            chosen_text,
            max_length=max_length,
            padding=False,  # Padding handled by collator
            truncation=True,
            add_special_tokens=False  # Assume prefix/completion don't need extra special tokens here
        )

        # Tokenize rejected sequence
        rejected_tokenized = tokenizer(
            rejected_text,
            max_length=max_length,
            padding=False,  # Padding handled by collator
            truncation=True,
            add_special_tokens=False
        )

        model_inputs["chosen_input_ids"].append(chosen_tokenized["input_ids"])
        model_inputs["chosen_attention_mask"].append(chosen_tokenized["attention_mask"])
        model_inputs["rejected_input_ids"].append(rejected_tokenized["input_ids"])
        model_inputs["rejected_attention_mask"].append(rejected_tokenized["attention_mask"])

        # --- Factor Calculation (Example) ---
        # If 'metadata' with 'pos_count', 'neg_count' exists in your preference.jsonl
        # metadata = examples.get("metadata", [{}])[i] # Get metadata for current example
        # neg_count = metadata.get("neg_count", 1) # Default to 1 if not present
        # pos_count = metadata.get("pos_count", 1) # Default to 1 if not present
        # if neg_count == 0 or pos_count == 0:
        #     factor = 1.0 / max(1, neg_count + pos_count) # Avoid division by zero
        # else:
        #     factor = 1.0 / (neg_count * pos_count)
        # model_inputs["factor"].append(factor)
        # --- End Factor Calculation ---

    return model_inputs


@dataclass
class PairwiseDataCollatorWithPadding(DataCollatorForSeq2Seq):
    """
    Data collator for pairwise data.
    Concatenates chosen and rejected examples into a single batch.
    """

    def __call__(self, features: Sequence[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        concatenated_features = []
        # First add all chosen examples, then all rejected examples
        for feature in features:
            concatenated_features.append({
                "input_ids": feature["chosen_input_ids"],
                "attention_mask": feature["chosen_attention_mask"],
                # "factor": feature.get("factor"), # Include factor if using it
            })
        for feature in features:
            concatenated_features.append({
                "input_ids": feature["rejected_input_ids"],
                "attention_mask": feature["rejected_attention_mask"],
                # "factor": feature.get("factor"), # Include factor if using it
            })

        # Use the parent DataCollatorForSeq2Seq to handle padding
        batch = self.tokenizer.pad(
            concatenated_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
        )
        # Remove factor from batch if it exists, as it's handled separately in trainer
        # if "factor" in batch:
        #    del batch["factor"]
        return batch


# --- Custom Trainer for Pairwise Loss ---

class RMTrainer(Trainer):
    """
    Trainer subclass to compute the pairwise ranking loss for reward modeling.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.can_return_loss = True  # Ensure compute_loss can return loss

    def compute_loss(
            self, model, inputs: Dict[str, torch.Tensor], return_outputs: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        """
        Computes the pairwise ranking loss.

        Args:
            model: The RewardModelWithValueHead.
            inputs: Batch inputs prepared by PairwiseDataCollatorWithPadding.
                    Contains concatenated chosen and rejected sequences.
            return_outputs: Whether to return model outputs along with the loss.

        Returns:
            The computed loss tensor, or a tuple of (loss, outputs) if return_outputs is True.
        """
        # Extract factor if it was included in the inputs
        # factor = inputs.pop("factor", None) # Example if using factor

        # The model forward pass expects standard transformer inputs
        # It internally handles extracting the score for the last token
        outputs = model(**inputs, return_dict=True)
        all_rewards = outputs.logits  # Shape: (batch_size * 2,)

        # Split the rewards into chosen and rejected scores
        # First half corresponds to chosen, second half to rejected
        batch_size = all_rewards.size(0) // 2
        chosen_rewards = all_rewards[:batch_size]
        rejected_rewards = all_rewards[batch_size:]

        # Calculate the pairwise loss: -log(sigmoid(chosen - rejected))
        # Ensure float for stability with F.logsigmoid
        loss = -F.logsigmoid(chosen_rewards.float() - rejected_rewards.float())

        # --- Apply Factor Weighting (Example) ---
        # if factor is not None:
        #     # Ensure factor tensor is correctly shaped and on the right device
        #     factor = factor.to(loss.device).float()
        #     # Assuming factor was collated correctly (e.g., corresponds to chosen pairs)
        #     weighted_loss = loss * factor[:batch_size] # Apply weight
        #     final_loss = weighted_loss.mean() # Use mean for weighted loss
        # else:
        #     final_loss = loss.mean() # Use mean loss if no weighting
        # --- End Factor Weighting ---

        final_loss = loss.mean()  # Default: use mean loss across the batch

        if return_outputs:
            # Return model outputs alongside loss if requested
            # Note: 'outputs' here are the raw model outputs before splitting
            return final_loss, {"loss": final_loss, "chosen_rewards": chosen_rewards,
                                "rejected_rewards": rejected_rewards}
        return final_loss


# --- Evaluation Metric ---

@dataclass
class ComputeAccuracy:
    """
    Computes reward model accuracy (fraction of pairs where chosen > rejected).
    """

    def __call__(self, eval_preds) -> Dict[str, float]:
        """
        Computes accuracy from evaluation predictions.

        Args:
            eval_preds: Dataclass containing predictions and label_ids.
                        For RM, predictions are the reward scores.

        Returns:
            Dictionary containing the accuracy metric.
        """
        # eval_preds.predictions contains the concatenated reward scores
        rewards = eval_preds.predictions
        if rewards is None or len(rewards.shape) == 0 or rewards.shape[0] == 0:
            logger.warning("Received empty predictions in ComputeAccuracy. Returning accuracy 0.")
            return {"accuracy": 0.0}

        batch_size = rewards.shape[0] // 2
        chosen_scores = rewards[:batch_size]
        rejected_scores = rewards[batch_size:]

        # Calculate accuracy: proportion of pairs where chosen_score > rejected_score
        accuracy = np.mean(chosen_scores > rejected_scores)
        return {"accuracy": accuracy}


# --- Main Training Script ---

if __name__ == "__main__":
    # --- Argument Parsing ---
    # Simple arg parser, replace with HfArgumentParser or integrate with Config if needed
    parser = argparse.ArgumentParser(description="Train a Reward Model (PPM).")
    parser.add_argument('--config_path', type=str, default=None, help="Path to project YAML config file (optional).")
    parser.add_argument('--model_name_or_path', type=str, required=True, help="Path or HF ID of the base model.")
    parser.add_argument('--preference_dataset_path', type=str, required=True,
                        help="Path to the preference.jsonl dataset.")
    parser.add_argument('--output_dir', type=str, required=True, help="Directory to save the trained reward model.")
    parser.add_argument('--max_seq_length', type=int, default=2048, help="Maximum sequence length for tokenization.")
    parser.add_argument('--num_train_epochs', type=int, default=1, help="Number of training epochs.")
    parser.add_argument('--per_device_train_batch_size', type=int, default=4, help="Batch size per GPU for training.")
    parser.add_argument('--per_device_eval_batch_size', type=int, default=8, help="Batch size per GPU for evaluation.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=8, help="Gradient accumulation steps.")
    parser.add_argument('--learning_rate', type=float, default=5e-6, help="Initial learning rate.")
    parser.add_argument('--lr_scheduler_type', type=str, default='cosine', help="Learning rate scheduler type.")
    parser.add_argument('--warmup_ratio', type=float, default=0.03, help="Warmup ratio for LR scheduler.")
    parser.add_argument('--logging_steps', type=int, default=10, help="Log metrics every N steps.")
    parser.add_argument('--eval_steps', type=int, default=100, help="Evaluate every N steps.")
    parser.add_argument('--save_steps', type=int, default=100, help="Save checkpoint every N steps.")
    parser.add_argument('--save_total_limit', type=int, default=2, help="Limit the total number of checkpoints.")
    parser.add_argument('--seed', type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument('--bf16', action='store_true', help="Enable BF16 training.")
    parser.add_argument('--gradient_checkpointing', action='store_true', help="Enable gradient checkpointing.")
    parser.add_argument('--report_to', type=str, default="wandb",
                        help="Integration for reporting metrics (e.g., 'wandb', 'none').")
    parser.add_argument('--test_size', type=float, default=0.05, help="Fraction of data to use for evaluation set.")

    args = parser.parse_args()

    # --- Setup ---
    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    # --- Load Config (Optional) ---
    # If using a central Config object
    # config = Config() # Load defaults + potentially from args.config_path
    # Replace args.X with config.X below

    # --- Load Tokenizer ---
    logger.info(f"Loading tokenizer: {args.model_name_or_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        logger.info(f"Set pad token to EOS token: {tokenizer.pad_token}")
    # RM typically doesn't need right padding, default left is fine for inference/scoring
    # tokenizer.padding_side = "right" # Only if needed for specific base models

    # --- Load Base Model ---
    logger.info(f"Loading base model: {args.model_name_or_path}")
    model_dtype = torch.bfloat16 if args.bf16 else torch.float32
    base_model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=True,
        torch_dtype=model_dtype,
        # Add quantization config here if needed
    )

    # --- Instantiate Reward Model ---
    logger.info("Instantiating RewardModelWithValueHead...")
    model = RewardModelWithValueHead(base_model)
    # Enable gradient checkpointing if requested
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        logger.info("Gradient checkpointing enabled.")

    # --- Load and Prepare Dataset ---
    logger.info(f"Loading preference dataset from: {args.preference_dataset_path}")
    try:
        raw_datasets = load_dataset('json', data_files=args.preference_dataset_path, split='train')
        # Split dataset
        raw_datasets = raw_datasets.train_test_split(test_size=args.test_size, seed=args.seed)
        logger.info(f"Dataset loaded and split: {raw_datasets}")
    except Exception as e:
        logger.error(f"Failed to load or split dataset: {e}", exc_info=True)
        sys.exit(1)

    # Preprocess dataset
    logger.info("Preprocessing dataset...")
    partial_preprocess_func = partial(
        preprocess_preference_dataset,
        tokenizer=tokenizer,
        max_length=args.max_seq_length
    )
    # Determine columns to remove after preprocessing
    # Example: remove original text columns if they exist
    remove_columns = ["prefix", "chosen", "rejected"]  # Adjust based on actual dataset columns
    tokenized_datasets = raw_datasets.map(
        partial_preprocess_func,
        batched=True,
        num_proc=max(1, os.cpu_count() // 2),  # Use multiple cores
        remove_columns=[col for col in remove_columns if col in raw_datasets['train'].column_names]
        # Safely remove columns
    )
    logger.info(f"Dataset preprocessed: {tokenized_datasets}")
    logger.info(f"Columns after preprocessing: {tokenized_datasets['train'].column_names}")

    # --- Training Arguments ---
    logger.info("Setting up Training Arguments...")
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        lr_scheduler_type=args.lr_scheduler_type,
        warmup_ratio=args.warmup_ratio,
        logging_dir=os.path.join(args.output_dir, 'logs'),
        logging_strategy="steps",
        logging_steps=args.logging_steps,
        evaluation_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        bf16=args.bf16,
        gradient_checkpointing=args.gradient_checkpointing,
        # gradient_checkpointing_kwargs={'use_reentrant': False} if args.gradient_checkpointing else None, # Recommended for newer torch versions
        report_to=args.report_to,
        seed=args.seed,
        load_best_model_at_end=True,  # Load best model based on eval metric
        metric_for_best_model="accuracy",  # Use accuracy to select best model
        greater_is_better=True,  # Higher accuracy is better
        remove_unused_columns=False,  # Important for custom trainer/collator
        label_names=[],  # No standard labels used in loss calculation
    )

    # --- Initialize Trainer ---
    logger.info("Initializing RMTrainer...")
    trainer = RMTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["test"],
        tokenizer=tokenizer,
        data_collator=PairwiseDataCollatorWithPadding(
            tokenizer=tokenizer,
            padding='max_length',  # Pad to max_length
            max_length=args.max_seq_length
        ),
        compute_metrics=ComputeAccuracy()
    )

    # --- Train ---
    logger.info("Starting training...")
    train_result = trainer.train()

    # --- Save Final Model & Metrics ---
    logger.info("Training finished. Saving final model and metrics...")
    final_metrics = train_result.metrics
    trainer.log_metrics("train", final_metrics)
    trainer.save_metrics("train", final_metrics)
    trainer.save_state()
    trainer.save_model(args.output_dir)  # Saves the full model including value head
    logger.info(f"Model saved to {args.output_dir}")

    # --- Final Evaluation ---
    logger.info("Running final evaluation...")
    eval_metrics = trainer.evaluate(eval_dataset=tokenized_datasets["test"])
    trainer.log_metrics("eval", eval_metrics)
    trainer.save_metrics("eval", eval_metrics)
    logger.info(f"Final Evaluation Metrics: {eval_metrics}")

    logger.info("Reward Model training script finished successfully!")
