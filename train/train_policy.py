import logging
import os
import sys

import torch
import wandb

# Configure wandb to use less resources
os.environ["WANDB_SILENT"] = "true"  # Reduce console output
os.environ["WANDB_CONSOLE"] = "off"  # Disable wandb console logging
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)

# --- Setup Logging ---
# Configure logging to output to console and optionally to a file
logging.basicConfig(
    level=logging.INFO,  # Log INFO level messages and above
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),  # Log to console
    ]
)

logger = logging.getLogger(__name__)  # Get logger for this module

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from rstar_deepthink.arc_task import ARCTask  # Example import
from rstar_deepthink.arc_task.task_utils import task_to_prompt  # Example import
from constants import NET_SCRATCH_PATH, SFT_SYSTEM_PROMPT, SFT_IN_BETWEEN_PROMPT

logger.info("Project root added to path and custom modules imported.")

# --- Configuration ---
logger.info("--- Configuration ---")
MODEL_ID = "Qwen/Qwen2.5-Coder-1.5B"
TRAINING_DATASET_PATH = os.path.join(NET_SCRATCH_PATH, "sft_data", f"round_{1}", "dataset_training.jsonl")
VALIDATION_DATASET_PATH = os.path.join(NET_SCRATCH_PATH, "sft_data", f"round_{1}", "dataset_validation.jsonl")
OUTPUT_DIR = os.path.join(NET_SCRATCH_PATH, "models", "fine_tuned", "policy", f"fine-tuned-{MODEL_ID.split('/')[1]}")
MAX_SEQ_LENGTH = 8192  # Adjust based on your data and GPU memory
WANDB_PROJECT = "deepthink-sft"  # Added wandb project name
WANDB_ENTITY = None  # Set to your team name or username if needed

logger.info(f"MODEL_ID: {MODEL_ID}")
logger.info(f"TRAINING_DATASET_PATH: {TRAINING_DATASET_PATH}")
logger.info(f"VALIDATION_DATASET_PATH: {VALIDATION_DATASET_PATH}")
logger.info(f"OUTPUT_DIR: {OUTPUT_DIR}")
logger.info(f"MAX_SEQ_LENGTH: {MAX_SEQ_LENGTH}")
logger.info(f"WANDB_PROJECT: {WANDB_PROJECT}")

# --- LoRA Configuration (specify which layers to adapt) ---
lora_config = LoraConfig(
    r=128,
    lora_alpha=16,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)
logger.info(
    f"LoraConfig: r={lora_config.r}, alpha={lora_config.lora_alpha}, dropout={lora_config.lora_dropout}, target_modules={lora_config.target_modules}")

# --- Training Arguments ---
run_name = f"{MODEL_ID.split('/')[-1]}-finetune-{os.path.basename(TRAINING_DATASET_PATH).split('.')[0]}"
training_arguments = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=1,  # Keep small for small models/memory
    gradient_accumulation_steps=16,  # Effective batch size = batch_size * grad_accum_steps
    optim="adamw_torch",  # Changed from paged_adamw_8bit to standard adamw
    learning_rate=4e-5,
    lr_scheduler_type="cosine",
    num_train_epochs=2,
    warmup_ratio=0.03,
    logging_strategy="steps",  # Log metrics every logging_steps
    logging_steps=10,  # Reduced frequency to minimize IO
    logging_first_step=True,  # Log metrics for the very first step
    save_strategy="steps",  # Save checkpoints every save_steps
    save_steps=100,  # Save checkpoint frequency
    save_total_limit=2,  # Keep only the last 2 checkpoints
    fp16=False,  # Enable fp16 for memory efficiency
    bf16=True,  # Disable bf16 if using fp16
    report_to="wandb",  # Use Weights & Biases with limited metrics
    log_level="warning",  # Reduce logging verbosity
    gradient_checkpointing=True,  # Save memory during training
    gradient_checkpointing_kwargs={'use_reentrant': False},  # Recommended setting

    evaluation_strategy="steps",  # Evaluate every eval_steps
    eval_steps=10,  # Evaluation frequency (match save_steps is common)
    per_device_eval_batch_size=1,  # Can often be larger than train batch size
    load_best_model_at_end=True,  # Load the best model based on metric_for_best_model
    metric_for_best_model="eval_loss",  # Primary metric to determine the best model (lower is better)
    greater_is_better=False,  # False for loss and perplexity
    run_name=run_name,  # Descriptive run name for tracking
)
# Log the arguments dictionary for detailed records
logger.info(f"TrainingArguments: {training_arguments.to_dict()}")

# --- Load Tokenizer ---
logger.info(f"Loading tokenizer: {MODEL_ID}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)

# Set padding token if it's not already set
if tokenizer.pad_token is None:
    logger.warning("Tokenizer does not have a pad token. Setting pad_token to eos_token.")
    tokenizer.pad_token = tokenizer.eos_token

# Set padding side to 'right' for training Causal LMs
tokenizer.padding_side = "right"
logger.info(
    f"Tokenizer loaded. Pad token: '{tokenizer.pad_token}', ID: {tokenizer.pad_token_id}. Padding side set to '{tokenizer.padding_side}'.")


# --- Data Preprocessing Function ---
def preprocess_data(examples):
    """Formats prompt and solution, then tokenizes for Causal LM."""
    try:
        # Combine prompt and completion, adding EOS token for Causal LM training
        texts = [
            SFT_SYSTEM_PROMPT + task_to_prompt(
                ARCTask.from_dict(task_json)) + SFT_IN_BETWEEN_PROMPT + solution + tokenizer.eos_token
            for task_json, solution in zip(examples["task_json"], examples["solution"])
        ]
        # Tokenize, ensuring truncation and padding
        model_inputs = tokenizer(
            texts,
            max_length=MAX_SEQ_LENGTH,
            truncation=True,
            padding="max_length"  # Pad sequences to max_length for batching
        )
        # For Causal LM, labels are the same as input_ids. The model learns to predict the next token.
        # The loss function in Trainer handles shifting internally.
        model_inputs["labels"] = model_inputs["input_ids"].copy()

        # weights
        model_inputs["weight"] = [float(w) for w in examples["weight"]]

        return model_inputs
    except Exception as e:
        logger.error(f"Error during preprocessing: {e}")
        # Return empty dict or raise error depending on desired behavior
        return {}


# --- Load and Prepare Dataset ---
logger.info("Loading dataset...")
try:
    dataset = load_dataset("json", data_files={"train": TRAINING_DATASET_PATH, "validation": VALIDATION_DATASET_PATH})
    logger.info(f"Dataset loaded: {dataset}")

    dataset["train"] = dataset["train"].shuffle(seed=42)
    logger.info("Dataset shuffled.")

    logger.info("Preprocessing and tokenizing dataset (this may take a while)...")
    tokenized_datasets = dataset.map(
        preprocess_data,
        batched=True,
        remove_columns=[col for col in dataset["train"].column_names if col != "weight"],
        # Keep 'weight' for loss calculation
        num_proc=max(1, os.cpu_count() // 2)  # Use multiple cores if available
    )
    logger.info(f"Dataset preprocessing finished: {tokenized_datasets}")

except FileNotFoundError as e:
    logger.error(f"Dataset file not found: {e}. Please check paths: {TRAINING_DATASET_PATH}, {VALIDATION_DATASET_PATH}")
    sys.exit(1)  # Exit if data is missing
except Exception as e:
    logger.error(f"Error loading or processing dataset: {e}")
    sys.exit(1)

# --- Load Model ---
logger.info(f"Loading base model: {MODEL_ID} for LoRA fine-tuning...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    device_map="auto",  # Automatically distribute across available GPUs/CPU
    trust_remote_code=True,
    torch_dtype=torch.bfloat16
)
logger.info("Base model loaded.")

# Configure model for training
model.config.use_cache = False  # Disable cache for efficiency with gradient checkpointing
model.config.pretraining_tp = 1  # Usually 1, adjust if model requires different tensor parallelism

# --- Prepare Model for PEFT (LoRA) ---
logger.info("Applying LoRA configuration...")

# Apply LoRA configuration
model = get_peft_model(model, lora_config)
logger.info("LoRA adapter applied to the model.")

# Print trainable parameters to verify LoRA setup
logger.info("Trainable parameters overview:")
model.print_trainable_parameters()

# --- Initialize wandb (lightweight configuration) ---
logger.info("Initializing wandb with lightweight configuration...")
wandb.init(
    project=WANDB_PROJECT,
    entity=WANDB_ENTITY,
    name=run_name,
    config={
        "model_name": MODEL_ID.split('/')[-1],  # Just log model name without full path
        "lora_r": lora_config.r,
        "lora_alpha": lora_config.lora_alpha,
        "learning_rate": training_arguments.learning_rate,
        "effective_batch_size": training_arguments.per_device_train_batch_size * training_arguments.gradient_accumulation_steps,
        "epochs": training_arguments.num_train_epochs,
        "train_samples": len(tokenized_datasets["train"]),
        "validation_samples": len(tokenized_datasets["validation"]),
    }
)

# Calculate the trainable parameters percentage
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
trainable_params_pct = (trainable_params / total_params) * 100 if total_params > 0 else 0

# Log the calculated percentage directly
wandb.log({"trainable_params_pct": trainable_params_pct})

logger.info("wandb initialized successfully with lightweight logging.")

# --- Initialize Trainer ---
# Data Collator for Causal LM. mlm=False ensures standard next-token prediction.
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

logger_trainer = logging.getLogger("WeightedTrainer")  # Use a specific logger if desired


class WeightedTrainer(Trainer):  # Inherit from the Hugging Face Trainer
    # Match the specified signature, explicitly including num_items_in_batch
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        Computes the weighted loss for causal language modeling.

        Overrides the default Trainer.compute_loss to incorporate sample weights.
        Matches the expected signature including `num_items_in_batch`, although
        this specific parameter is not used in the weighted loss calculation itself.

        Args:
            model: The model to train.
            inputs (dict): The inputs and targets of the model. Needs to contain
                           a 'weight' key with sample weights in addition to standard
                           model inputs like 'input_ids', 'attention_mask', 'labels'.
            return_outputs (bool): Whether to return model outputs in addition to the loss.
            num_items_in_batch (int, optional): The number of items in the batch, passed
                                                by the Trainer's training_step. Ignored here.


        Returns:
            Union[float, Tuple[float, Any]]: The computed weighted loss, or a tuple of
                                             (loss, model_outputs) if return_outputs=True.
        """
        logger_trainer.debug(f"Received inputs keys in compute_loss: {list(inputs.keys())}")
        # Log if num_items_in_batch is provided (optional)
        # if num_items_in_batch is not None:
        #     logger_trainer.debug(f"Received num_items_in_batch: {num_items_in_batch}")

        # Pop weights BEFORE passing inputs to the model if they aren't model inputs
        sample_weights = inputs.pop("weight", None)  # Pop weights from inputs dict

        if sample_weights is None:
            logger_trainer.warning("Sample weights ('weight' key) not found in batch inputs. Using uniform weighting.")
            # Use the batch size derived from a known input key
            if 'input_ids' in inputs:
                batch_size = inputs['input_ids'].size(0)
                device = inputs['input_ids'].device
                sample_weights = torch.ones(batch_size, device=device)
                logger_trainer.debug(f"Defaulting to uniform weights of size {batch_size} on device {device}")
            else:
                # Cannot determine batch size, handle appropriately
                logger_trainer.error("Cannot determine batch size for default weights as 'input_ids' not found.")
                # Fallback: Try getting loss directly from model without weighting
                outputs = model(**inputs)  # Pass remaining inputs
                loss = outputs.loss if hasattr(outputs, "loss") else torch.tensor(0.0)
                return (loss, outputs) if return_outputs else loss
        else:
            # Ensure weights are a tensor and on the correct device
            if not isinstance(sample_weights, torch.Tensor):
                sample_weights = torch.tensor(sample_weights, dtype=torch.float32)
            # Move weights to the same device as inputs (assuming input_ids exists)
            if 'input_ids' in inputs:
                target_device = inputs['input_ids'].device
                if sample_weights.device != target_device:
                    sample_weights = sample_weights.to(target_device)
                logger_trainer.debug(
                    f"Using provided weights of size {sample_weights.size(0)} on device {sample_weights.device}")
            else:
                logger_trainer.error("Cannot determine target device for weights as 'input_ids' not found.")
                # Fallback: move to model's device if possible
                try:
                    target_device = next(model.parameters()).device
                    if sample_weights.device != target_device:
                        sample_weights = sample_weights.to(target_device)
                except StopIteration:  # Handle case where model has no parameters
                    logger_trainer.error("Cannot determine model device.")
                    # Handle error appropriately, maybe raise or use CPU?
                    pass

        # Prepare inputs for the model (filter out non-model keys like the original 'weight')
        # Identify expected model input keys (common ones listed, adapt if your model needs others)
        model_input_keys = ['input_ids', 'attention_mask', 'labels', 'position_ids']  # Add any other relevant keys
        model_inputs = {k: v for k, v in inputs.items() if k in model_input_keys}

        # Get standard model outputs using the filtered inputs
        outputs = model(**model_inputs)

        logits = outputs.get("logits")
        # Get labels from the original inputs dict (needed for loss calculation)
        labels = inputs.get("labels")

        if logits is not None and labels is not None:
            # --- Standard Causal LM Loss Calculation ---
            loss_fct = torch.nn.CrossEntropyLoss(reduction="none")  # Compute loss per token
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            # Flatten the tokens
            flat_logits = shift_logits.view(-1, shift_logits.size(-1))
            flat_labels = shift_labels.view(-1)

            # Filter out ignored indices (-100)
            active_loss = shift_labels.view(-1) != -100
            active_logits = flat_logits[active_loss]
            active_labels = flat_labels[active_loss]

            if active_logits.numel() > 0:  # Ensure there are active tokens
                # Calculate loss per token for active tokens
                per_token_loss = loss_fct(active_logits, active_labels)

                # --- Weighted Loss Calculation ---
                # Reshape per-token loss back to sequence shape (batch_size, seq_len-1), filling inactive spots with 0
                loss_unflattened = torch.zeros_like(shift_labels, dtype=logits.dtype)
                loss_unflattened.view(-1)[active_loss] = per_token_loss  # Place calculated losses back

                # Calculate mean loss per sequence (sum loss / number of active tokens in sequence)
                loss_per_sequence = loss_unflattened.sum(dim=1)
                active_tokens_per_sequence = (shift_labels != -100).sum(dim=1)
                # Avoid division by zero for sequences with no active tokens (edge case)
                active_tokens_per_sequence = torch.max(active_tokens_per_sequence,
                                                       torch.tensor(1.0, device=active_tokens_per_sequence.device))
                mean_loss_per_sequence = loss_per_sequence / active_tokens_per_sequence

                # Apply sample weights (element-wise multiplication)
                # Ensure sample_weights has the correct shape (batch_size,)
                if sample_weights.dim() == 0:  # Handle scalar weight if batch size is 1
                    sample_weights = sample_weights.unsqueeze(0)
                if sample_weights.size(0) != mean_loss_per_sequence.size(0):
                    logger_trainer.error(
                        f"Weight dimension mismatch: weights ({sample_weights.size(0)}) vs sequences ({mean_loss_per_sequence.size(0)})")
                    # Fallback or error handling needed
                    loss = mean_loss_per_sequence.mean()  # Fallback to unweighted mean
                else:
                    weighted_loss_per_sequence = mean_loss_per_sequence * sample_weights
                    # Final loss is the mean of weighted sequence losses
                    loss = weighted_loss_per_sequence.mean()
                    logger_trainer.debug(f"Computed weighted loss: {loss.item()}")
            else:
                # Handle cases where there are no active labels in the batch after shifting/masking
                logger_trainer.warning("No active labels found in batch after shifting/masking.")
                loss = torch.tensor(0.0, device=logits.device, requires_grad=True)  # Return zero loss

        elif hasattr(outputs, "loss"):
            # Fallback if model directly computes loss (e.g., some T5 variants)
            # Weighting might be approximate here, as we only have the final loss scalar
            logger_trainer.warning("Using model's pre-computed loss. Weighting by mean sample weight.")
            loss = outputs.loss * sample_weights.mean()  # Approximate weighting
        else:
            # If loss cannot be computed by either method
            logger_trainer.error(
                "Cannot compute loss: Logits/Labels missing, or model does not provide 'loss' attribute.")
            loss = torch.tensor(0.0, device=model.device, requires_grad=True)  # Return zero loss with grad requirement

        return (loss, outputs) if return_outputs else loss


logger.info("Initializing WeightedTrainer...")
trainer = WeightedTrainer(
    model=model,
    args=training_arguments,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    tokenizer=tokenizer,
    data_collator=data_collator,
)
logger.info("Trainer initialized.")

# --- Start Training ---
logger.info("Starting training...")
try:
    train_result = trainer.train()

    # --- Save Training Statistics and Final Model ---
    metrics = train_result.metrics
    metrics["train_samples"] = len(tokenized_datasets["train"])  # Add number of train samples

    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()  # Save optimizer, scheduler, etc.

    logger.info(f"Training finished. Saving final LoRA adapter weights to {OUTPUT_DIR}")
    # Saves only the trained LoRA adapter weights + config
    trainer.save_model(OUTPUT_DIR)
    logger.info(f"LoRA adapter saved to {OUTPUT_DIR}")

    # --- Explicit Final Evaluation ---
    # If load_best_model_at_end=True, the trainer already evaluated the best checkpoint.
    # This provides metrics for the *final* state, which might differ slightly.
    logger.info("Running final evaluation on the validation set using the *last* checkpoint state...")
    final_eval_results = trainer.evaluate(eval_dataset=tokenized_datasets["validation"])
    logger.info(f"Final Evaluation Results (last checkpoint): {final_eval_results}")
    # Log and save these final metrics separately if needed
    trainer.log_metrics("final_eval", final_eval_results)
    trainer.save_metrics("final_eval", final_eval_results)

    # Log only the key final metric to wandb
    wandb.log({
        "final_eval/loss": final_eval_results.get("eval_loss")
    })

except Exception as e:
    logger.error(f"An error occurred during training: {e}", exc_info=True)  # Log traceback
    wandb.finish()  # Make sure to close wandb run even on error
    sys.exit(1)

# Finish wandb run
wandb.finish()
logger.info("Script finished successfully!")
