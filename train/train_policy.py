import json  # Import the json library
import logging
import os
import sys

import torch
import wandb
from datasets import load_dataset, Features, Value  # Import Features and Value
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)

# --- Setup Logging ---
# Configure logging to output to console
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
    ]
)
logger = logging.getLogger(__name__)

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
logger.info(f"Project root '{project_root}' added to sys.path.")

# Import custom modules after adding project root to path
from rstar_deepthink.arc_task import ARCTask
from rstar_deepthink.arc_task.task_utils import task_to_prompt
from constants import NET_SCRATCH_PATH

logger.info("Custom modules (ARCTask, task_to_prompt, NET_SCRATCH_PATH) imported successfully.")

# --- Prompts ---
SFT_SYSTEM_PROMPT = """Generate Python code step-by-step to solve the ARC task presented below. Implement the solution within a `solve(I)` function using the required markers."""
IN_BETWEEN_PROMPT = """Solution Code:"""

# --- Configuration ---
logger.info("--- Configuration ---")
MODEL_ID = "Qwen/Qwen2.5-Coder-0.5B"
TRAINING_DATASET_PATH = os.path.join(NET_SCRATCH_PATH, "sft_data", f"round_{1}", "dataset_training.jsonl")
VALIDATION_DATASET_PATH = os.path.join(NET_SCRATCH_PATH, "sft_data", f"round_{1}", "dataset_validation.jsonl")
OUTPUT_DIR = os.path.join(NET_SCRATCH_PATH, "models", "fine_tuned", "policy")

MAX_SEQ_LENGTH = 4096
WANDB_PROJECT = "deepthink-sft"
WANDB_ENTITY = None  # Set to your team name or username if needed

logger.info(f"MODEL_ID: {MODEL_ID}")
logger.info(f"TRAINING_DATASET_PATH: {TRAINING_DATASET_PATH}")
logger.info(f"VALIDATION_DATASET_PATH: {VALIDATION_DATASET_PATH}")
logger.info(f"OUTPUT_DIR: {OUTPUT_DIR}")
logger.info(f"MAX_SEQ_LENGTH: {MAX_SEQ_LENGTH}")
logger.info(f"WANDB_PROJECT: {WANDB_PROJECT}")

# --- LoRA Configuration ---
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
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    optim="adamw_torch",
    learning_rate=5e-5,
    lr_scheduler_type="cosine",
    num_train_epochs=2,
    warmup_ratio=0.03,
    logging_strategy="steps",
    logging_steps=100,
    logging_first_step=True,
    save_strategy="steps",
    save_steps=100,
    save_total_limit=2,
    fp16=False,  # Prefer bf16 if available
    bf16=True if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else False,  # Check bf16 support
    report_to="wandb",
    log_level="warning",  # Keep log level reasonable
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={'use_reentrant': False},
    evaluation_strategy="steps",
    eval_steps=100,
    per_device_eval_batch_size=1,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    run_name=run_name,
    remove_unused_columns=False,  # *** IMPORTANT: Keep 'weight' column for WeightedTrainer ***
)
logger.info(f"TrainingArguments: {training_arguments.to_dict()}")  # Log the dict representation

# --- Load Tokenizer ---
logger.info(f"Loading tokenizer: {MODEL_ID}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)

if tokenizer.pad_token is None:
    logger.warning("Tokenizer does not have a pad token. Setting pad_token to eos_token.")
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"
logger.info(
    f"Tokenizer loaded. Pad token: '{tokenizer.pad_token}', ID: {tokenizer.pad_token_id}. Padding side set to '{tokenizer.padding_side}'.")


# --- Data Preprocessing Function ---
def preprocess_data(examples):
    """Formats prompt and solution, then tokenizes for Causal LM."""
    model_inputs = {"input_ids": [], "attention_mask": [], "labels": [], "weight": []}

    # Get data from examples dictionary
    task_json_strings = examples.get("task_json", [])
    solutions = examples.get("solution", [])
    weights = examples.get("weight", [])

    # Process each example individually
    for i, (task_json_str, solution, weight) in enumerate(zip(task_json_strings, solutions, weights)):
        try:
            # Validate inputs
            if not task_json_str or not solution:
                logger.warning(f"Skipping example {i}: Missing task_json string or solution")
                continue

            # *** FIX: Parse the task_json string into a dictionary ***
            try:
                task_json_dict = json.loads(task_json_str)
            except json.JSONDecodeError as e:
                logger.warning(
                    f"Skipping example {i}: Error decoding task_json string: {e}. Content: {task_json_str[:100]}...")
                continue

            # Create the prompt using the parsed dictionary
            try:
                arc_task = ARCTask.from_dict(task_json_dict)  # Use the parsed dict
                task_prompt = task_to_prompt(arc_task)
            except Exception as e:
                logger.warning(f"Skipping example {i}: Error creating task prompt from parsed JSON: {e}")
                continue

            # Combine prompt and completion
            text = SFT_SYSTEM_PROMPT + task_prompt + IN_BETWEEN_PROMPT + solution + tokenizer.eos_token

            # Tokenize the individual example
            encoded = tokenizer(
                text,
                max_length=MAX_SEQ_LENGTH,
                truncation=True,
                padding="max_length"  # Pad to max_length for consistent tensor shapes
            )

            # Add to batch lists
            model_inputs["input_ids"].append(encoded["input_ids"])
            model_inputs["attention_mask"].append(encoded["attention_mask"])
            # For Causal LM, labels are typically the same as input_ids
            model_inputs["labels"].append(encoded["input_ids"].copy())
            # Ensure weight is float, default to 1.0 if missing/invalid
            try:
                model_inputs["weight"].append(float(weight) if weight is not None else 1.0)
            except (ValueError, TypeError):
                logger.warning(f"Invalid weight '{weight}' for example {i}. Using default 1.0.")
                model_inputs["weight"].append(1.0)


        except Exception as e:
            # Catch any other unexpected error during processing of a single example
            logger.error(f"Unexpected error processing example {i}: {e}", exc_info=True)
            continue  # Skip this example

    # Note: Conversion to tensors is handled by the Trainer/DataCollator
    return model_inputs


# --- Load and Prepare Dataset ---
logger.info("Loading dataset...")

# *** FIX: Define the features explicitly to load task_json as a string ***
data_features = Features({
    'task_name': Value('string'),
    'task_json': Value('string'),  # Load task_json as a string
    'solution': Value('string'),
    'weight': Value('float32')  # Load weight as float
})

try:
    dataset = load_dataset(
        "json",
        data_files={"train": TRAINING_DATASET_PATH, "validation": VALIDATION_DATASET_PATH},
        features=data_features  # Apply the defined features
    )
    logger.info(f"Dataset loaded successfully using explicit features: {dataset}")

    # Basic validation after loading
    if not dataset:
        raise ValueError("Dataset loaded is empty or invalid.")
    if "train" not in dataset or len(dataset["train"]) == 0:
        raise ValueError("Training split is missing or empty.")
    if "validation" not in dataset or len(dataset["validation"]) == 0:
        logger.warning("Validation split is missing or empty.")

    logger.info("Preprocessing and tokenizing dataset...")
    # Use num_proc=1 initially for safer debugging, increase later if needed and stable
    tokenized_datasets = dataset.map(
        preprocess_data,
        batched=True,
        batch_size=100,  # Adjust batch size based on memory
        num_proc=1,  # Start with 1 for stability, increase later
        # remove_columns=dataset["train"].column_names # Keep columns like 'weight'
    )

    # Verify the processed datasets have valid lengths
    for split in tokenized_datasets:
        if len(tokenized_datasets[split]) == 0:
            logger.error(f"Processed {split} dataset is empty. Check preprocessing logic and source data.")
            raise ValueError(f"Empty dataset after preprocessing: {split}")
        # Log a sample from the tokenized data for verification
        logger.info(f"Sample tokenized {split} entry keys: {tokenized_datasets[split][0].keys()}")

    logger.info(f"Dataset preprocessing finished: {tokenized_datasets}")

except FileNotFoundError as e:
    logger.error(f"Dataset file not found: {e}. Please check paths: {TRAINING_DATASET_PATH}, {VALIDATION_DATASET_PATH}")
    sys.exit(1)
except Exception as e:
    logger.error(f"Error loading or processing dataset: {e}", exc_info=True)
    sys.exit(1)

# --- Load Model ---
logger.info(f"Loading base model: {MODEL_ID}...")
# Determine torch_dtype based on training arguments
model_dtype = torch.bfloat16 if training_arguments.bf16 else (
    torch.float16 if training_arguments.fp16 else torch.float32)
logger.info(f"Using torch_dtype: {model_dtype}")

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    device_map="auto",
    trust_remote_code=True,
    torch_dtype=model_dtype  # Use determined dtype
)
logger.info("Base model loaded.")

model.config.use_cache = False  # Important for training with gradient checkpointing
model.config.pretraining_tp = 1  # Usually 1 for most models

# --- Prepare Model for PEFT (LoRA) ---
logger.info("Applying LoRA configuration...")
model = get_peft_model(model, lora_config)
logger.info("LoRA adapter applied to the model.")
logger.info("Trainable parameters overview:")
model.print_trainable_parameters()  # Prints to console

# --- Initialize Wandb ---
if training_arguments.report_to and "wandb" in training_arguments.report_to:
    logger.info("Initializing wandb...")
    try:
        wandb.init(
            project=WANDB_PROJECT,
            entity=WANDB_ENTITY,
            name=run_name,
            config=training_arguments.to_sanitized_dict()  # Log training args
        )
        # Log additional config
        wandb.config.update({
            "model_id": MODEL_ID,
            "max_seq_length": MAX_SEQ_LENGTH,
            "lora_r": lora_config.r,
            "lora_alpha": lora_config.lora_alpha,
            "lora_dropout": lora_config.lora_dropout,
            "lora_target_modules": list(lora_config.target_modules),
            "effective_batch_size": training_arguments.per_device_train_batch_size * training_arguments.gradient_accumulation_steps * training_arguments.world_size,
            # Consider world_size for multi-GPU
            "train_samples": len(tokenized_datasets["train"]),
            "validation_samples": len(tokenized_datasets.get("validation", []))  # Handle missing validation
        })
        logger.info("wandb initialized successfully.")
    except Exception as e:
        logger.error(f"Failed to initialize wandb: {e}", exc_info=True)
        # Optionally disable wandb reporting if init fails
        training_arguments.report_to = [r for r in training_arguments.report_to if r != "wandb"]
        logger.warning("Proceeding without wandb reporting.")

# --- Initialize Trainer ---
# Data Collator for Causal LM
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

logger_trainer = logging.getLogger("WeightedTrainer")


class WeightedTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        """
        Computes the weighted loss for causal language modeling.
        Assumes 'weight' is present in the inputs dictionary.
        """
        # Pop 'weight' from inputs. Ensure it's handled correctly by the data loading
        # and `remove_unused_columns=False` in TrainingArguments.
        sample_weights = inputs.pop("weight", None)

        # Get standard model outputs (logits, loss)
        outputs = model(**inputs)
        # Hugging Face models typically return loss directly when labels are provided
        loss = outputs.get("loss")
        logits = outputs.get("logits")
        labels = inputs.get("labels")  # Labels should still be in inputs if not popped

        if loss is not None and sample_weights is not None:
            # If loss is pre-computed by the model, we need to recompute it per-sequence
            # to apply weights correctly. This requires logits and labels.
            if logits is not None and labels is not None:
                loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
                # Shift so that tokens < n predict n
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()

                # Flatten the tokens and compute per-token loss
                flat_logits = shift_logits.view(-1, shift_logits.size(-1))
                flat_labels = shift_labels.view(-1)

                # Filter out padding tokens (-100)
                active_loss_mask = flat_labels != -100
                active_logits = flat_logits[active_loss_mask]
                active_labels = flat_labels[active_loss_mask]

                if active_logits.numel() > 0:
                    per_token_loss = loss_fct(active_logits, active_labels)

                    # Reshape per-token loss back to sequence shape to sum per sequence
                    loss_unflattened = torch.zeros_like(shift_labels, dtype=logits.dtype)
                    loss_unflattened.view(-1)[active_loss_mask] = per_token_loss

                    # Calculate loss per sequence (sum over sequence length)
                    loss_per_sequence = loss_unflattened.sum(dim=1)

                    # Normalize loss by the number of non-padding tokens in each sequence
                    active_tokens_per_sequence = (shift_labels != -100).sum(dim=1)
                    # Avoid division by zero for sequences with only padding/special tokens
                    active_tokens_per_sequence = torch.max(
                        active_tokens_per_sequence,
                        torch.tensor(1.0, device=active_tokens_per_sequence.device)
                    )
                    mean_loss_per_sequence = loss_per_sequence / active_tokens_per_sequence

                    # Ensure weights are on the correct device and apply them
                    sample_weights = sample_weights.to(mean_loss_per_sequence.device)
                    weighted_loss_per_sequence = mean_loss_per_sequence * sample_weights

                    # Final loss is the mean of weighted per-sequence losses
                    loss = weighted_loss_per_sequence.mean()
                else:
                    # Handle case where the batch contains only padding tokens after shifting
                    loss = torch.tensor(0.0, device=logits.device, requires_grad=True)
                    logger_trainer.warning("Batch contained no active tokens after shifting/masking.")

            else:
                # Cannot recompute loss, use approximate weighting on the model's loss
                logger_trainer.warning(
                    "Logits or labels missing, cannot compute exact weighted loss. Using approximate weighting on model's loss.")
                sample_weights = sample_weights.to(loss.device)
                loss = loss * sample_weights.mean()  # Approximate weighting

        elif loss is None:
            logger_trainer.error("Model did not return 'loss'. Cannot compute loss.")
            # Return 0 loss or raise error depending on desired behavior
            loss = torch.tensor(0.0, device=model.device, requires_grad=True)

        elif sample_weights is None:
            logger_trainer.warning("Sample weights ('weight' key) not found in batch inputs. Using unweighted loss.")
            # Loss is already computed by the model, just return it

        return (loss, outputs) if return_outputs else loss


logger.info("Initializing WeightedTrainer...")
trainer = WeightedTrainer(
    model=model,
    args=training_arguments,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets.get("validation"),  # Use .get for optional validation set
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
    # Ensure train_samples metric exists before trying to access/add
    if "train_samples" not in metrics:
        metrics["train_samples"] = len(tokenized_datasets["train"])

    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    logger.info(f"Training finished. Saving final LoRA adapter weights to {OUTPUT_DIR}")
    trainer.save_model(OUTPUT_DIR)  # Saves only the adapter weights by default with PEFT
    logger.info(f"LoRA adapter saved to {OUTPUT_DIR}")

    # --- Final Evaluation (if validation set exists) ---
    if tokenized_datasets.get("validation"):
        logger.info("Running final evaluation on the validation set...")
        final_eval_results = trainer.evaluate(eval_dataset=tokenized_datasets["validation"])
        logger.info(f"Final Evaluation Results: {final_eval_results}")
        trainer.log_metrics("final_eval", final_eval_results)
        trainer.save_metrics("final_eval", final_eval_results)

        # Log key final metric to wandb if enabled
        if training_arguments.report_to and "wandb" in training_arguments.report_to:
            final_loss = final_eval_results.get("eval_loss")
            if final_loss is not None:
                wandb.log({"final_eval/loss": final_loss})
            else:
                logger.warning("Could not find 'eval_loss' in final evaluation results to log to wandb.")

except Exception as e:
    logger.error(f"An error occurred during training or evaluation: {e}", exc_info=True)
    if training_arguments.report_to and "wandb" in training_arguments.report_to:
        wandb.finish(exit_code=1)  # Ensure wandb run is marked as failed
    sys.exit(1)

# Finish wandb run if successful
if training_arguments.report_to and "wandb" in training_arguments.report_to:
    wandb.finish()

logger.info("Script finished successfully!")
