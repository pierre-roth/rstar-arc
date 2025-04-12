import logging
import math
import os
import sys

import torch
import wandb
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
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
from constants import NET_SCRATCH_PATH  # Example import

logger.info("Project root added to path and custom modules imported.")

# --- Configuration ---
logger.info("--- Configuration ---")
MODEL_ID = "Qwen/Qwen2.5-Coder-0.5B"
# TODO: change file name to 'augmented.jsonl'
TRAINING_DATASET_PATH = os.path.join(NET_SCRATCH_PATH, "sft_data", f"round_{0}", "test_small.jsonl")
VALIDATION_DATASET_PATH = os.path.join(NET_SCRATCH_PATH, "sft_data", f"round_{0}", "validation.jsonl")
OUTPUT_DIR = os.path.join(NET_SCRATCH_PATH, "models", "fine_tuned", "policy")
# TODO: set max_seq_length based even higher
MAX_SEQ_LENGTH = 8192  # Adjust based on your data and GPU memory
WANDB_PROJECT = "deepthink-sft"  # Added wandb project name
WANDB_ENTITY = None  # Set to your team name or username if needed

logger.info(f"MODEL_ID: {MODEL_ID}")
logger.info(f"TRAINING_DATASET_PATH: {TRAINING_DATASET_PATH}")
logger.info(f"VALIDATION_DATASET_PATH: {VALIDATION_DATASET_PATH}")
logger.info(f"OUTPUT_DIR: {OUTPUT_DIR}")
logger.info(f"MAX_SEQ_LENGTH: {MAX_SEQ_LENGTH}")
logger.info(f"WANDB_PROJECT: {WANDB_PROJECT}")

# --- QLoRA Configuration (for efficiency) ---
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)
logger.info(
    f"BitsAndBytesConfig: load_in_4bit={bnb_config.load_in_4bit}, quant_type={bnb_config.bnb_4bit_quant_type}, compute_dtype={bnb_config.bnb_4bit_compute_dtype}")

# --- LoRA Configuration (specify which layers to adapt) ---
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
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
    gradient_accumulation_steps=8,  # Effective batch size = batch_size * grad_accum_steps
    optim="paged_adamw_8bit",
    learning_rate=2e-4,
    lr_scheduler_type="cosine",
    num_train_epochs=1,  # Start with 1, increase if needed
    warmup_ratio=0.03,
    logging_strategy="steps",  # Log metrics every logging_steps
    logging_steps=10,  # Log training loss frequently
    logging_first_step=True,  # Log metrics for the very first step
    save_strategy="steps",  # Save checkpoints every save_steps
    save_steps=100,  # Save checkpoint frequency
    save_total_limit=2,  # Keep only the last 2 checkpoints
    fp16=False,  # Disable fp16 if using bf16
    bf16=True,  # Enable bf16 for Ampere+ GPUs (preferred)
    report_to="wandb",  # Use Weights & Biases
    gradient_checkpointing=True,  # Save memory during training
    gradient_checkpointing_kwargs={'use_reentrant': False},  # Recommended setting

    evaluation_strategy="steps",  # Evaluate every eval_steps
    eval_steps=100,  # Evaluation frequency (match save_steps is common)
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
            task_to_prompt(ARCTask.from_dict(task_json)) + solution + tokenizer.eos_token
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

    logger.info("Preprocessing and tokenizing dataset (this may take a while)...")
    tokenized_datasets = dataset.map(
        preprocess_data,
        batched=True,
        remove_columns=dataset["train"].column_names,  # Remove original columns
        num_proc=max(1, os.cpu_count() // 2)  # Use multiple cores if available
    )
    logger.info(f"Dataset preprocessing finished: {tokenized_datasets}")

except FileNotFoundError as e:
    logger.error(f"Dataset file not found: {e}. Please check paths: {TRAINING_DATASET_PATH}, {VALIDATION_DATASET_PATH}")
    sys.exit(1)  # Exit if data is missing
except Exception as e:
    logger.error(f"Error loading or processing dataset: {e}")
    sys.exit(1)

# --- Load Model with Quantization ---
logger.info(f"Loading base model: {MODEL_ID} with QLoRA config...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_config,
    device_map="auto",  # Automatically distribute across available GPUs/CPU
    trust_remote_code=True,
    torch_dtype=torch.bfloat16  # Match compute dtype in bnb_config
)
logger.info("Base model loaded.")

# Configure model for training
model.config.use_cache = False  # Disable cache for efficiency with gradient checkpointing
model.config.pretraining_tp = 1  # Usually 1, adjust if model requires different tensor parallelism

# --- Prepare Model for PEFT (LoRA) ---
logger.info("Preparing model for k-bit training (if applicable) and applying LoRA...")
# Prepare model for k-bit training (required for QLoRA)
model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=training_arguments.gradient_checkpointing)
logger.info("Model prepared for k-bit training.")

# Apply LoRA configuration
model = get_peft_model(model, lora_config)
logger.info("LoRA adapter applied to the model.")

# Print trainable parameters to verify LoRA setup
logger.info("Trainable parameters overview:")
model.print_trainable_parameters()

# --- Initialize wandb ---
logger.info("Initializing wandb...")
wandb.init(
    project=WANDB_PROJECT,
    entity=WANDB_ENTITY,
    name=run_name,
    config={
        "model_name": MODEL_ID,
        "training_dataset": TRAINING_DATASET_PATH,
        "validation_dataset": VALIDATION_DATASET_PATH,
        "max_seq_length": MAX_SEQ_LENGTH,
        "lora_r": lora_config.r,
        "lora_alpha": lora_config.lora_alpha,
        "lora_dropout": lora_config.lora_dropout,
        "lora_target_modules": lora_config.target_modules,
        "learning_rate": training_arguments.learning_rate,
        "effective_batch_size": training_arguments.per_device_train_batch_size * training_arguments.gradient_accumulation_steps,
        "epochs": training_arguments.num_train_epochs,
        "warmup_ratio": training_arguments.warmup_ratio,
        "train_samples": len(tokenized_datasets["train"]),
        "validation_samples": len(tokenized_datasets["validation"]),
        "quantization": {
            "load_in_4bit": bnb_config.load_in_4bit,
            "quant_type": bnb_config.bnb_4bit_quant_type,
            "use_double_quant": bnb_config.bnb_4bit_use_double_quant,
        }
    }
)

# Log model architecture
model_info = {
    "model_config": model.config.to_dict(),
    "trainable_params": model.print_trainable_parameters(),
    "total_params": sum(p.numel() for p in model.parameters()),
}
wandb.log({"model_info": model_info})

# Log a sample prompt/completion pair for reference
if len(dataset["train"]) > 0:
    sample_idx = 0
    sample = dataset["train"][sample_idx]
    sample_task = ARCTask.from_dict(sample["task_json"])
    sample_prompt = task_to_prompt(sample_task)
    sample_completion = sample["solution"]
    wandb.log({
        "sample_data": {
            "prompt": sample_prompt,
            "completion": sample_completion
        }
    })

logger.info("wandb initialized successfully.")

# --- Initialize Trainer ---
# Data Collator for Causal LM. mlm=False ensures standard next-token prediction.
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

logger.info("Initializing Trainer...")
trainer = Trainer(
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

    # Log final results to wandb
    wandb.log({
        "final_eval/loss": final_eval_results.get("eval_loss"),
        "final_eval/perplexity": final_eval_results.get("perplexity", math.exp(final_eval_results.get("eval_loss", 0)))
    })

except Exception as e:
    logger.error(f"An error occurred during training: {e}", exc_info=True)  # Log traceback
    wandb.finish()  # Make sure to close wandb run even on error
    sys.exit(1)

# --- Qualitative Evaluation ---
logger.info("Performing qualitative evaluation on a few validation samples...")
GENERATION_PREFIX = "<beginning_of_code>\ndef solve(I):"

try:
    # Ensure the best model checkpoint is loaded if load_best_model_at_end=True
    # If not using load_best_model_at_end, the model variable holds the last state.
    # For robust qualitative eval, explicitly load the saved adapter.
    logger.info(f"Loading base model {MODEL_ID} again for inference...")
    base_model_for_eval = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,  # Use the same quantization
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16
    )
    logger.info(f"Loading fine-tuned LoRA adapter from {OUTPUT_DIR}...")
    model_for_inference = PeftModel.from_pretrained(base_model_for_eval, OUTPUT_DIR)
    model_for_inference = model_for_inference.merge_and_unload()  # Optional: Merge adapter for faster inference if memory allows
    model_for_inference.eval()  # Set to evaluation mode
    logger.info("Model and adapter loaded for inference.")

    # Reload tokenizer with left padding for generation
    tokenizer_for_eval = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    if tokenizer_for_eval.pad_token is None:
        tokenizer_for_eval.pad_token = tokenizer_for_eval.eos_token
    tokenizer_for_eval.padding_side = "left"  # Crucial for generation

    # Load a few raw validation examples
    raw_validation_dataset = load_dataset("json", data_files={"validation": VALIDATION_DATASET_PATH})["validation"]
    num_samples_to_check = 2  # Number of samples to generate for

    # Create a table to log sample predictions
    prediction_table = wandb.Table(columns=["sample_id", "prompt", "actual_solution", "generated_solution"])

    for i in range(min(num_samples_to_check, len(raw_validation_dataset))):
        example = raw_validation_dataset[i]
        prompt_text = task_to_prompt(ARCTask.from_dict(example["task_json"])) + "\n" + GENERATION_PREFIX
        actual_solution = example["solution"]

        logger.info(f"\n--- Qualitative Sample {i + 1} ---")
        logger.info(f"Prompt:\n{prompt_text}")
        logger.info(f"\nActual Solution:\n{actual_solution}")

        # Ensure prompt is not too long before encoding
        # Simple truncation here, more sophisticated handling might be needed
        max_prompt_len = MAX_SEQ_LENGTH - 150  # Reserve space for generation
        if len(tokenizer_for_eval.encode(prompt_text)) > max_prompt_len:
            logger.warning(f"Prompt for sample {i + 1} is too long, truncating.")
            # A simple way to truncate (might cut mid-word)
            prompt_text = tokenizer_for_eval.decode(tokenizer_for_eval.encode(prompt_text)[:max_prompt_len])

        inputs = tokenizer_for_eval(prompt_text, return_tensors="pt", padding=True, truncation=True,
                                    max_length=max_prompt_len).to(model_for_inference.device)

        # Generate output
        with torch.no_grad():
            outputs = model_for_inference.generate(
                **inputs,
                max_new_tokens=MAX_SEQ_LENGTH // 4,  # Max tokens to generate
                temperature=0.8,  # Control randomness (lower = more focused)
                top_p=0.95,  # Nucleus sampling
                do_sample=True,  # Use sampling
                pad_token_id=tokenizer_for_eval.pad_token_id,
                eos_token_id=tokenizer_for_eval.eos_token_id
            )

        # Decode only the newly generated part
        generated_text = tokenizer_for_eval.decode(outputs[0, inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        logger.info(f"\nGenerated Solution:\n{generated_text}")
        logger.info("-" * 30)

        # Add row to the wandb Table
        prediction_table.add_data(i, prompt_text, actual_solution, generated_text)

    # Log the table to wandb
    wandb.log({"sample_predictions": prediction_table})

except FileNotFoundError:
    logger.error(
        f"Could not load adapter from {OUTPUT_DIR} for qualitative evaluation. Ensure training saved the model correctly.")
except Exception as e:
    logger.error(f"An error occurred during qualitative evaluation: {e}", exc_info=True)

# Finish wandb run
wandb.finish()
logger.info("Script finished successfully!")
