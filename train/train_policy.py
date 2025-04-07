import os
import sys

import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)

# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from rstar_deepthink.arc_task import ARCTask
from rstar_deepthink.arc_task.task_utils import task_to_prompt
from constants import NET_SCRATCH_PATH


# --- Configuration ---
MODEL_ID = "Qwen/Qwen2.5-Coder-0.5B"  # The specific model from Hugging Face Hub
DATASET_PATH = os.path.join(NET_SCRATCH_PATH, "sft_data", f"round_{0}", "augmented.jsonl")
OUTPUT_DIR = os.path.join(NET_SCRATCH_PATH, "models", "fine_tuned", "policy")  # Directory to save the trained adapter
MAX_SEQ_LENGTH = 4096  # Adjust based on your data and GPU memory

# --- QLoRA Configuration (for efficiency) ---
# Use 4-bit quantization to reduce memory footprint
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",  # Use NF4 quantization
    bnb_4bit_compute_dtype=torch.bfloat16,  # Use bfloat16 for computation
    bnb_4bit_use_double_quant=True,  # Optional, can improve quality slightly
)

# --- LoRA Configuration (specify which layers to adapt) ---
lora_config = LoraConfig(
    r=16,  # Rank of the update matrices. Lower = fewer parameters, faster training
    lora_alpha=32,  # Alpha parameter for scaling. Common practice: alpha = 2*r
    # Specify modules to target. Common for Qwen-like models:
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
    lora_dropout=0.05,  # Dropout probability for LoRA layers
    bias="none",  # Usually set to 'none' for LoRA
    task_type="CAUSAL_LM",  # Specify the task type
)

# --- Training Arguments ---
training_arguments = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=1,  # Adjust based on GPU memory (start small)
    gradient_accumulation_steps=4,  # Increase effective batch size (batch_size * grad_accum)
    optim="paged_adamw_8bit",  # Optimizer suitable for QLoRA
    learning_rate=2e-4,  # Learning rate for LoRA
    lr_scheduler_type="cosine",  # Learning rate scheduler
    num_train_epochs=2,  # Start with 1 epoch, increase if needed
    warmup_ratio=0.03,  # Warmup steps proportion
    logging_steps=50,  # Log metrics every N steps
    save_steps=100,  # Save checkpoint every N steps
    save_total_limit=2,  # Keep only the last 2 checkpoints
    fp16=False,  # Use bf16 for training with Ampere+ GPUs
    bf16=True,  # Set to True if your GPU supports bfloat16 (recommended)
    # report_to="tensorboard",  # Or "wandb" or "none"
    gradient_checkpointing=True,  # Saves memory, but slows down training slightly
    gradient_checkpointing_kwargs={'use_reentrant': False},  # Recommended for newer PyTorch versions
    # You might need to add evaluation args if you have a validation set:
    # evaluation_strategy="steps",
    # eval_steps=100,
    # per_device_eval_batch_size=4,
)

# --- Load Tokenizer ---
# trust_remote_code=True is often required for newer models like Qwen
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
# Set padding token if it's not already set
if tokenizer.pad_token is None:
    print("Setting pad_token to eos_token")
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"  # Important for causal LM

print(f"Tokenizer loaded. Pad token ID: {tokenizer.pad_token_id}")


# --- Data Preprocessing Function ---
# This function formats the prompt and completion into a single sequence
# for causal language modeling.
def preprocess_data(examples):
    # Combine prompt and completion, adding EOS token at the end
    texts = [task_to_prompt(ARCTask.from_dict(task_json)) + solution for task_json, solution in zip(examples["task_json"], examples["solution"])]
    # Tokenize the combined texts
    # Ensure truncation to handle sequences longer than max_length
    model_inputs = tokenizer(
        texts,
        max_length=MAX_SEQ_LENGTH,
        truncation=True,
        padding="max_length"  # Pad sequences to max_length
    )
    # For Causal LM, the labels are the same as the input_ids.
    # The model learns to predict the next token.
    model_inputs["labels"] = model_inputs["input_ids"].copy()
    return model_inputs


# --- Load and Prepare Dataset ---
print("Loading dataset...")
# Assuming your dataset has 'train' split. Add 'validation' if you have one.
dataset = load_dataset("json", data_files={"train": DATASET_PATH})
print(f"Dataset loaded: {dataset}")

print("Preprocessing and tokenizing dataset...")
tokenized_datasets = dataset.map(
    preprocess_data,
    batched=True,  # Process multiple examples at once
    remove_columns=dataset["train"].column_names,  # Remove original columns
    num_proc=os.cpu_count() // 2  # Use multiple CPU cores for faster processing
)
print(f"Dataset preprocessing finished: {tokenized_datasets}")

# --- Load Model with Quantization ---
print(f"Loading base model: {MODEL_ID} with QLoRA config...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_config,
    device_map="auto",  # Automatically distribute model across available GPUs
    trust_remote_code=True,  # Required for some models like Qwen
    # torch_dtype=torch.bfloat16 # Optionally set dtype here if not using quantization fully
)
model.config.use_cache = False  # Disable cache during training for efficiency with gradient checkpointing
model.config.pretraining_tp = 1  # Set if needed, depends on model architecture, usually 1 is fine

# --- Prepare Model for PEFT (LoRA) ---
print("Preparing model for k-bit training and applying LoRA...")
# Prepare model for k-bit training (necessary for QLoRA)
model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=training_arguments.gradient_checkpointing)

# Apply LoRA configuration to the model
model = get_peft_model(model, lora_config)

# Print trainable parameters to verify LoRA setup
model.print_trainable_parameters()

# --- Initialize Trainer ---
# Data Collator for Causal LM. mlm=False ensures standard next-token prediction.
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

print("Initializing Trainer...")
trainer = Trainer(
    model=model,
    args=training_arguments,
    train_dataset=tokenized_datasets["train"],
    # eval_dataset=tokenized_datasets["validation"], # Uncomment if you have a validation set
    tokenizer=tokenizer,
    data_collator=data_collator,
)

# --- Start Training ---
print("Starting training...")
train_result = trainer.train()

# --- Save Training Statistics and Final Model ---
metrics = train_result.metrics
trainer.log_metrics("train", metrics)
trainer.save_metrics("train", metrics)

print(f"Training finished. Saving final LoRA adapter weights to {OUTPUT_DIR}")
# This saves only the trained LoRA adapter weights, not the entire base model.
trainer.save_model(OUTPUT_DIR)

print("Script finished successfully!")
