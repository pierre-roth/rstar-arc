import logging
import os
import sys

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from constants import NET_SCRATCH_PATH
from rstar_deepthink.config import Config
from utils import setup_logging

logger = logging.getLogger(__name__)

config = Config()
setup_logging(config.numeric_log_level)

# Path to the directory containing the saved LoRA adapter files (adapter_model.bin/safetensors and adapter_config.json)
"""Determine the LoRA adapter path based on training output naming convention"""
basename = config.policy_model.split('/')[-1]
# Adapter directories are saved under: models/fine_tuned/policy/ft-<basename>-<max_seq_len>-<learning_rate>-<lora_rank>-<lora_alpha>
adapter_base_dir = os.path.join(NET_SCRATCH_PATH, "models", "fine_tuned", "policy")
adapter_name = f"ft-{basename}-{config.max_seq_len}-{config.learning_rate}-{config.lora_rank}-{config.lora_alpha}"
LORA_ADAPTER_PATH = os.path.join(adapter_base_dir, adapter_name)

# Directory where the final merged model and tokenizer will be saved, matching adapter_name
merged_base_dir = os.path.join(NET_SCRATCH_PATH, "models", "policy")
MERGED_MODEL_OUTPUT_DIR = os.path.join(merged_base_dir, adapter_name)


def merge_lora_adapter(
        base_model_id: str,
        lora_adapter_path: str,
        output_dir: str,
        torch_dtype_str: str = "bfloat16",
):
    """
    Loads a base model and a LoRA adapter, merges them, and saves the result.

    Args:
        base_model_id (str): The Hugging Face model ID of the base model.
        lora_adapter_path (str): Path to the directory containing the saved LoRA adapter.
        output_dir (str): The directory where the merged model and tokenizer will be saved.
        torch_dtype_str (str): The torch dtype to use for loading ('bfloat16', 'float16', 'float32').
    """
    logger.info("--- Starting LoRA Merge Process ---")

    # --- Validate Inputs (from hardcoded vars) ---
    if not base_model_id:
        logger.error("BASE_MODEL_ID is not set in the script.")
        sys.exit(1)
    if not lora_adapter_path or not os.path.isdir(lora_adapter_path):
        logger.error(f"LORA_ADAPTER_PATH '{lora_adapter_path}' is invalid, not set, or not a directory.")
        sys.exit(1)
    if not output_dir:
        logger.error("MERGED_MODEL_OUTPUT_DIR is not set in the script.")
        sys.exit(1)

    # Check for adapter files
    adapter_model_file_bin = os.path.join(lora_adapter_path, "adapter_model.bin")
    adapter_model_file_st = os.path.join(lora_adapter_path, "adapter_model.safetensors")
    adapter_config_file = os.path.join(lora_adapter_path, "adapter_config.json")
    if not (os.path.exists(adapter_model_file_bin) or os.path.exists(adapter_model_file_st)) or not os.path.exists(
            adapter_config_file):
        logger.warning(
            f"Could not find 'adapter_model.bin'/'adapter_model.safetensors' or 'adapter_config.json' "
            f"in {lora_adapter_path}. Ensure the path points to a valid saved LoRA adapter directory."
        )
        # Continue for now, but loading might fail

    logger.info(f"Base Model ID: {base_model_id}")
    logger.info(f"LoRA Adapter Path: {lora_adapter_path}")
    logger.info(f"Output Directory: {output_dir}")
    logger.info(f"Torch dtype: {torch_dtype_str}")

    # --- Determine torch dtype ---
    if torch_dtype_str == "bfloat16":
        dtype = torch.bfloat16
    elif torch_dtype_str == "float16":
        dtype = torch.float16
    elif torch_dtype_str == "float32":
        dtype = torch.float32
    else:
        logger.warning(f"Invalid TORCH_DTYPE_STR '{torch_dtype_str}'. Defaulting to float32.")
        dtype = torch.float32

    # --- Load Base Model ---
    logger.info(f"Loading base model: {base_model_id} with dtype {dtype}...")
    try:
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_id,
            torch_dtype=dtype,
            device_map="auto",
            trust_remote_code=True,
        )
        logger.info("Base model loaded successfully.")
    except Exception as e:
        logger.error(f"Error loading base model '{base_model_id}': {e}", exc_info=True)
        sys.exit(1)

    # --- Load LoRA Adapter onto Base Model ---
    logger.info(f"Loading LoRA adapter from: {lora_adapter_path}...")
    try:
        model = PeftModel.from_pretrained(
            base_model,
            lora_adapter_path,
            # is_trainable=False # Optional: Set if no further training intended
        )
        logger.info("LoRA adapter loaded successfully.")
    except Exception as e:
        logger.error(f"Error loading LoRA adapter from '{lora_adapter_path}': {e}", exc_info=True)
        logger.error("Ensure the adapter path is correct and compatible with the base model.")
        sys.exit(1)

    # --- Merge LoRA Weights ---
    logger.info("Merging LoRA adapter weights into the base model...")
    try:
        merged_model = model.merge_and_unload()
        logger.info("LoRA weights merged successfully.")
    except Exception as e:
        logger.error(f"Error merging LoRA weights: {e}", exc_info=True)
        sys.exit(1)

    # --- Load Tokenizer ---
    logger.info(f"Loading tokenizer from adapter path: {lora_adapter_path}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(lora_adapter_path, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            logger.info("Set tokenizer pad_token to eos_token.")
        logger.info("Tokenizer loaded successfully.")
    except Exception as e:
        logger.error(f"Error loading tokenizer for '{base_model_id}': {e}", exc_info=True)
        sys.exit(1)

    # --- Save Merged Model and Tokenizer ---
    logger.info(f"Saving merged model and tokenizer to: {output_dir}...")
    try:
        os.makedirs(output_dir, exist_ok=True)
        merged_model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        logger.info("Merged model and tokenizer saved successfully.")
    except Exception as e:
        logger.error(f"Error saving merged model/tokenizer to '{output_dir}': {e}", exc_info=True)
        sys.exit(1)

    logger.info("--- LoRA Merge Process Completed ---")


if __name__ == "__main__":
    # Call the function using the hardcoded variables defined at the top
    merge_lora_adapter(
        base_model_id=config.policy_model,
        lora_adapter_path=LORA_ADAPTER_PATH,
        output_dir=MERGED_MODEL_OUTPUT_DIR,
        torch_dtype_str=config.dtype
    )
