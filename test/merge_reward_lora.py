"""
Merge a fine-tuned RewardModel LoRA adapter into the base LM and save the final merged reward model.
"""
import logging
import os
import sys

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM

# Ensure project root is on path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from constants import NET_SCRATCH_PATH
from rstar_deepthink.config import Config
from utils import setup_logging
from rstar_deepthink.llms.reward import RewardModelModule

logger = logging.getLogger(__name__)

config = Config()
setup_logging(config.numeric_log_level)

# Directory where the LoRA adapter and v_head were saved by train_reward.py
LORA_ADAPTER_PATH = f"{NET_SCRATCH_PATH}/models/fine_tuned/reward/fine-tuned-{config.policy_model}"

# Name of the value head state file
V_HEAD_FILENAME = "v_head.bin"

# Where to write the merged reward model (will contain merged LM weights, tokenizer, and v_head.bin)
MERGED_MODEL_OUTPUT_DIR = f"{NET_SCRATCH_PATH}/models/reward/fine-tuned-{config.policy_model}"


def merge_reward_adapter(
        base_model_id: str,
        lora_adapter_path: str,
        v_head_filename: str,
        output_dir: str,
        torch_dtype_str: str = "bfloat16",
):
    """
    Loads a base LM and RewardModel LoRA adapter, merges them, and saves the final RewardModel.
    """
    logger.info("--- Starting Reward LoRA Merge Process ---")
    # Validate inputs
    if not base_model_id:
        logger.error("BASE_MODEL_ID is not set.")
        sys.exit(1)
    if not lora_adapter_path or not os.path.isdir(lora_adapter_path):
        logger.error(f"LORA_ADAPTER_PATH '{lora_adapter_path}' is invalid or not a directory.")
        sys.exit(1)
    if not output_dir:
        logger.error("MERGED_MODEL_OUTPUT_DIR is not set.")
        sys.exit(1)

    # Check adapter files
    adapter_bin = os.path.join(lora_adapter_path, "adapter_model.bin")
    adapter_st = os.path.join(lora_adapter_path, "adapter_model.safetensors")
    adapter_cfg = os.path.join(lora_adapter_path, "adapter_config.json")
    v_head_file = os.path.join(lora_adapter_path, v_head_filename)
    if not (os.path.exists(adapter_bin) or os.path.exists(adapter_st)) or not os.path.exists(adapter_cfg):
        logger.warning(f"LoRA adapter files not found in {lora_adapter_path}.")
    if not os.path.exists(v_head_file):
        logger.warning(f"Value head file '{v_head_file}' not found. Skipping head load.")

    logger.info(f"Base Model ID: {base_model_id}")
    logger.info(f"LoRA Adapter Path: {lora_adapter_path}")
    logger.info(f"V Head File: {v_head_file}")
    logger.info(f"Output Directory: {output_dir}")
    logger.info(f"Torch dtype: {torch_dtype_str}")

    # Determine dtype
    if torch_dtype_str == "bfloat16":
        dtype = torch.bfloat16
    elif torch_dtype_str == "float16":
        dtype = torch.float16
    elif torch_dtype_str == "float32":
        dtype = torch.float32
    else:
        logger.warning(f"Invalid TORCH_DTYPE_STR '{torch_dtype_str}', defaulting to float32.")
        dtype = torch.float32

    # Load base model
    logger.info(f"Loading base model: {base_model_id} with dtype {dtype}...")
    try:
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_id,
            torch_dtype=dtype,
            device_map="auto",
            trust_remote_code=True,
        )
        logger.info("Base model loaded.")
    except Exception as e:
        logger.error(f"Error loading base model '{base_model_id}': {e}", exc_info=True)
        sys.exit(1)

    # Load LoRA adapter onto base
    logger.info(f"Loading LoRA adapter from: {lora_adapter_path}...")
    try:
        peft_model = PeftModel.from_pretrained(
            base_model,
            lora_adapter_path,
        )
        logger.info("LoRA adapter loaded.")
    except Exception as e:
        logger.error(f"Error loading LoRA adapter: {e}", exc_info=True)
        sys.exit(1)

    # Merge LoRA weights
    logger.info("Merging LoRA weights into base model...")
    try:
        merged_backbone = peft_model.merge_and_unload()
        logger.info("LoRA weights merged.")
    except Exception as e:
        logger.error(f"Error merging LoRA weights: {e}", exc_info=True)
        sys.exit(1)

    # Instantiate RewardModel
    logger.info("Instantiating RewardModel...")
    reward_model = RewardModelModule(merged_backbone, dtype=dtype)
    # Load value head weights
    if os.path.exists(v_head_file):
        try:
            v_state = torch.load(v_head_file, map_location=reward_model.device)
            reward_model.v_head.load_state_dict(v_state)
            logger.info("Value head weights loaded.")
        except Exception as e:
            logger.error(f"Error loading value head weights: {e}", exc_info=True)
    else:
        logger.warning("Skipping value head load.")

    # Save merged reward model
    logger.info(f"Saving merged RewardModel to: {output_dir}...")
    try:
        reward_model.save_pretrained(output_dir)
        logger.info("Merged RewardModel saved successfully.")
    except Exception as e:
        logger.error(f"Error saving merged RewardModel: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    merge_reward_adapter(
        base_model_id=config.reward_model,
        lora_adapter_path=LORA_ADAPTER_PATH,
        v_head_filename=V_HEAD_FILENAME,
        output_dir=MERGED_MODEL_OUTPUT_DIR,
        torch_dtype_str=config.dtype
    )
