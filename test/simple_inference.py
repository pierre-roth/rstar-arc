import logging
import os
import sys

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    set_seed,
)

# ------------------- project imports -------------------
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from constants import (
    SFT_SYSTEM_PROMPT,
    SFT_IN_BETWEEN_PROMPT,
)
from utils import setup_logging
from rstar_deepthink import Config
from rstar_deepthink.arc_task import ARCTask
from rstar_deepthink.arc_task.task_utils import load_tasks, task_to_prompt

logger = logging.getLogger(__name__)

# ------------------- config -------------------
config = Config()
set_seed(config.seed or 42)
setup_logging(config.numeric_log_level)

MODEL = os.path.join(config.policy_model_dir, config.policy_model)

prompts = [SFT_SYSTEM_PROMPT + task_to_prompt(task) + SFT_IN_BETWEEN_PROMPT for task in load_tasks(config)]

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16)

# Set device
device = "mps"
model.to(device)
logger.info(f"Using device: {device}")

# Prepare the input

inputs = tokenizer(prompts, return_tensors="pt").to(device)

# Generate completions
logger.info("Starting inference...")

outputs = model.generate(
    **inputs,
    max_new_tokens=2048,
    do_sample=True,
    temperature=0.8,
    top_p=1,
    num_return_sequences=1,
    eos_token_id=tokenizer.eos_token_id,
    pad_token_id=tokenizer.eos_token_id
)

for i, output in enumerate(outputs):
    text = tokenizer.decode(output, skip_special_tokens=True)

    # extract everything between CODE and CODE_END
    start = text.find("<beginning_of_code>")
    end = text.find("<end_of_code>")

    code = text[start:end + len("<end_of_code>")].strip()

    print(f"#{i + 1}: {code}")

logger.info("Inference complete.")
