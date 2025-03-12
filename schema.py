import os
from typing import Any, Dict, List, Callable, Optional, Union, Literal
from dataclasses import dataclass, field


@dataclass
class ParamSchema:
    """Schema definition for a single parameter"""
    name: str
    cli_flag: str
    bash_flag: str
    default: Any
    type: type
    help: str
    is_required: bool = False
    is_flag: bool = False
    validation: Optional[Callable[[Any], bool]] = None


# Define constants used throughout the application
TIMEOUT_SECONDS = 15
TIMEOUT_MESSAGE = f"Execution of the code snippet has timed out for exceeding {TIMEOUT_SECONDS} seconds."
ERROR_COLOR = "red"
TOO_MANY_CODE_ERRORS = "Too many consecutive steps have code errors."
TOO_MANY_STEPS = "Fail to solve the problem within limited steps."
NO_VALID_CHILD = "Fail to generate parsable text for next step."

# Output format tags
OUTPUT = "<o>"
OUTPUT_END = "<end_of_output>"
STEP_END = "<end_of_step>"
CODE = "<code>"
CODE_TAG = "Now print the final answer"
CODE_END = "<end_of_code>"
ANSWER = "<answer>"
ANSWER_END = "<end_of_answer>"
REFINE = "<refine>"
REFINE_PASS = "I am sure that my answer is correct"
REFINE_END = "<end_of_refine>"

# Environment-aware base paths
OUTPUT_BASE_PATH = f"/itet-stor/piroth/net_scratch/outputs"
MODEL_BASE_PATH = f"/itet-stor/piroth/net_scratch/models"
DATA_BASE_PATH = f"/itet-stor/piroth/net_scratch/data"

# Default paths for data
DATA_SAMPLE_BASE_PATH = "data_sample"
DEFAULT_TRAINING_DATA_PATH = "data_sample/training"
DEFAULT_EVALUATION_DATA_PATH = "data_sample/evaluation"

# Default model configurations
DEFAULT_POLICY_LLM = "Qwen/Qwen2.5-Coder-7B-Instruct"
DEFAULT_PP_LLM = "Qwen/Qwen2.5-Coder-7B-Instruct"
DEFAULT_MAX_TOKENS = 2048

# Default runtime configurations
DEFAULT_VERBOSE = True
DEFAULT_MAX_ITERATIONS = 5
DEFAULT_MAX_DEPTH = 10
DEFAULT_SEARCH_MODE = "beam_search"
DEFAULT_TEMPERATURE = 0.3
DEFAULT_SEED = 42
DEFAULT_DTYPE = "float16"

# Beam search specific configurations
DEFAULT_BEAM_WIDTH = 3
DEFAULT_BRANCHING_FACTOR = 3

# Parameter schema definition
PARAM_SCHEMA = [
    # SLURM parameters (cluster-specific)
    ParamSchema(
        name="mem", 
        cli_flag="--mem", 
        bash_flag="--mem=", 
        default="32G",
        type=str, 
        help="Memory to allocate for the SLURM job",
        is_required=False
    ),
    ParamSchema(
        name="cpus", 
        cli_flag="--cpus", 
        bash_flag="--cpus=", 
        default=4,
        type=int, 
        help="Number of CPUs to allocate for the SLURM job",
    ),
    ParamSchema(
        name="gpus", 
        cli_flag="--gpus", 
        bash_flag="--gpus=", 
        default=1,
        type=int, 
        help="Number of GPUs to use for the LLM",
    ),
    ParamSchema(
        name="partition", 
        cli_flag="--partition", 
        bash_flag="--partition=", 
        default="",
        type=str, 
        help="SLURM partition to use",
    ),
    ParamSchema(
        name="exclude", 
        cli_flag="--exclude", 
        bash_flag="--exclude=", 
        default="",
        type=str, 
        help="SLURM nodes to exclude",
    ),
    ParamSchema(
        name="nodelist", 
        cli_flag="--nodelist", 
        bash_flag="--nodelist=", 
        default="",
        type=str, 
        help="SLURM nodes to include",
    ),
    ParamSchema(
        name="time", 
        cli_flag="--time", 
        bash_flag="--time=", 
        default="",
        type=str, 
        help="Time limit for the SLURM job (format: HH:MM:SS)",
    ),
    
    # Application parameters
    ParamSchema(
        name="task_index", 
        cli_flag="--task-index", 
        bash_flag="--task=", 
        default=1,
        type=int, 
        help="Index of the task to use",
        validation=lambda x: x > 0
    ),
    ParamSchema(
        name="task_name", 
        cli_flag="--task-name", 
        bash_flag="--task-name=", 
        default="",
        type=str, 
        help="Specific task name to use (overrides task-index)",
    ),
    ParamSchema(
        name="max_iterations", 
        cli_flag="--max-iterations", 
        bash_flag="--iter=", 
        default=DEFAULT_MAX_ITERATIONS,
        type=int, 
        help=f"Maximum number of iterations allowed (default: {DEFAULT_MAX_ITERATIONS})",
        validation=lambda x: x > 0
    ),
    ParamSchema(
        name="policy_model", 
        cli_flag="--policy-model", 
        bash_flag="--policy-model=", 
        default=DEFAULT_POLICY_LLM,
        type=str, 
        help=f"LLM model to use for step candidate generation (default: {DEFAULT_POLICY_LLM})",
    ),
    ParamSchema(
        name="pp_model", 
        cli_flag="--pp-model", 
        bash_flag="--pp-model=", 
        default=DEFAULT_PP_LLM,
        type=str, 
        help=f"LLM model to use for selecting the most promising candidate steps (default: {DEFAULT_PP_LLM})",
    ),
    ParamSchema(
        name="hint", 
        cli_flag="--hint", 
        bash_flag="--hint=", 
        default="",
        type=str, 
        help="Hint to provide to the model",
    ),
    ParamSchema(
        name="verbose", 
        cli_flag="--verbose", 
        bash_flag="--verbose", 
        default=DEFAULT_VERBOSE,
        type=bool, 
        help=f"Print detailed progress information (default: {DEFAULT_VERBOSE})",
        is_flag=True
    ),
    ParamSchema(
        name="dtype", 
        cli_flag="--dtype", 
        bash_flag="--dtype=", 
        default=DEFAULT_DTYPE,
        type=str, 
        help="Data type for model (float16, bfloat16) - use float16 for older GPUs",
    ),
    ParamSchema(
        name="output_dir", 
        cli_flag="--output-dir", 
        bash_flag="--output-dir=", 
        default=OUTPUT_BASE_PATH,
        type=str, 
        help=f"Directory to store output files (default: {OUTPUT_BASE_PATH})",
    ),
    ParamSchema(
        name="search_mode", 
        cli_flag="--search-mode", 
        bash_flag="--search-mode=", 
        default=DEFAULT_SEARCH_MODE,
        type=str, 
        help=f"Search mode for inference (default: {DEFAULT_SEARCH_MODE})",
    ),
    ParamSchema(
        name="max_depth", 
        cli_flag="--max-depth", 
        bash_flag="--max-depth=", 
        default=DEFAULT_MAX_DEPTH,
        type=int, 
        help=f"Maximum number of depth iterations allowed (default: {DEFAULT_MAX_DEPTH})",
        validation=lambda x: x > 0
    ),
    ParamSchema(
        name="temperature", 
        cli_flag="--temperature", 
        bash_flag="--temperature=", 
        default=DEFAULT_TEMPERATURE,
        type=float, 
        help=f"Temperature for LLM sampling (default: {DEFAULT_TEMPERATURE})",
        validation=lambda x: 0 <= x <= 1.0
    ),
    ParamSchema(
        name="seed", 
        cli_flag="--seed", 
        bash_flag="--seed=", 
        default=DEFAULT_SEED,
        type=int, 
        help=f"Seed for reproducibility (default: {DEFAULT_SEED})",
    ),
    ParamSchema(
        name="all_tasks", 
        cli_flag="--all-tasks", 
        bash_flag="--all-tasks", 
        default=False,
        type=bool, 
        help="Process all tasks in the data_sample/[training or evaluation] directory",
        is_flag=True
    ),
    ParamSchema(
        name="config_file", 
        cli_flag="--config-file", 
        bash_flag="--config-file=", 
        default="",
        type=str, 
        help="Path to configuration file",
    ),
    ParamSchema(
        name="max_tokens", 
        cli_flag="--max-tokens", 
        bash_flag="--max-tokens=", 
        default=DEFAULT_MAX_TOKENS,
        type=int, 
        help=f"Maximum number of tokens per step for policy LLM (default: {DEFAULT_MAX_TOKENS})",
        validation=lambda x: x > 0
    ),
    ParamSchema(
        name="data_folder", 
        cli_flag="--data-folder", 
        bash_flag="--data-folder=", 
        default=DEFAULT_TRAINING_DATA_PATH,
        type=str, 
        help=f"Path to the folder containing task JSON files (default: {DEFAULT_TRAINING_DATA_PATH})",
    ),
    ParamSchema(
        name="deterministic", 
        cli_flag="--deterministic", 
        bash_flag="--deterministic", 
        default=False,
        type=bool, 
        help="Set seed everywhere for deterministic reproducibility",
        is_flag=True
    ),
    ParamSchema(
        name="beam_width", 
        cli_flag="--beam-width", 
        bash_flag="--beam-width=", 
        default=DEFAULT_BEAM_WIDTH,
        type=int, 
        help=f"Width of the beam for beam search (default: {DEFAULT_BEAM_WIDTH})",
        validation=lambda x: x > 0
    ),
    ParamSchema(
        name="branching_factor", 
        cli_flag="--branching-factor", 
        bash_flag="--branching-factor=", 
        default=DEFAULT_BRANCHING_FACTOR,
        type=int, 
        help=f"Number of candidates to generate at each step (default: {DEFAULT_BRANCHING_FACTOR})",
        validation=lambda x: x > 0
    ),
]

# Create parameter lookup by name
PARAM_BY_NAME = {param.name: param for param in PARAM_SCHEMA}

