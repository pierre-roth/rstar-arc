from __future__ import annotations
import argparse
import os
import sys
import yaml
from dataclasses import dataclass, field, asdict, fields
from typing import Optional, Any, Type, Callable, get_type_hints
from enum import Enum

# Define default paths and values
DEFAULT_MODEL_BASE_PATH = "/itet-stor/piroth/net_scratch/models"
DEFAULT_OUTPUT_PATH = "/itet-stor/piroth/net_scratch/outputs"
DEFAULT_DATA_SAMPLE_PATH = "data_sample"
DEFAULT_TRAINING_DATA_PATH = "data_sample/training"
DEFAULT_EVALUATION_DATA_PATH = "data_sample/evaluation"

DEFAULT_POLICY_LLM = "Qwen/Qwen2.5-Coder-7B-Instruct"
DEFAULT_PP_LLM = "Qwen/Qwen2.5-Coder-7B-Instruct"

DEFAULT_MAX_TOKENS = 2048
DEFAULT_MAX_DEPTH = 10
DEFAULT_BEAM_WIDTH = 3
DEFAULT_BRANCHING_FACTOR = 3
DEFAULT_TEMPERATURE = 0.3
DEFAULT_SEED = 42

# Timeouts
TIMEOUT_SECONDS = 15
TIMEOUT_MESSAGE = f"Execution of the code snippet has timed out for exceeding {TIMEOUT_SECONDS} seconds."

# Error messages
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


@dataclass
class Config:
    """Unified configuration class for rStar-ARC"""

    # Application parameters
    verbose: bool = True

    # Model paths and settings
    policy_model: str = DEFAULT_POLICY_LLM
    pp_model: str = DEFAULT_PP_LLM
    model_base_path: str = DEFAULT_MODEL_BASE_PATH
    max_tokens: int = DEFAULT_MAX_TOKENS
    dtype: str = "bfloat16"

    # Generation parameters
    max_depth: int = DEFAULT_MAX_DEPTH

    # Data parameters
    data_folder: str = DEFAULT_TRAINING_DATA_PATH
    task_index: int = 1
    task_name: str = ""
    all_tasks: bool = False

    # Search parameters
    search_mode: str = "bs"
    temperature: float = DEFAULT_TEMPERATURE
    seed: int = DEFAULT_SEED
    deterministic: bool = False

    # BEAM search specific parameters
    beam_width: int = DEFAULT_BEAM_WIDTH
    branching_factor: int = DEFAULT_BRANCHING_FACTOR

    # Hardware parameters
    gpus: int = 1

    # Output parameters
    output_dir: str = DEFAULT_OUTPUT_PATH
    hint: str = ""

    # Config file
    config_file: str = ""

    # SLURM parameters - not used in Python app directly
    mem: Optional[str] = None
    cpus: Optional[int] = None
    partition: Optional[str] = None
    exclude: Optional[str] = None
    nodelist: Optional[str] = None
    constraint: Optional[str] = None
    time: Optional[str] = None

    # Computed fields
    policy_model_dir: Optional[str] = None
    pp_model_dir: Optional[str] = None

    def __post_init__(self):
        """Initialize computed fields after instance creation"""
        # Set computed model directories
        self.policy_model_dir = os.path.join(self.model_base_path, "policy")
        self.pp_model_dir = os.path.join(self.model_base_path, "pp")

    @classmethod
    def from_args(cls, args: Optional[list[str]] = None) -> Config:
        """Create configuration from command line arguments"""
        parser = cls._create_argument_parser()
        parsed_args = parser.parse_args(args)

        # Start with empty config
        config_data = {}

        # First load from config file if provided
        if parsed_args.config_file:
            config_data.update(cls._load_from_file(parsed_args.config_file))

        # Then override with command line arguments (but only if they're explicitly provided)
        for key, value in vars(parsed_args).items():
            if value is not None and key in cls.__annotations__:
                config_data[key] = value

        # Create config instance
        return cls(**config_data)

    @classmethod
    def _create_argument_parser(cls) -> argparse.ArgumentParser:
        """Create argument parser based on Config fields"""
        parser = argparse.ArgumentParser(description='rSTAR meets ARC')

        # Get type hints for all fields
        hints = get_type_hints(cls)

        # For each field in the dataclass
        for field_obj in fields(cls):
            field_name = field_obj.name
            field_def = field_obj

            # Skip private fields and computed fields
            if field_name.startswith('_') or field_name in ['policy_model_dir', 'pp_model_dir']:
                continue

            # Determine CLI flag name (convert snake_case to kebab-case)
            flag_name = f"--{field_name.replace('_', '-')}"

            # Get field type and default
            field_type = hints.get(field_name, Any)
            default_value = field_def.default

            # Handle special field types
            help_text, processed_type = cls._process_field_type(field_type, field_def)

            # Skip SLURM parameters in Python CLI
            if field_name in ['mem', 'cpus', 'partition', 'exclude', 'nodelist', 'time', 'constraint']:
                continue

            # Add argument to parser
            cls._add_argument_to_parser(parser, flag_name, processed_type, default_value, help_text, field_name)

        return parser

    @staticmethod
    def _process_field_type(field_type, field_def):
        """Process field type and generate help text"""
        help_text = field_def.metadata.get('help', '')
        
        # Special case for enum types
        if hasattr(field_type, '__origin__') and field_type.__origin__ is Optional:
            # Handle Optional[Type]
            inner_type = field_type.__args__[0]
            if hasattr(inner_type, '__mro__') and Enum in inner_type.__mro__:
                # Format enum options for help text
                enum_options = [e.value for e in inner_type]
                help_text = f"{help_text} Options: {enum_options}"
                return help_text, str
            else:
                return help_text, inner_type
        elif hasattr(field_type, '__mro__') and Enum in field_type.__mro__:
            # Format enum options for help text
            enum_options = [e.value for e in field_type]
            help_text = f"{help_text} Options: {enum_options}"
            return help_text, str
        else:
            return help_text, field_type

    @staticmethod
    def _add_argument_to_parser(parser, flag_name, field_type, default_value, help_text, field_name):
        """Add the appropriate argument type to the parser"""
        # Boolean fields are flags
        if field_type is bool:
            parser.add_argument(
                flag_name,
                action='store_true',
                default=default_value,
                help=help_text
            )
        else:
            # Regular argument
            parser.add_argument(
                flag_name,
                type=field_type if field_type is not Any else str,
                default=default_value,
                help=help_text,
                required=False
            )

    @staticmethod
    def _load_from_file(config_file: str) -> dict[str, Any]:
        """Load configuration from YAML file"""
        try:
            # Check if file exists
            if not os.path.exists(config_file):
                # Try with config/ prefix
                if os.path.exists(f"config/{config_file}"):
                    config_file = f"config/{config_file}"
                else:
                    print(f"Warning: Config file not found: {config_file}")
                    return {}
                
            with open(config_file, 'r') as f:
                config_data = yaml.safe_load(f) or {}

            # Convert dashed keys to underscore
            return {k.replace('-', '_'): v for k, v in config_data.items()}
        except Exception as e:
            print(f"Error loading config file: {e}")
            return {}

    def list_task_files(self) -> list[str]:
        """List all JSON files in the data folder"""
        if not os.path.isdir(self.data_folder):
            print(f"Error: Directory '{self.data_folder}' not found. Please check your ARC data directory.")
            sys.exit(1)

        files = sorted([f for f in os.listdir(self.data_folder) if f.endswith('.json')],
                       key=lambda x: x.lower())

        if not files:
            print(f"No JSON files found in directory '{self.data_folder}'.")
            sys.exit(1)

        if self.verbose:
            print(f"Found {len(files)} JSON files in '{self.data_folder}'")
        return files

    def select_task_file(self) -> str:
        """Select a task file based on configuration"""
        # If task_name is provided, use it
        if self.task_name:
            task_file = f"{self.task_name}.json"
            task_path = os.path.join(self.data_folder, task_file)

            if not os.path.exists(task_path):
                print(f"Error: Task file '{task_path}' not found.")
                sys.exit(1)

            if self.verbose:
                print(f"Using task file: {task_path}")
            return task_path

        # Otherwise use task_index
        files = self.list_task_files()

        if self.task_index < 1 or self.task_index > len(files):
            print(f"Error: Task index {self.task_index} is out of range (1-{len(files)})")
            sys.exit(1)

        chosen_file = os.path.join(self.data_folder, files[self.task_index - 1])
        if self.verbose:
            print(f"Selected file by index: {chosen_file}")

        return chosen_file

