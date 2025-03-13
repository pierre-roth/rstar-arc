from __future__ import annotations
import argparse
import os
import sys
import yaml
from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict, Any, Type, Callable, get_type_hints
from enum import Enum, auto

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


class SearchMode(Enum):
    """Enum for search modes"""
    BEAM_SEARCH = "beam_search"
    MCTS = "mcts"


class DataType(Enum):
    """Enum for model data types"""
    FLOAT16 = "float16"
    BFLOAT16 = "bfloat16"


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
    dtype: DataType = DataType.FLOAT16

    # Generation parameters
    max_depth: int = DEFAULT_MAX_DEPTH

    # Data parameters
    data_folder: str = DEFAULT_TRAINING_DATA_PATH
    task_index: int = 1
    task_name: str = ""
    all_tasks: bool = False

    # Search parameters
    search_mode: SearchMode = SearchMode.BEAM_SEARCH
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
    time: Optional[str] = None

    # Computed fields
    policy_model_dir: Optional[str] = None
    pp_model_dir: Optional[str] = None

    def __post_init__(self):
        """Initialize computed fields after instance creation"""
        self.policy_model_dir = os.path.join(self.model_base_path, "policy")
        self.pp_model_dir = os.path.join(self.model_base_path, "pp")

        # Handle enum values
        if isinstance(self.search_mode, str):
            try:
                self.search_mode = SearchMode(self.search_mode)
            except ValueError:
                raise ValueError(f"Invalid search mode: {self.search_mode}. "
                                 f"Valid options are: {[m.value for m in SearchMode]}")

        if isinstance(self.dtype, str):
            try:
                self.dtype = DataType(self.dtype)
            except ValueError:
                raise ValueError(f"Invalid dtype: {self.dtype}. "
                                 f"Valid options are: {[d.value for d in DataType]}")

        # Validate numeric fields
        if self.task_index < 1:
            raise ValueError("task_index must be greater than 0")

        if self.max_depth < 1:
            raise ValueError("max_depth must be greater than 0")

        if self.max_tokens < 1:
            raise ValueError("max_tokens must be greater than 0")

        if self.beam_width < 1:
            raise ValueError("beam_width must be greater than 0")

        if self.branching_factor < 1:
            raise ValueError("branching_factor must be greater than 0")

        if not 0 <= self.temperature <= 1.0:
            raise ValueError("temperature must be between 0 and 1")

    @classmethod
    def from_args(cls, args: Optional[List[str]] = None) -> Config:
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
        for field_name, field_def in cls.__dataclass_fields__.items():
            # Skip private fields and computed fields
            if field_name.startswith('_') or field_name in ['policy_model_dir', 'pp_model_dir']:
                continue

            # Determine CLI flag name (convert snake_case to kebab-case)
            flag_name = f"--{field_name.replace('_', '-')}"

            # Get field type and default
            field_type = hints.get(field_name, Any)
            default_value = field_def.default

            # Special case for enum types
            if hasattr(field_type, '__origin__') and field_type.__origin__ is Optional:
                # Handle Optional[Type]
                inner_type = field_type.__args__[0]
                if issubclass(inner_type, Enum):
                    field_type = str
                    help_text = f"{field_def.metadata.get('help', '')} Options: {[e.value for e in inner_type]}"
                else:
                    field_type = inner_type
                    help_text = field_def.metadata.get('help', '')
            elif issubclass(field_type, Enum):
                field_type = str
                help_text = f"{field_def.metadata.get('help', '')} Options: {[e.value for e in field_type]}"
            else:
                help_text = field_def.metadata.get('help', '')

            # Boolean fields are flags
            if field_type is bool:
                parser.add_argument(
                    flag_name,
                    action='store_true',
                    default=default_value,
                    help=help_text
                )
            else:
                # Skip SLURM parameters in Python CLI
                if field_name in ['mem', 'cpus', 'partition', 'exclude', 'nodelist', 'time']:
                    continue

                # Regular argument
                parser.add_argument(
                    flag_name,
                    type=field_type if field_type is not Any else str,
                    default=default_value,
                    help=help_text,
                    required=False
                )

        return parser

    @staticmethod
    def _load_from_file(config_file: str) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        try:
            with open(config_file, 'r') as f:
                config_data = yaml.safe_load(f) or {}

            # Convert dashed keys to underscore
            return {k.replace('-', '_'): v for k, v in config_data.items()}
        except Exception as e:
            print(f"Error loading config file: {e}")
            return {}

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary, handling enum values"""
        data = {}
        for key, value in asdict(self).items():
            if isinstance(value, Enum):
                data[key] = value.value
            else:
                data[key] = value
        return data

    def save_to_file(self, file_path: str) -> None:
        """Save configuration to a YAML file"""
        try:
            # Convert underscores back to dashes for YAML
            data = {k.replace('_', '-'): v for k, v in self.to_dict().items()}

            with open(file_path, 'w') as f:
                yaml.dump(data, f, default_flow_style=False)
            print(f"Config saved to {file_path}")
        except Exception as e:
            print(f"Error saving config: {e}")

    def list_task_files(self) -> List[str]:
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

    @staticmethod
    def print_help() -> None:
        """Print help information with example usage"""
        print("\nrSTAR-ARC: Self-play muTuAl Reasoning for ARC\n")
        print("This program applies the rStar methodology to solve ARC (Abstraction and Reasoning Corpus) tasks.\n")

        print("Usage examples:")
        print("  Local run:  python main.py --task-index=1 --verbose")
        print("  Cluster run: ./run.sh --task=1 --gpus=1 --dtype=bfloat16 --verbose\n")

        print("Configuration:")
        print("  You can specify parameters via:")
        print("  1. Command line arguments")
        print("  2. Config file (--config-file=config/basic_bs.yaml)")
        print("  3. Default values\n")

        # Generate parameter help dynamically
        print("Available parameters:")
        print("-" * 60)

        # Get the parser to extract help text
        parser = Config._create_argument_parser()

        # Group parameters by category
        categories = {
            "SLURM Parameters": ['mem', 'cpus', 'partition', 'exclude', 'nodelist', 'time'],
            "Model Parameters": ['policy-model', 'pp-model', 'max-tokens', 'model-base-path', 'dtype'],
            "Task Parameters": ['task-index', 'task-name', 'all-tasks', 'data-folder'],
            "Search Parameters": ['search-mode', 'max-depth', 'beam-width', 'branching-factor',
                                  'temperature', 'deterministic'],
            "Output Parameters": ['output-dir', 'verbose', 'hint'],
            "System Parameters": ['gpus', 'seed'],
            "Config Parameters": ['config-file']
        }

        for category, param_names in categories.items():
            print(f"\n{category}:")
            for action in parser._actions:
                # Skip the help action
                if action.dest == 'help':
                    continue

                # Check if this action belongs to the current category
                param_name = action.dest.replace('_', '-')
                if param_name in param_names:
                    default_str = f" (default: {action.default})" if action.default is not None else ""
                    print(f"  {action.option_strings[0]:<20} {action.help}{default_str}")

        print("\nFor more information, check the README.md file.")
