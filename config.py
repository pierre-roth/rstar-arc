from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass, fields
from typing import Optional, Any, get_type_hints

import yaml

#############################################
# CONSTANTS AND DEFAULT CONFIGURATION VALUES
#############################################

ETH_USERNAME = "piroth"

# Default paths for models, outputs and data
# These paths are used if not overridden by command line or config file
NET_SCRATCH_PATH = f"/itet-stor/{ETH_USERNAME}/net_scratch"  # network scratch directory
LOCAL_SCRATCH_PATH = f"/scratch/{ETH_USERNAME}"  # local scratch directory

DEFAULT_DATA_SAMPLE_PATH = "data_sample"  # Root folder for ARC data
DEFAULT_TRAINING_DATA_PATH = "data_sample/training"  # Training data location
DEFAULT_EVALUATION_DATA_PATH = "data_sample/evaluation"  # Evaluation data location

# Default model names - identifies which language models to use
DEFAULT_POLICY_LLM = "Qwen/Qwen2.5-Coder-7B-Instruct"  # Policy model (generates reasoning steps)
DEFAULT_REWARD_LLM = "Qwen/Qwen2.5-Coder-7B-Instruct"  # Reward Model (evaluates steps)

# Default hyperparameters
DEFAULT_MAX_TOKENS = 1024  # Maximum tokens for model generation
DEFAULT_MAX_DEPTH = 10  # Maximum depth of search tree (max steps)
DEFAULT_BEAM_WIDTH = 3  # Width of beam in beam search (solutions to track)
DEFAULT_BRANCHING_FACTOR = 3  # Number of child nodes to expand per parent
DEFAULT_POLICY_TEMPERATURE = 0.7  # Sampling temperature for policy model
DEFAULT_SEED = 42  # Random seed for reproducibility
DEFAULT_C_PUCT = 2.0  # PUCT exploration constant for MCTS

###########################################
# EXECUTION SETTINGS
###########################################

# Code execution timeout settings
TIMEOUT_SECONDS = 15  # Maximum time allowed for code execution
MEMORY_LIMIT_MB = 256
MEMORY_LIMIT_BYTES = MEMORY_LIMIT_MB * 1024 * 1024

# Terminal node constants
TERMINAL_SUCCESS = "Successful solution"
TERMINAL_FAILURE = "Failed solution"
TERMINAL_INVALID = "Invalid code"
TERMINAL_MAX_DEPTH = "Maximum depth reached"
TERMINAL_CODE_END = "Code end marker"

###########################################
# OUTPUT FORMAT MARKERS
###########################################
# These tags are used to parse the model's output into structured sections

# Step output markers
STEP_END = "<end_of_step>"  # Marks the end of a reasoning step

# Code section markers
CODE = "<code>"  # Begins a code section
CODE_END = "<end_of_code>"  # Ends a code section


@dataclass
class Config:
    """
    Unified configuration class for rStar-ARC
    
    This class handles all configuration aspects of the rStar-ARC system:
    - Loading config from YAML files and command-line arguments
    - Providing default values
    - Processing and validating configuration options
    """

    ###########################################
    # APPLICATION PARAMETERS
    ###########################################
    verbose: bool = True  # Controls logging verbosity

    ###########################################
    # PROMPT PARAMETERS
    ###########################################
    num_examples: int = 2  # Controls the number of few shot examples in the prompt

    ###########################################
    # MODEL CONFIGURATION
    ###########################################
    # Language model selection
    policy_model: str = DEFAULT_POLICY_LLM  # Model that generates reasoning steps
    reward_model: str = DEFAULT_REWARD_LLM  # Reward Model for evaluating steps

    # general model configuration
    model_base_path: str = os.path.join(NET_SCRATCH_PATH, "models")  # Base path where models are stored

    # Policy model configuration
    max_tokens: int = DEFAULT_MAX_TOKENS  # Maximum tokens for generation
    dtype: str = "bfloat16"  # Data type for model (affects precision/speed)
    max_model_len: int = 16384  # Affects the context window size
    top_p: float = 0.95  # Top-p sampling parameter (cumulative probability cutoff)
    top_k: int = -1  # Top-k sampling parameter (number of candidates to consider)
    policy_temperature: float = DEFAULT_POLICY_TEMPERATURE  # Sampling temperature for LLM generation
    seed: int = DEFAULT_SEED  # Random seed for reproducibility
    deterministic: bool = False  # Whether to enforce deterministic behavior

    # Reward model configuration

    ###########################################
    # GENERATION PARAMETERS
    ###########################################
    max_depth: int = DEFAULT_MAX_DEPTH  # Maximum number of reasoning steps

    ###########################################
    # DATA CONFIGURATION
    ###########################################
    data_folder: str = DEFAULT_TRAINING_DATA_PATH  # Path to ARC task data
    task_index: int = 1  # Index of task to run (1-based indexing)
    task_name: str = ""  # Name of specific task (overrides task_index if provided)
    all_tasks: bool = False  # Whether to run all tasks in data_folder

    ###########################################
    # SEARCH ALGORITHM PARAMETERS
    ###########################################
    search_mode: str = "bs"  # Search algorithm - "bs" for beam search, "mcts" for Monte Carlo Tree Search

    # BEAM search specific parameters
    beam_width: int = DEFAULT_BEAM_WIDTH  # Number of top-scoring beams to track
    branching_factor: int = DEFAULT_BRANCHING_FACTOR  # Number of children to generate per step

    # MCTS search specific parameters
    c_puct: float = DEFAULT_C_PUCT  # PUCT exploration constant
    num_simulations: int = 8  # Number of simulations to run for MCTS

    ###########################################
    # SLURM AND HARDWARE CONFIGURATION
    ###########################################
    gpus: int = int(os.getenv("SLURM_GPUS", 1))  # Number of GPUs available (read from environment variable)
    job_id: int = int(os.getenv("SLURM_JOB_ID", 0))  # N

    ###########################################
    # OUTPUT CONFIGURATION
    ###########################################
    output_dir: str = os.path.join(NET_SCRATCH_PATH, "outputs")  # Directory to save results

    ###########################################
    # CONFIG FILE
    ###########################################
    config_file: str = ""  # Path to YAML config file

    ###########################################
    # SLURM PARAMETERS
    ###########################################
    # These parameters are used by the SLURM script, not directly by the Python application
    mem: Optional[str] = None  # Memory allocation for SLURM
    cpus: Optional[int] = None  # Number of CPUs for SLURM
    partition: Optional[str] = None  # SLURM partition to use
    exclude: Optional[str] = None  # Nodes to exclude
    nodelist: Optional[str] = None  # Specific nodes to use
    constraint: Optional[str] = None  # Hardware constraints
    time: Optional[str] = None  # Time limit for job

    ###########################################
    # COMPUTED FIELDS
    ###########################################
    # These are calculated automatically in __post_init__
    policy_model_dir: Optional[str] = None  # Full path to policy model
    reward_model_dir: Optional[str] = None  # Full path to reward model

    def __post_init__(self):
        """
        Initialize computed fields after instance creation
        
        This method runs automatically after the dataclass is instantiated,
        calculating derived values based on the provided configuration.
        """
        # Set computed model directories based on model_base_path
        self.policy_model_dir = os.path.join(self.model_base_path, "policy")
        self.reward_model_dir = os.path.join(self.model_base_path, "reward")
        self.temporary_path = os.path.join(LOCAL_SCRATCH_PATH, f"log_{self.job_id}")

    @classmethod
    def from_args(cls, args: Optional[list[str]] = None) -> Config:
        """
        Create a configuration from command line arguments
        
        This method:
        1. Parses command line arguments
        2. Loads configuration from YAML file (if specified)
        3. Overrides file settings with command line arguments
        
        Args:
            args: Command line arguments to parse (uses sys.argv if None)
            
        Returns:
            Configured Config instance
        """
        # Create and use argument parser
        parser = cls._create_argument_parser()
        parsed_args = parser.parse_args(args)
        args_dict = vars(parsed_args)

        # Start with defaults from the dataclass
        config_data = {field.name: field.default for field in fields(cls)
                       if field.name not in ['policy_model_dir', 'reward_model_dir']}

        # First load from config file if provided (overrides defaults)
        if args_dict.get("config_file"):
            yaml_config = cls._load_from_file(args_dict["config_file"])
            print(f"DEBUG - YAML config loaded: {yaml_config}")
            config_data.update(yaml_config)

        # Get a list of arguments that were explicitly provided on the command line
        provided_args = set()
        if args:
            for i, arg in enumerate(args):
                if arg.startswith('--'):
                    arg_name = arg[2:].replace('-', '_')  # Convert from kebab-case to snake_case
                    provided_args.add(arg_name)

        # Then override with command line arguments (highest priority), but only if explicitly provided
        for key, value in args_dict.items():
            if key == "config_file":
                continue  # Skip config_file, we've already processed it

            # Only include values from arguments that were explicitly provided
            if key in provided_args and key in cls.__annotations__:
                print(f"DEBUG - Setting from CLI args: {key}={value}")
                config_data[key] = value

        print(f"DEBUG - Final config:")
        for key in sorted(config_data.keys()):
            print(f"  {key}: {config_data[key]}")

        # Create and return config instance with the merged settings
        return cls(**config_data)

    @classmethod
    def _create_argument_parser(cls) -> argparse.ArgumentParser:
        """
        Create an argument parser with all configuration options
        
        This method dynamically builds a command-line parser based on the
        Config class fields, converting them to appropriate CLI arguments.
        
        Returns:
            Configured ArgumentParser instance
        """
        parser = argparse.ArgumentParser(description='rSTAR meets ARC')

        # Get type information for all fields using Python's typing system
        hints = get_type_hints(cls)

        # Process each field in the dataclass to create corresponding CLI arguments
        for field_obj in fields(cls):
            field_name = field_obj.name
            field_def = field_obj

            # Skip private fields and computed fields
            if (field_name.startswith('_') or
                    field_name in ['policy_model_dir', 'reward_model_dir']):
                continue

            # Convert snake_case field names to kebab-case for CLI flags
            flag_name = f"--{field_name.replace('_', '-')}"

            # Get the field's type and default value
            field_type = hints.get(field_name, Any)
            help_text = field_def.metadata.get('help', '')

            # Handle Optional types by extracting the inner type
            if hasattr(field_type, '__origin__') and field_type.__origin__ is Optional:
                field_type = field_type.__args__[0]

            # Add argument to parser based on its type
            if field_type is bool:
                # For boolean args, add both --flag and --no-flag options
                # This makes the CLI intent explicit for boolean values
                group = parser.add_mutually_exclusive_group()

                # The --flag option sets the value to True
                group.add_argument(
                    flag_name,
                    action='store_true',
                    dest=field_name,
                    help=f"{help_text} (enable)"
                )

                # The --no-flag option sets the value to False
                group.add_argument(
                    f"--no-{field_name.replace('_', '-')}",
                    action='store_false',
                    dest=field_name,
                    help=f"{help_text} (disable)"
                )
            else:
                # All other argument types
                parser.add_argument(
                    flag_name,
                    # Use the field's type, defaulting to str for Any or unknown types
                    type=field_type if field_type is not Any else str,
                    dest=field_name,
                    default=None,  # This is fine - we're checking if the arg was provided in from_args
                    help=help_text,
                    required=False
                )

        return parser

    @staticmethod
    def _load_from_file(config_file: str) -> dict[str, Any]:
        """
        Load configuration settings from a YAML file
        
        This method:
        1. Checks if the file exists (in current directory or config/ subfolder)
        2. Loads and parses the YAML content
        3. Converts kebab-case keys to snake_case for Python compatibility
        
        Args:
            config_file: Path to the YAML configuration file
            
        Returns:
            Dictionary of configuration values
        """
        try:
            # Check if file exists at the specified path
            if not os.path.exists(config_file):
                # If not, try looking in the config/ directory
                if os.path.exists(f"config/{config_file}"):
                    config_file = f"config/{config_file}"
                else:
                    print(f"Warning: Config file not found: {config_file}")
                    return {}

            # Read and parse the YAML file
            with open(config_file, 'r') as f:
                config_data = yaml.safe_load(f) or {}

            # Convert kebab-case keys (like "model-name") to snake_case (like "model_name")
            # This ensures compatibility with Python variable naming conventions
            result = {k.replace('-', '_'): v for k, v in config_data.items()}

            # Debug the loaded configuration
            print(f"DEBUG - Loaded from {config_file}:")
            for k, v in result.items():
                print(f"  {k}: {v}")

            return result

        except Exception as e:
            print(f"Error loading config file: {e}")
            return {}

    def list_task_files(self) -> list[str]:
        """
        List all JSON task files in the configured data folder
        
        This method:
        1. Verifies the data directory exists
        2. Finds all JSON files in the directory
        3. Sorts them alphabetically (case-insensitive)
        
        Returns:
            List of JSON filenames (without directory path)
            
        Raises:
            SystemExit: If directory doesn't exist or no JSON files are found
        """
        # Verify the data directory exists
        if not os.path.isdir(self.data_folder):
            print(f"Error: Directory '{self.data_folder}' not found. Please check your ARC data directory.")
            sys.exit(1)

        # Get all JSON files and sort them alphabetically (case-insensitive)
        files = sorted([f for f in os.listdir(self.data_folder) if f.endswith('.json')],
                       key=lambda x: x.lower())

        # Ensure we found at least one file
        if not files:
            print(f"No JSON files found in directory '{self.data_folder}'.")
            sys.exit(1)

        # Report number of files if in verbose mode
        if self.verbose:
            print(f"Found {len(files)} JSON files in '{self.data_folder}'")

        return files

    def select_task_file(self) -> str:
        """
        Select a specific ARC task file based on configuration
        
        This method determines which task file to use based on:
        1. First, specific task name if provided (task_name)
        2. Otherwise, task index if provided (task_index)
        
        Returns:
            Full path to the selected task file
            
        Raises:
            SystemExit: If the requested task file doesn't exist
        """
        # Debug info
        print(f"DEBUG - In select_task_file: task_name='{self.task_name}', task_index={self.task_index}")

        # PRIORITY 1: If a specific task name is provided, use it directly
        if self.task_name and self.task_name.strip():
            # Convert task name to filename with .json extension
            task_file = f"{self.task_name}.json"
            task_path = os.path.join(self.data_folder, task_file)

            # Verify the file exists
            if not os.path.exists(task_path):
                print(f"Error: Task file '{task_path}' not found.")
                sys.exit(1)

            # Report selected file if in verbose mode
            if self.verbose:
                print(f"Using task file: {task_path}")
            return task_path

        # PRIORITY 2: Use task_index to select from available files
        files = self.list_task_files()

        # Verify index is in valid range (1-based indexing)
        if self.task_index < 1 or self.task_index > len(files):
            print(f"Error: Task index {self.task_index} is out of range (1-{len(files)})")
            sys.exit(1)

        # Select the file by index (adjusting for 0-based list indexing)
        chosen_file = os.path.join(self.data_folder, files[self.task_index - 1])

        # Report selected file if in verbose mode
        if self.verbose:
            print(f"Selected file by index: {chosen_file}")

        return chosen_file
