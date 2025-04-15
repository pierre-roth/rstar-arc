import argparse
import logging
import os
from dataclasses import dataclass, field
from typing import Optional
from random import choices, shuffle, sample

import yaml

from constants import *


@dataclass
class Config:
    """
    Unified configuration class for rStar-ARC

    This class handles all configuration aspects of the rStar-ARC system,
    loading configuration from a YAML file and providing default values.
    """

    arc_prize: bool = False  # Whether this is a competition submission

    log_level: str = "INFO"  # Logging level: DEBUG, INFO, WARNING, ERROR, CRITICAL
    save_for_visualization: bool = True  # Whether to visualize the reasoning steps
    solve_only_unsolved: bool = False  # Whether to only solve unsolved tasks
    solutions_per_task: int | None = None  # Bound the number of solutions (that pass training examples) to find for each task (None means as many as possible)
    save_sft_data: bool = True  # Whether to save SFT data
    evaluation: bool = False  # Whether the model is running on the evaluation set

    numeric_log_level: Optional[int] = None  # Numeric logging level (set automatically)
    model_initialization_times = {"policy": None, "reward": None}  # Time taken to initialize models

    num_examples: int = -1  # Number of examples to use for training (default: all)
    example_names: list[list[str]] = field(default_factory=lambda: [["6d0aefbc", "1cf80156"], ["1cf80156",
                                                                                               "00d62c1b"]])  # list of names of example tasks (to be used sequentially in different rollouts)
    rotate_example: bool = False  # Whether to rotate the example tasks in each rollout

    policy_model: str = "Qwen/Qwen2.5-Coder-7B-Instruct"  # Model that generates reasoning steps
    reward_model: str = "Qwen/Qwen2.5-Coder-7B-Instruct"  # Reward Model for evaluating steps
    # policy_model: str = "Qwen/Qwen2.5-Coder-7B"  # Model that generates reasoning steps
    # reward_model: str = "Qwen/Qwen2.5-Coder-7B"  # Reward Model for evaluating steps
    enforce_eager: bool = False

    fine_tuned: bool = False  # Whether the model is fine-tuned

    net_scratch_model_base_path: str = os.path.join(NET_SCRATCH_PATH, "models")  # Base path where models are stored
    local_scratch_model_base_path: str = os.path.join(LOCAL_SCRATCH_PATH, "models")  # Base path where models are stored

    max_tokens: int = 512  # Maximum tokens for generation of a single step
    dtype: str = "bfloat16"  # Data type for model (affects precision/speed)
    max_model_len: int = 16384  # Affects the context window size
    # max_num_seqs: int = 1024  # Maximum number of sequences to generate in parallel
    # max_num_batched_tokens = 16384  # Maximum number of tokens to process in a batch

    top_p: float = 0.95  # Top-p sampling parameter (cumulative probability cutoff)
    top_k: int = -1  # Top-k sampling parameter (number of candidates to consider)
    policy_temperature: float = 0.7  # Sampling temperature for LLM generation
    seed: int = 42  # Random seed for reproducibility
    deterministic: bool = False  # Whether to enforce deterministic behavior

    num_tasks: int = -1  # Number of tasks to process (-1 means all tasks)
    data_folder: str = DEFAULT_DATA_PATH  # Path to ARC task data
    task_names: list[str] | None = None  # List of task names to process
    sort_by_length = False  # Whether to sort tasks by length

    search_mode: str = "bs"  # Search algorithm - "bs" for beam search, "mcts" for Monte Carlo Tree Search

    max_depth: int = 10  # Maximum number of reasoning steps
    batch_size: int = -1  # Batch size for parallel inference (-1 means all at once, otherwise batch size)

    beam_width: int = 3  # Number of top-scoring beams to track
    branching_factor: int = 4  # Number of children to generate if no children exist yet
    regeneration_probability: float = 1 / 8  # Probability of regenerating a node if it has no children

    variable_temperature: bool = False  # Whether to use variable temperature for sampling
    min_policy_temperature: float = 0.7  # Minimum temperature for variable temperature sampling
    max_policy_temperature: float = 1.1  # Maximum temperature for variable temperature sampling

    c_puct: float = 2.0  # PUCT exploration constant
    num_rollouts: int = 8  # Number of simulations to run for MCTS
    hint_rollouts: int | None = None  # Number of rollouts with hint enabled (None means all rollouts)

    negative_reward: float = -1.0  # Negative reward for invalid/incorrect code
    positive_reward: float = 1.0  # Positive reward for correct code

    job_id: int = int(os.getenv("SLURM_JOB_ID", 0))  # Job ID (read from environment variable)
    cpus: int = int(os.getenv("SLURM_CPUS_PER_TASK", 16))  # Number of CPUs available
    gpus: int = int(os.getenv("SLURM_GPUS", 1))  # Number of GPUs available
    mem: int = int(os.getenv("SLURM_MEM_PER_NODE", 32768))  # Memory per node in MB
    nodelist: str = os.getenv("SLURM_JOB_NODELIST", "")  # List of nodes assigned to the job

    output_dir: str = os.path.join(NET_SCRATCH_PATH, "outputs")  # Directory to save results
    sft_data_dir: str = os.path.join(NET_SCRATCH_PATH, "sft_data")  # Directory to save SFT data
    round_number: int = 1  # "rStar-Math round number"

    config_file: str = ""  # Path to YAML config file

    partition: Optional[str] = None  # SLURM partition to use
    exclude: Optional[str] = None  # Nodes to exclude
    time: Optional[str] = None  # Time limit for job

    # Computed fields
    policy_model_dir: Optional[str] = None  # Full path to policy model
    reward_model_dir: Optional[str] = None  # Full path to reward model
    local_job_dir: Optional[str] = None  # Temporary path for job

    def __post_init__(self):
        """
        Initialize computed fields and load configuration from file if provided

        This method runs automatically after the dataclass is instantiated,
        loading configuration from a file (if specified) and calculating derived values.
        """
        if self.arc_prize:
            self.save_for_visualization = False
            # TODO: adapt config to ARC prize competition settings

        # Parse command line arguments to get config file path
        parser = argparse.ArgumentParser(description='rSTAR meets ARC')
        parser.add_argument('--config-file', type=str, help='Path to YAML config file')
        args, _ = parser.parse_known_args()

        # If config file path is provided via command line, update the config_file field
        if args.config_file:
            self.config_file = args.config_file

        # Load configuration from file
        self._load_from_file()

        # Set computed model directories and other derived values
        if self.fine_tuned and not self.max_model_len > 32768:
            self.policy_model_dir = os.path.join(self.net_scratch_model_base_path, "policy")
            self.reward_model_dir = os.path.join(self.net_scratch_model_base_path, "reward")
        else:
            self.policy_model_dir = os.path.join(self.local_scratch_model_base_path, "policy")
            self.reward_model_dir = os.path.join(self.local_scratch_model_base_path, "reward")

        self.local_job_dir = os.path.join(LOCAL_SCRATCH_PATH, f"job_{self.job_id}")
        self.final_job_dir = os.path.join(self.output_dir, "detailed_logs", f"job_{self.job_id}")
        self.numeric_log_level = getattr(logging, self.log_level.upper(), logging.DEBUG)

        self.search_mode = self.search_mode.lower()

        # limit the number of examples and choose them randomly (with replacement)
        if self.num_examples > 0:
            self.example_names = sample(self.example_names, k=self.num_examples)

        shuffle(self.example_names)

        # Handle search mode specific settings
        if self.search_mode == "bs":
            self.num_rollouts = 1
        elif self.search_mode in ["mcts", "custom"]:
            self.beam_width = 1

    def _load_from_file(self):
        """
        Load configuration settings from a YAML file

        This method:
        1. Checks if the file exists (in configs/ or tmp_configs/ subfolder)
        2. Loads and parses the YAML content
        3. Updates the instance with values from the config file
        """
        try:
            # Define possible config file locations
            config_paths = [
                os.path.join(f"{PROJECT_PATH}/configs", self.config_file),
                os.path.join(f"{HOME_PATH}/tmp_configs", self.config_file)
            ]

            # Find the first existing config file
            config_path = next((path for path in config_paths if os.path.isfile(path)), None)

            if not config_path:
                raise FileNotFoundError(f"Config file not found: {self.config_file}")

            # Read and parse the YAML file
            with open(config_path, 'r') as f:
                config_data = yaml.safe_load(f) or {}

            # Convert kebab-case keys to snake_case for Python compatibility
            config_data = {k.replace('-', '_'): v for k, v in config_data.items()}

            # Update instance attributes with values from config file
            for key, value in config_data.items():
                if hasattr(self, key):
                    setattr(self, key, value)
                    logging.debug(f"Loaded from {config_path}: {key}={value}")

        except Exception as e:
            logging.error(f"Error loading config file: {e}")
