import argparse
import logging
import os
import sys
from dataclasses import dataclass, field
from random import shuffle, sample
from typing import Optional

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
    evaluation: bool = False  # Whether the model is running on the evaluation set

    log_level: str = "INFO"  # Logging level: DEBUG, INFO, WARNING, ERROR, CRITICAL
    numeric_log_level: int = -1  # Numeric logging level (set automatically)

    save_for_visualization: bool = True  # Whether to visualize the reasoning steps
    solve_only_unsolved: bool = False  # Whether to only solve unsolved tasks
    solutions_per_task: int = None  # Bound the number of solutions (that pass training examples) to find for each task (None means as many as possible)
    solution_per_pair: int = 4  # Number of complete solutions to store for each preference pair
    save_sft_data: bool = True  # Whether to save SFT data

    num_examples: int = -1  # Number of examples to use for training (default: all)
    example_names: list[list[str]] = field(default_factory=lambda: [["6d0aefbc", "1cf80156"], ["1cf80156",
                                                                                               "00d62c1b"]])  # list of names of example tasks (to be used sequentially in different rollouts)
    rotate_example: bool = False  # Whether to rotate the example tasks in each rollout

    policy_model: str = "Qwen/Qwen2.5-Coder-7B-Instruct"  # Model that generates reasoning steps
    reward_model: str = "Qwen/Qwen2.5-Coder-7B-Instruct"  # Reward Model for evaluating steps
    model_initialization_times = {"policy": None, "reward": None}  # Time taken to initialize models
    enforce_eager: bool = False
    policy_vram_percentage: float = 0.45  # Percentage of VRAM to use for the policy model

    use_reward_model: bool = False  # Whether to use the reward model
    fine_tuned: bool = False  # Whether the model is fine-tuned

    net_scratch_model_base_path: str = os.path.join(NET_SCRATCH_PATH, "models")  # Base path where models are stored
    local_scratch_model_base_path: str = os.path.join(LOCAL_SCRATCH_PATH, "models")  # Base path where models are stored

    max_tokens: int = 512  # Maximum tokens for generation of a single step
    dtype: str = "bfloat16"  # Data type for model (affects precision/speed)
    max_seq_len: int = 14336  # Affects the context window size
    # max_num_seqs: int = 1024 # Maximum number of sequences to generate in parallel
    # max_num_batched_tokens = 16384 # Maximum number of tokens to process in a batch
    top_p: float = 1.0  # Top-p sampling parameter (cumulative probability cutoff)
    top_k: int = -1  # Top-k sampling parameter (number of candidates to consider)
    repetition_penalty: float = 1.05  # Penalty for repeating tokens in generation

    policy_temperature: float = 0.9  # Sampling temperature for LLM generation

    variable_temperature: bool = False  # Whether to use variable temperature for sampling
    min_policy_temperature: float = 0.7  # Minimum temperature for variable temperature sampling
    max_policy_temperature: float = 1.1  # Maximum temperature for variable temperature sampling

    seed: int = 42  # Random seed for reproducibility
    deterministic: bool = False  # Whether to enforce deterministic behavior

    num_tasks: int = -1  # Number of tasks to process (-1 means all tasks)
    data_folder: str = DEFAULT_DATA_FOLDER  # Path to ARC task data
    task_names: list[str] | None = None  # List of task names to process
    sort_by_length = False  # Whether to sort tasks by length

    search_mode: str = "bs"  # Search algorithm - "bs" for beam search, "mcts" for Monte Carlo Tree Search

    max_depth: int = 10  # Maximum number of reasoning steps

    beam_width: int = 3  # Number of top-scoring beams to track
    branching_factor: int = 4  # Number of children to generate if no children exist yet
    regeneration_probability: float = 1 / 8  # Probability of regenerating a node if it has no children
    min_step_margin: float = 0.3  # Minimum avg q value margin for preference pairs

    c_puct: float = 2.0  # PUCT exploration constant
    num_rollouts: int = 8  # Number of simulations to run for MCTS
    hint_rollouts: int = -1  # Number of rollouts with hint enabled (None means all rollouts)

    negative_reward: float = -1.0  # Negative reward for invalid/incorrect code
    positive_reward: float = 1.0  # Positive reward for correct code

    batch_size: int = -1  # Batch size for parallel inference (-1 means all at once, otherwise batch size)

    job_id: int = int(os.getenv("SLURM_JOB_ID", 0))  # Job ID (read from environment variable)
    cpus: int = int(os.getenv("SLURM_CPUS_PER_TASK", 16))  # Number of CPUs available
    gpus: int = int(os.getenv("SLURM_GPUS", 1))  # Number of GPUs available
    mem: int = int(os.getenv("SLURM_MEM_PER_NODE", 32768))  # Memory per node in MB
    nodelist: str = os.getenv("SLURM_JOB_NODELIST", "")  # List of nodes assigned to the job
    partition: Optional[str] = None  # SLURM partition to use
    exclude: Optional[str] = None  # Nodes to exclude
    time: Optional[str] = None  # Time limit for job

    output_dir: str = os.path.join(NET_SCRATCH_PATH, "outputs")  # Directory to save results
    sft_data_dir: str = os.path.join(NET_SCRATCH_PATH, "sft_data")  # Directory to save SFT data
    task_dir: str = os.path.join(NET_SCRATCH_PATH, "task_data")  # Directory to save task data
    round_number: int = 2  # "rStar-Math round number"

    config_file: str = ""  # Path to YAML config file

    # Computed fields
    policy_model_dir: Optional[str] = None  # Full path to policy model
    reward_model_dir: Optional[str] = None  # Full path to reward model
    local_job_dir: Optional[str] = None  # Temporary path for the job

    # training related
    train_on_prompts: bool = False  # Whether to train on prompts

    task_validation_fraction: float = 0.05  # fraction of examples to use for validation (not used for training)
    example_validation_fraction: float = 0.05  # fraction of examples of tasks used for training that are used for validation
    curriculum_learning: bool = False  # Whether to sort training examples by code length (curriculum learning)

    # If true, skips LoRA adapters and fine-tunes all model parameters
    full_finetune: bool = False
    learning_rate: float = 2e-5
    num_train_epochs: int = 1
    per_device_train_batch_size: int = 1
    per_device_eval_batch_size: int = 1
    gradient_accumulation_steps: int = 16
    weight_decay: float = 0.0
    max_grad_norm: float = 1.0  # Maximum gradient norm for gradient clipping
    label_smoothing_factor: float = 0.0  # Label smoothing factor for training

    logging_steps: int = 25
    eval_steps: int = 50
    save_steps: int = 50
    save_total_limit: int = 2
    warmup_ratio: float = 0.03

    use_bf16: bool = True
    gradient_checkpointing: bool = True
    lr_scheduler_type: str = "cosine"

    # LoRA
    lora_rank: int = 32
    lora_alpha: int = 32
    lora_dropout: float = 0.03

    # Logging / tracking
    report_to: str = "wandb"
    wandb_project: str = "deepthink-sft"
    wandb_entity: str | None = None  # optional team name

    # Evaluation options
    num_validation_samples: int = 5  # Number of validation prompts to sample for qualitative evaluation
    num_training_samples: int = 5  # Number of training prompts to sample for qualitative evaluation
    pass_k: int = 8  # Number of generations per task to sample for pass@k evaluation
    eval_temperatures: tuple[float, ...] = (0.1, 0.4, 0.8)  # Sampling temperatures for multiple generations during evaluation

    # curriculum settings
    val_examples_per_task: int = 2
    test_examples_per_task: int = 2
    max_task_description_chars: int = 2048
    min_active_tasks: int = 8  # trigger refill when active set < this
    max_stagnation_epochs: int = 16
    curriculum_eval_temperatures: list[float] = field(default_factory=lambda: [0.2, 0.8])  # tried in round‑robin
    task_forgetting_threshold: float = 0.5

    perplexity_window_size: Optional[
        int] = None  # Window size for smoothing per-token perplexity (None for no smoothing)
    min_steps_for_format_adherence: int = 2  # Minimum number of steps required for format adherence

    # reward
    reward_value_head_dropout: float = 0.1  # dropout for the value head
    reward_batch_size: int = 512

    def __post_init__(self):
        """
        Initialize computed fields and load configuration from file if provided

        This method runs automatically after the dataclass is instantiated,
        loading configuration from a file (if specified) and calculating derived values.
        """
        if self.arc_prize:
            self.save_for_visualization = False
            self.save_sft_data = False
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
        if self.fine_tuned or self.max_seq_len > 32768:
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
        elif self.search_mode in ["mcts", "custom", "bootstrap"]:
            self.beam_width = 1

        if not self.use_reward_model:
            self.policy_vram_percentage = 0.9

    def _load_from_file(self):
        """
        Load configuration settings from a YAML file

        This method:
        1. Checks if the file exists (in configs/ or tmp_configs/ subfolder)
        2. Loads and parses the YAML content
        3. Updates the instance with values from the config file
        """
        try:
            # Read and parse the YAML file
            with open(os.path.join(PROJECT_PATH, self.config_file), 'r') as f:
                config_data = yaml.safe_load(f) or {}

            # Convert kebab-case keys to snake_case for Python compatibility
            config_data = {k.replace('-', '_'): v for k, v in config_data.items()}

            # Update instance attributes with values from config file
            for key, value in config_data.items():
                if hasattr(self, key):
                    setattr(self, key, value)
                    logging.debug(f"Loaded from {self.config_file}: {key}={value}")

        except FileNotFoundError:
            logging.error(f"Config file not found: {self.config_file}")
            sys.exit(1)

        except Exception as e:
            logging.error(f"Error loading config file: {e}")
