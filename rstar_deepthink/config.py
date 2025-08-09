import argparse
import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Optional

import yaml

from constants import (
    NET_SCRATCH_PATH,
    LOCAL_SCRATCH_PATH,
    DEFAULT_DATA_FOLDER,
)


@dataclass
class Config:
    """
    Unified configuration for rStar meets ARC.

    Loads from an optional YAML file passed via --config-file, then applies
    sensible defaults and derives common paths. This class intentionally keeps
    imports light to avoid heavy dependencies during tests or simple scripts.
    """

    # General
    evaluation: bool = False
    log_level: str = "INFO"
    numeric_log_level: int = -1

    save_for_visualization: bool = True
    solve_only_unsolved: bool = False
    solutions_per_task: int | None = None
    solution_per_pair: int = 4
    save_sft_data: bool = True

    # Prompt examples
    num_examples: int = -1
    example_names: list[list[str]] = field(
        default_factory=lambda: [["6d0aefbc", "1cf80156"], ["1cf80156", "00d62c1b"]]
    )
    rotate_example: bool = False

    # Models
    run_name: str | None = None
    policy_model: str = "Qwen/Qwen2.5-Coder-7B-Instruct"
    reward_model: str = "Qwen/Qwen2.5-Coder-7B-Instruct"
    model_initialization_times = {"policy": None, "reward": None}
    enforce_eager: bool = False
    policy_vram_percentage: float = 0.45

    use_reward_model: bool = False
    fine_tuned: bool = False

    net_scratch_model_base_path: str = os.path.join(NET_SCRATCH_PATH, "models")
    local_scratch_model_base_path: str = os.path.join(LOCAL_SCRATCH_PATH, "models")

    # Sampling/generation
    max_tokens: int = 512
    dtype: str = "bfloat16"
    max_seq_len: int = 14336
    top_p: float = 1.0
    top_k: int = -1
    repetition_penalty: float = 1.05
    policy_temperature: float = 0.9
    variable_temperature: bool = False
    min_policy_temperature: float = 0.7
    max_policy_temperature: float = 1.1

    # Reproducibility
    seed: int = 42
    deterministic: bool = False

    # Data
    num_tasks: int = -1
    data_folder: str = DEFAULT_DATA_FOLDER
    task_names: list[str] | None = None
    sort_by_length: bool = False
    length_pre_filtering: bool = False

    # Search
    search_mode: str = "bs"
    max_depth: int = 10
    beam_width: int = 3
    branching_factor: int = 4
    regeneration_probability: float = 1 / 8
    min_step_margin: float = 0.4

    # MCTS
    c_puct: float = 2.0
    num_rollouts: int = 8
    hint_rollouts: int = -1

    # Rewards
    negative_reward: float = -1.0
    positive_reward: float = 1.0

    # Batching across tasks
    batch_size: int = -1
    skip_batches: int = 0
    num_batches: int = -1

    # SLURM-related (read via env when present; defaults are fine for tests)
    job_id: int = int(os.getenv("SLURM_JOB_ID", 0))
    cpus: int = int(os.getenv("SLURM_CPUS_PER_TASK", 72))
    gpus: int = int(os.getenv("SLURM_GPUS", 1))
    mem: int = int(os.getenv("SLURM_MEM_PER_NODE", 32768))
    nodelist: str = os.getenv("SLURM_JOB_NODELIST", "")
    partition: Optional[str] = None
    exclude: Optional[str] = None
    time: Optional[str] = None

    # Output paths
    output_dir: str = os.path.join(NET_SCRATCH_PATH, "outputs")
    sft_data_dir: str = os.path.join(NET_SCRATCH_PATH, "sft_data")
    task_dir: str = os.path.join(NET_SCRATCH_PATH, "task_data")
    round_number: int = 6

    # CLI/config file
    config_file: str = ""
    execute_in_subprocess: bool = True

    # Derived path fields (filled in __post_init__)
    policy_model_dir: Optional[str] = None
    reward_model_dir: Optional[str] = None
    local_job_dir: Optional[str] = None

    # Training
    resume_from_checkpoint: bool = False
    training_dataset_name: str = "policy_dataset_training.jsonl"
    validation_dataset_name: str = "policy_dataset_validation.jsonl"
    train_on_prompts: bool = False
    qualitative_eval: bool = False
    attn_implementation: Optional[str] = None
    torch_compile: bool = False

    task_validation_fraction: float = 0.003
    example_validation_num: int = 1
    example_validation_threshold: int = 5
    example_validation_probability: float = 0.015

    full_finetune: bool = True
    learning_rate: float = 2e-5
    num_train_epochs: int = 1
    per_device_train_batch_size: int = 1
    per_device_eval_batch_size: int = 1
    gradient_accumulation_steps: int = 16
    weight_decay: float = 0.0
    max_grad_norm: float = 1.0
    label_smoothing_factor: float = 0.0

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

    # Tracking
    report_to: str = "wandb"
    wandb_project: str = "deepthink-sft"
    wandb_entity: str | None = None

    # Eval options
    num_validation_samples: int = 5
    num_training_samples: int = 5
    pass_k: int = 8
    eval_temperatures: tuple[float, ...] = (0.1, 0.4, 0.8)

    # Curriculum
    val_examples_per_task: int = 2
    test_examples_per_task: int = 2
    max_task_description_chars: int = 2048
    min_active_tasks: int = 8
    max_stagnation_epochs: int = 16
    curriculum_eval_temperatures: list[float] = field(default_factory=lambda: [0.2, 0.8])
    task_forgetting_threshold: float = 0.5

    perplexity_window_size: Optional[int] = None
    min_steps_for_format_adherence: int = 2

    # Reward model scoring
    reward_value_head_dropout: float = 0.1
    reward_batch_size: int = 64

    # ------------------------------------------------------------------
    def __post_init__(self):
        """Initialize from CLI/YAML, then derive paths and logging."""
        self._load_cli_config_if_provided()
        self._derive_paths()
        self._configure_logging()

    # Initialization helpers
    def _load_cli_config_if_provided(self) -> None:
        """Populate fields from a YAML config passed via --config-file."""
        parser = argparse.ArgumentParser(add_help=False)
        parser.add_argument("--config-file", type=str)
        try:
            args, _ = parser.parse_known_args()
        except SystemExit:
            return

        if args and args.config_file:
            self.config_file = args.config_file
            self._load_from_file(self.config_file)

    def _load_from_file(self, path: str) -> None:
        """Load YAML and set matching attributes on this dataclass."""
        try:
            with open(path, "r") as f:
                data = yaml.safe_load(f) or {}
        except FileNotFoundError:
            logging.error(f"Config file not found: {path}")
            sys.exit(1)
        except Exception as e:
            logging.error(f"Error loading config file '{path}': {e}")
            sys.exit(1)

        for key, value in data.items():
            if hasattr(self, key):
                setattr(self, key, value)
                logging.debug(f"Loaded from {path}: {key}={value}")

    def _derive_paths(self) -> None:
        """Set or validate paths that can be inferred without changing semantics.

        Keep model dirs as-is (may be None) for full backward compatibility;
        only ensure data_folder has a default.
        """
        # Do not auto-fill policy_model_dir or reward_model_dir to avoid changing
        # where users expect models to be loaded from or downloaded to.
        # Leave local_job_dir unchanged as well; tests or callers set it explicitly.

        # Ensure data folder is set to a sensible default if empty/falsey
        self.data_folder = self.data_folder or DEFAULT_DATA_FOLDER

    def _configure_logging(self) -> None:
        """Configure logging according to string level in config."""
        level = (self.log_level or "INFO").upper()
        numeric = getattr(logging, level, logging.INFO)
        self.numeric_log_level = int(numeric)
        # Do a basic configuration only if root handlers are not yet set up
        if not logging.getLogger().handlers:
            logging.basicConfig(level=self.numeric_log_level, format="%(levelname)s:%(name)s:%(message)s")
        else:
            logging.getLogger().setLevel(self.numeric_log_level)
