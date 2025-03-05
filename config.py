import argparse
from constants import *
import os


class Config:
    def __init__(self, args: argparse.Namespace):
        self.verbose = args.verbose
        # paths
        self.policy_model_dir = os.path.join(MODEL_BASE_PATH, "policy")
        self.pp_model_dir = os.path.join(MODEL_BASE_PATH, "pp")

        self.policy_model = args.policy_model
        self.pp_model = args.pp_model

        self.max_tokens: int = args.max_tokens

        self.max_depth: int = args.max_depth

        self.data_folder: str = args.data_folder

        self.task_name: str = args.task_name

        self.all_tasks: bool = args.all_tasks

        self.temperature: float = 0.0

        self.seed: int = 0

        self.gpus = args.gpus

        self.dtype: str = args.dtype


