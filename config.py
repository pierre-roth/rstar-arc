import argparse
from constants import *
import os
import yaml


class Config:
    def __init__(self, args: argparse.Namespace):
        # First load config from file if provided
        if hasattr(args, 'config_file') and args.config_file:
            self.load_from_file(args.config_file)
        
        # Then override with command line arguments
        self.verbose = args.verbose
        # paths
        self.policy_model_dir = os.path.join(MODEL_BASE_PATH, "policy")
        self.pp_model_dir = os.path.join(MODEL_BASE_PATH, "pp")

        self.policy_model = args.policy_model
        self.pp_model = args.pp_model

        self.max_tokens = args.max_tokens
        self.max_depth = args.max_depth
        self.max_iterations = args.max_iterations

        self.data_folder = args.data_folder
        self.task_index = args.task_index
        self.task_name = args.task_name
        self.all_tasks = args.all_tasks
        
        self.search_mode = args.search_mode
        self.temperature = args.temperature
        self.seed = args.seed if hasattr(args, 'seed') else 42
        self.deterministic = args.deterministic
        
        self.gpus = args.gpus
        self.dtype = args.dtype
        
        self.output_dir = args.output_dir
        self.hint = args.hint if hasattr(args, 'hint') else ""
        self.eval = args.eval if hasattr(args, 'eval') else False
        
        # Beam search specific parameters
        self.beam_width = args.beam_width if hasattr(args, 'beam_width') else DEFAULT_BEAM_WIDTH

    def load_from_file(self, config_file: str):
        """Load configuration from YAML file."""
        try:
            with open(config_file, 'r') as f:
                config_data = yaml.safe_load(f)
                
            # Update attributes from file
            for key, value in config_data.items():
                # Convert dashed keys to underscore
                attr_name = key.replace('-', '_')
                setattr(self, attr_name, value)
        except Exception as e:
            print(f"Error loading config file: {e}")

    def to_dict(self):
        """Convert config to dictionary for saving."""
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}
        
    def save_to_file(self, file_path: str):
        """Save configuration to a YAML file."""
        try:
            with open(file_path, 'w') as f:
                yaml.dump(self.to_dict(), f, default_flow_style=False)
            print(f"Config saved to {file_path}")
        except Exception as e:
            print(f"Error saving config: {e}")
            

