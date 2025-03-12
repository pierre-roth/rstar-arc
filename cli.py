import argparse
import os
import sys
from typing import List, Any, Optional
from config import Config
from schema import PARAM_SCHEMA, PARAM_BY_NAME, DEFAULT_VERBOSE


class CLI:
    """Handles command line interface with auto-generated arguments from schema"""

    @staticmethod
    def parse_args() -> argparse.Namespace:
        """Parse command line arguments based on parameter schema."""
        parser = argparse.ArgumentParser(description='rSTAR meets ARC')
        
        # Add arguments based on parameter schema
        for param in PARAM_SCHEMA:
            # Skip SLURM-specific parameters in Python CLI
            if param.name in ['mem', 'cpus', 'partition', 'exclude', 'nodelist', 'time']:
                continue
                
            if param.is_flag:
                # For boolean flags
                parser.add_argument(
                    param.cli_flag,
                    action='store_true',
                    default=param.default,
                    help=param.help
                )
            else:
                # For regular arguments
                parser.add_argument(
                    param.cli_flag,
                    type=param.type,
                    default=param.default,
                    help=param.help,
                    required=param.is_required
                )

        return parser.parse_args()

    @staticmethod
    def list_task_files(directory: str) -> List[str]:
        """List all JSON files in the given directory."""
        if not os.path.isdir(directory):
            print(f"Error: Directory '{directory}' not found. Please check your ARC data directory.")
            sys.exit(1)

        files = sorted([f for f in os.listdir(directory) if f.endswith('.json')],
                       key=lambda x: x.lower())

        if not files:
            print(f"No JSON files found in directory '{directory}'.")
            sys.exit(1)

        print(f"Found {len(files)} JSON files in '{directory}'")
        return files

    @staticmethod
    def select_task_file(files: List[str], directory: str, task_index: int,
                         verbose: bool = DEFAULT_VERBOSE) -> str:
        """Select a task file by index."""
        if task_index < 1 or task_index > len(files):
            print(f"Error: Task index {task_index} is out of range (1-{len(files)})")
            sys.exit(1)

        chosen_file = os.path.join(directory, files[task_index - 1])
        if verbose:
            print(f"Selected file by index: {chosen_file}")

        return chosen_file

    @staticmethod
    def create_config(args: argparse.Namespace) -> Config:
        """Create a Config object from command line arguments."""
        return Config(args)
        
    @staticmethod
    def print_available_params() -> None:
        """Print all available parameters for documentation."""
        print("Available parameters:")
        print("-" * 40)
        
        # Group parameters by category
        categories = {
            "SLURM Parameters": ['mem', 'cpus', 'partition', 'exclude', 'nodelist', 'time'],
            "Model Parameters": ['policy_model', 'pp_model', 'max_tokens'],
            "Task Parameters": ['task_index', 'task_name', 'all_tasks', 'data_folder'],
            "Search Parameters": ['search_mode', 'max_depth', 'max_iterations', 'beam_width', 'temperature'],
            "Output Parameters": ['output_dir', 'verbose', 'hint'],
            "System Parameters": ['gpus', 'dtype', 'seed', 'deterministic'],
            "Config Parameters": ['config_file']
        }
        
        for category, param_names in categories.items():
            print(f"\n{category}:")
            for name in param_names:
                if name in PARAM_BY_NAME:
                    param = PARAM_BY_NAME[name]
                    default_str = f" (default: {param.default})" if param.default is not None else ""
                    print(f"  {param.cli_flag:<20} {param.help}{default_str}")

