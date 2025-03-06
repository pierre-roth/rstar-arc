import argparse
import os
from constants import *
from config import Config
import sys


class CLI:
    """Handles command line interface"""

    @staticmethod
    def parse_args() -> argparse.Namespace:
        """Parse command line arguments."""
        parser = argparse.ArgumentParser(description='rSTAR meets ARC')
        # LLM choices
        parser.add_argument('--policy-model', type=str, default=DEFAULT_POLICY_LLM,
                            help=f'LLM model to use for step candidate generation (default: {DEFAULT_POLICY_LLM})')
        parser.add_argument('--pp-model', type=str, default=DEFAULT_PP_LLM,
                            help=f'LLM model to use for selecting the most promising candidate steps (default: '
                                 f'{DEFAULT_PP_LLM})')

        # Policy LLM choices
        parser.add_argument('--max-tokens', type=int, default=DEFAULT_MAX_TOKENS,
                            help=f'Maximum number of tokens per step for policy LLM (default: {DEFAULT_MAX_TOKENS})')

        # search mode
        parser.add_argument('--search-mode', type=str, default=DEFAULT_SEARCH_MODE,
                            help=f'Search mode for inference (default: {DEFAULT_SEARCH_MODE})')

        # MCTS parameters
        parser.add_argument('--max-depth', type=int, default=DEFAULT_MAX_DEPTH,
                            help=f'Maximum number of depth iterations allowed (default: {DEFAULT_MAX_DEPTH})')
        
        parser.add_argument('--max-iterations', type=int, default=DEFAULT_MAX_ITERATIONS,
                            help=f'Maximum number of iterations allowed (default: {DEFAULT_MAX_ITERATIONS})')
        
        # Beam search parameters
        parser.add_argument('--beam-width', type=int, default=DEFAULT_BEAM_WIDTH,
                            help=f'Width of the beam for beam search (default: {DEFAULT_BEAM_WIDTH})')

        parser.add_argument('--data-folder', action='store_true', default=False,
                            help=f'Evaluation tasks instead of training? (default: {False})')

        # verbosity
        parser.add_argument('--verbose', action='store_true', default=DEFAULT_VERBOSE,
                            help=f'Print detailed progress information (default: {DEFAULT_VERBOSE})')
        
        # task choice
        parser.add_argument('--task-index', type=int, default=1,
                            help=f'Index of the task to use (default: 1)')
                            
        parser.add_argument('--task-name', type=str, default='',
                            help='Specific task name to use (overrides task-index)')

        parser.add_argument('--all-tasks', action='store_true', default=False,
                            help='Process all tasks in the data_sample/[training or evaluation] directory')

        parser.add_argument('--gpus', type=int, default=1,
                            help='Number of GPUs to use for the LLM')

        # temperature
        parser.add_argument('--temperature', type=float, default=0.0,
                            help='Temperature for LLM sampling (default: 0.0)')

        parser.add_argument('--dtype', type=str, default='float16',
                            help='Data type for model (float16, bfloat16) - use float16 for older GPUs')

        parser.add_argument('--deterministic', action='store_true', default=False,
                            help='Set seed everywhere for deterministic reproducibility')

        parser.add_argument('--seed', type=int, default=42,
                            help='Seed for reproducibility')
        
        parser.add_argument('--output-dir', type=str, default=OUTPUT_BASE_PATH,
                            help=f'Directory to store output files (default: {OUTPUT_BASE_PATH})')
        
        parser.add_argument('--config-file', type=str, default='',
                            help='Path to configuration file')
                            
        parser.add_argument('--hint', type=str, default='',
                            help='Hint to provide to the model')
                            
        parser.add_argument('--eval', action='store_true', default=False,
                            help='Run in evaluation mode')

        return parser.parse_args()

    @staticmethod
    def list_task_files(directory: str) -> list[str]:
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
    def select_task_file(files: list[str], directory: str, task_index: int,
                         verbose: bool = DEFAULT_VERBOSE) -> str:
        """Select a task file by index."""
        if task_index < 1 or task_index > len(files):
            print(f"Error: Task index {task_index} is out of range (1-{len(files)})")
            sys.exit(1)

        chosen_file = os.path.join(directory, files[task_index - 1])
        if verbose:
            print(f"Selected file by index: {chosen_file}")

        return chosen_file

    # create config from args
    @staticmethod
    def create_config(args: argparse.Namespace) -> Config:
        """Create a Config object from command line arguments."""
        return Config(args)

