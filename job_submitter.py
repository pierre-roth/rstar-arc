#!/usr/bin/env python3
"""
Job Submitter for rStar-ARC

This script automatically generates configuration files with different parameter combinations
and submits them as separate SLURM batch jobs. This allows for easy parameter sweeps and
experimentation with different configurations.
"""

import os
import sys
import argparse
import itertools
import subprocess
import yaml
from datetime import datetime
from typing import Dict, List, Any, Tuple

# Ensure we're in the project directory
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(PROJECT_DIR)

# Create tmp_configs directory if it doesn't exist
TMP_CONFIGS_DIR = os.path.join(PROJECT_DIR, "tmp_configs")
os.makedirs(TMP_CONFIGS_DIR, exist_ok=True)

# Define parameter configurations to sweep over
PARAMETER_SWEEPS = {
    # For beam search (bs)
    "bs": {
        "search-mode": ["bs"],
        "beam-width": [3, 5, 8],
        "branching-factor": [3, 5, 8],
        "max-depth": [8, 12],
        "policy-temperature": [0.5, 0.7, 0.9]
    },
    # For Monte Carlo Tree Search (mcts)
    "mcts": {
        "search-mode": ["mcts"],
        "c-puct": [1.0, 2.0, 4.0],
        "num-simulations": [8, 16, 32],
        "max-depth": [8, 12],
        "policy-temperature": [0.5, 0.7, 0.9]
    },
    # For Pure-Walker Monte Carlo Tree Search (pwmcts)
    "pwmcts": {
        "search-mode": ["pwmcts"],
        "c-puct": [1.0, 2.0, 4.0],
        "num-simulations": [8, 16, 32],
        "max-depth": [8, 12],
        "policy-temperature": [0.5, 0.7, 0.9]
    }
}

# Base configurations that apply to all jobs
BASE_CONFIG = {
    "log-level": "INFO",
    "batch-size": 5,  # Process 5 tasks at a time
    "data-folder": "${HOME_PATH}/rstar-arc/data/evaluation",  # Path adjusted at runtime
    "num-examples": 2
}


def generate_configs(search_mode: str, num_configs: int = 5) -> List[Dict[str, Any]]:
    """
    Generate configuration dictionaries for the specified search mode.

    Args:
        search_mode: The search mode to generate configs for ("bs", "mcts", or "pwmcts")
        num_configs: Maximum number of configurations to generate

    Returns:
        List of configuration dictionaries
    """
    if search_mode not in PARAMETER_SWEEPS:
        raise ValueError(f"Unknown search mode: {search_mode}")

    # Get parameter sweep definition for the search mode
    params = PARAMETER_SWEEPS[search_mode]

    # Create all combinations of parameters
    param_names = list(params.keys())
    param_values = [params[name] for name in param_names]

    # Generate all combinations
    all_combinations = list(itertools.product(*param_values))

    # Limit to requested number of configurations
    selected_combinations = all_combinations[:num_configs]

    # Create config dictionaries
    configs = []
    for combination in selected_combinations:
        config = BASE_CONFIG.copy()
        for name, value in zip(param_names, combination):
            config[name] = value
        configs.append(config)

    return configs


def save_config(config: Dict[str, Any], search_mode: str, idx: int) -> str:
    """
    Save a configuration to a YAML file.

    Args:
        config: Configuration dictionary
        search_mode: Search mode identifier
        idx: Configuration index

    Returns:
        Path to the saved configuration file
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{search_mode}_{idx}_{timestamp}.yaml"
    filepath = os.path.join(TMP_CONFIGS_DIR, filename)

    with open(filepath, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)

    return filepath


def submit_job(config_path: str, partition: str = None, time_limit: str = None) -> Tuple[int, str]:
    """
    Submit a SLURM job with the given configuration.

    Args:
        config_path: Path to the configuration file
        partition: Optional SLURM partition
        time_limit: Optional time limit for the job

    Returns:
        Tuple of (job_id, config_path)
    """
    # Prepare the sbatch command
    cmd = ["sbatch"]

    # Add optional parameters if provided
    if partition:
        cmd.extend(["--partition", partition])
    if time_limit:
        cmd.extend(["--time", time_limit])

    # Add the run script and config file
    cmd.extend(["run.sh", os.path.basename(config_path)])

    # Run the sbatch command
    try:
        # We're capturing the output to extract the job ID
        result = subprocess.run(cmd, check=True, stdout=subprocess.PIPE, text=True)

        # Extract the job ID from the output (format: "Submitted batch job 12345")
        job_id = int(result.stdout.strip().split()[-1])

        print(f"Submitted job {job_id} with config {os.path.basename(config_path)}")
        return job_id, config_path

    except subprocess.CalledProcessError as e:
        print(f"Error submitting job with config {config_path}: {e}", file=sys.stderr)
        return -1, config_path


def main():
    """Main function for the job submitter."""
    parser = argparse.ArgumentParser(description="Submit batch jobs with different configurations")
    parser.add_argument("--search-mode", type=str, choices=["bs", "mcts", "pwmcts", "all"],
                        default="all", help="Search mode to use")
    parser.add_argument("--configs-per-mode", type=int, default=5,
                        help="Number of configurations to generate per search mode")
    parser.add_argument("--partition", type=str, help="SLURM partition to use")
    parser.add_argument("--time-limit", type=str, help="Time limit for jobs (e.g., '24:00:00')")
    parser.add_argument("--dry-run", action="store_true",
                        help="Generate configs but don't submit jobs")

    args = parser.parse_args()

    # Determine which search modes to use
    if args.search_mode == "all":
        search_modes = ["bs", "mcts", "pwmcts"]
    else:
        search_modes = [args.search_mode]

    submitted_jobs = []

    # Process each search mode
    for mode in search_modes:
        print(f"Generating configurations for {mode}...")
        configs = generate_configs(mode, args.configs_per_mode)

        for i, config in enumerate(configs):
            # Save the configuration
            config_path = save_config(config, mode, i + 1)
            print(f"Saved configuration to {config_path}")

            # Submit the job
            if not args.dry_run:
                job_id, _ = submit_job(config_path, args.partition, args.time_limit)
                if job_id > 0:
                    submitted_jobs.append((job_id, os.path.basename(config_path)))

    # Print summary
    if not args.dry_run and submitted_jobs:
        print("\nSubmitted Jobs Summary:")
        print("----------------------")
        for job_id, config_file in submitted_jobs:
            print(f"Job {job_id}: {config_file}")
        print(f"Total jobs submitted: {len(submitted_jobs)}")

    # If dry run, just show what would have been done
    if args.dry_run:
        print("\nDRY RUN - No jobs were submitted")
        print(f"Would have submitted {len(configs) * len(search_modes)} jobs")


if __name__ == "__main__":
    main()
