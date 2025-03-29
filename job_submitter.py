#!/usr/bin/env python3
"""
Job Submitter for rStar-ARC

This script automatically generates configuration files with different parameter combinations
and submits them as separate SLURM batch jobs. This allows for easy parameter sweeps and
experimentation with different configurations.
"""

import itertools
import os
import subprocess
import sys
from datetime import datetime
from typing import Any

from constants import HOME_PATH, DEFAULT_DATA_FOLDER

# Create tmp_configs directory if it doesn't exist
TMP_CONFIGS_DIR = os.path.join(HOME_PATH, "tmp_configs")
os.makedirs(TMP_CONFIGS_DIR, exist_ok=True)

# Define parameter configurations to sweep over
PARAMETER_SWEEPS = [
    {
        "search-mode": ["bs"],
        "beam-width": [12],
        "branching-factor": [12],
        "max-depth": [12],
        "policy-temperature": [0.7, 0.9, 1.1],
        "examples-mask": [[True, False, False], [False, True, False], [False, False, True],
                          [True, True, False], [True, False, True], [False, True, True]],
        "data-folder": [os.path.join(DEFAULT_DATA_FOLDER, "very_easy"), os.path.join(DEFAULT_DATA_FOLDER, "easy")]
    }
]

# Base configurations that apply to all jobs
BASE_CONFIG = {
    "log-level": "DEBUG"
}


def dump(data, stream, default_flow_style=False):
    """
    Emulates yaml.dump for basic data types using only the standard library.
    Writes a YAML-like representation of data (typically a dict) to stream.

    Parameters:
        data: The data to dump (dict, list, or scalar).
        stream: A file-like object with a write() method.
        default_flow_style: Ignored in this implementation (always block style).
    """

    def _format_scalar(value):
        # Format booleans, None, numbers, and strings.
        if isinstance(value, bool):
            return "true" if value else "false"
        elif value is None:
            return "null"
        elif isinstance(value, (int, float)):
            return str(value)
        elif isinstance(value, str):
            # Quote the string if it contains special characters or leading/trailing spaces.
            if value == "" or value.strip() != value or ":" in value or "#" in value or "\n" in value:
                # Use double quotes and escape any internal double quotes.
                escaped = value.replace('"', '\\"')
                return f'"{escaped}"'
            else:
                return value
        else:
            # Fallback for other types.
            return str(value)

    def _serialize(obj, indent=0):
        ind = "  " * indent
        if isinstance(obj, dict):
            lines = []
            for key, value in obj.items():
                key_str = str(key)
                # If value is a nested structure, output key on its own line.
                if isinstance(value, (dict, list)):
                    lines.append(f"{ind}{key_str}:")
                    lines.append(_serialize(value, indent + 1))
                else:
                    scalar = _format_scalar(value)
                    lines.append(f"{ind}{key_str}: {scalar}")
            return "\n".join(lines)
        elif isinstance(obj, list):
            lines = []
            for item in obj:
                # For nested structures in lists, put the dash alone.
                if isinstance(item, (dict, list)):
                    lines.append(f"{ind}-")
                    lines.append(_serialize(item, indent + 1))
                else:
                    scalar = _format_scalar(item)
                    lines.append(f"{ind}- {scalar}")
            return "\n".join(lines)
        else:
            # For a scalar, simply return its string representation.
            return ind + _format_scalar(obj)

    yaml_str = _serialize(data)
    stream.write(yaml_str)


def generate_configs(params: dict) -> list[dict[str, Any]]:
    """
    Generate configuration dictionaries for the specified search mode.

    Args:
        params: Dictionary of parameters to sweep over

    Returns:
        List of configuration dictionaries
    """

    # Get parameter sweep definition for the search mode

    # Create all combinations of parameters
    param_names = list(params.keys())
    param_values = list(params.values())

    # Generate all combinations
    all_combinations = list(itertools.product(*param_values))

    # Create config dictionaries
    configs = []
    for combination in all_combinations:
        config = BASE_CONFIG.copy()
        for name, value in zip(param_names, combination):
            config[name] = value
        configs.append(config)

    return configs


def save_config(config: dict[str, Any], idx: int) -> str:
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
    filename = f"{idx}_{timestamp}.yaml"
    filepath = os.path.join(TMP_CONFIGS_DIR, filename)

    with open(filepath, 'w') as f:
        dump(config, f, default_flow_style=False)

    return filepath


def submit_job(config_path: str) -> (int, str):
    """
    Submit a SLURM job with the given configuration.

    Args:
        config_path: Path to the configuration file

    Returns:
        Tuple of (job_id, config_path)
    """
    # Prepare the sbatch command
    cmd = ["sbatch", "run.sh", os.path.basename(config_path)]

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
    # Process each search mode
    for sweep in PARAMETER_SWEEPS:
        configs = generate_configs(sweep)

        for i, config in enumerate(configs):
            # Save the configuration
            config_path = save_config(config, i)
            print(f"Saved configuration to {config_path}")
            # submit_job(config_path)


if __name__ == "__main__":
    main()
