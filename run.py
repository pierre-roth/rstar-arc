import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

import questionary
import yaml

# Dynamically find python scripts that are likely targets for sbatch
PROJECT_ROOT = Path(__file__).parent.resolve()

# Add scripts explicitly found in the run_*.sh files (even if not found dynamically)
# Add more here if needed
SCRIPTS = [
    "main.py",
    "train/clean_sft_data.py",
    "train/augment_sft_data.py",
    "train/generate_policy_training_dataset.py",
    "train/generate_policy_evaluation_dataset.py",
    "train/train_policy.py",
    "test/merge_lora.py",
    "train/train_reward.py",
    "train/validate_arc_jsonl.py",
]

# --- SLURM Defaults ---
DEFAULT_SLURM_CONFIG = {
    "cpus_per_task": 16,
    "mem": "62G",
    "gpu_type": "geforce_rtx_2080",  # Default GPU type
    "num_gpus": 1,  # Default number of GPUs if type is selected
}

GPU_OPTIONS = ["none", "geforce_rtx_3090", "rtx_a6000", "a100_80gb", "titan_rtx", "geforce_rtx_2080", "other"]

# --- Environment Defaults ---
DEFAULT_ENV_CONFIG = {
    "install_python": True,
}


# --- Helper Functions ---
def find_yaml_configs(config_dir: Path) -> List[str]:
    """Finds YAML configuration files."""
    yaml_files = ["none"]  # Option to not use a config
    if config_dir.is_dir():
        for item in config_dir.glob("*.yaml"):
            if item.is_file():
                # Store relative path from project root
                try:
                    relative_path = item.relative_to(PROJECT_ROOT)
                    yaml_files.append(str(relative_path))
                except ValueError:
                    yaml_files.append(item.name)  # Fallback
    return sorted(yaml_files)


def ask_question(
        question_type: str, message: str, choices: Optional[List[str]] = None, default: Any = None, **kwargs
) -> Any:
    """Wrapper for questionary prompts."""
    # Create base kwargs dict
    question_kwargs = {"message": message, **kwargs}

    # Add parameters conditionally based on question type
    if question_type in ["select", "checkbox", "rawselect"]:
        question_kwargs["choices"] = choices

    # Default is supported by all question types
    if default is not None:
        question_kwargs["default"] = default

    # Get and call the appropriate questionary method
    question = getattr(questionary, question_type)(**question_kwargs)

    answer = question.ask()
    if answer is None:  # Handle Ctrl+C or escape
        print("\nOperation cancelled by user.")
        sys.exit(1)
    # Handle empty input for text prompts if a default exists
    if question_type == "text" and not answer and default is not None:
        return default
    return answer


# --- Main Logic ---

def main():
    parser = argparse.ArgumentParser(
        description="Generate and submit a SLURM job based on interactive input or a YAML config.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config-file",
        type=str,
        help="Path to a YAML configuration file to pre-fill the answers.",
    )
    args = parser.parse_args()

    config: Dict[str, Any] = {
        "script": None,
        "slurm": DEFAULT_SLURM_CONFIG.copy(),
        "env": DEFAULT_ENV_CONFIG.copy(),
        "script_args": "",
        "yaml_config": None,
    }

    # Load config from YAML if provided
    if args.config_file:
        print(f"Loading configuration from: {args.config_file}")
        try:
            with open(args.config_file, "r") as f:
                yaml_config = yaml.safe_load(f)
                # Simple merge: YAML values override defaults
                # More sophisticated merging could be added here if needed
                if 'script' in yaml_config:
                    config['script'] = yaml_config['script']
                if 'slurm' in yaml_config:
                    config['slurm'].update(yaml_config['slurm'])
                if 'env' in yaml_config:
                    config['env'].update(yaml_config['env'])
                if 'script_args' in yaml_config:
                    config['script_args'] = yaml_config['script_args']
                if 'yaml_config' in yaml_config:
                    config['yaml_config'] = yaml_config['yaml_config']
            print("Configuration loaded successfully.")
        except FileNotFoundError:
            print(f"Error: Config file '{args.config_file}' not found. Proceeding with interactive prompts.")
        except Exception as e:
            print(f"Error loading YAML config '{args.config_file}': {e}. Proceeding with interactive prompts.")

    # --- Interactive Prompts (if not fully specified by YAML) ---

    # 1. Select Python Script
    available_scripts = SCRIPTS
    if not config.get("script"):
        config["script"] = ask_question(
            "select",
            "Which Python script do you want to run?",
            choices=available_scripts,
            default=available_scripts[0] if available_scripts else None,
        )

    # 2. Select YAML Config for the Python script
    available_configs = find_yaml_configs(PROJECT_ROOT / "configs")
    if not config.get("yaml_config"):
        # Check if the script has a default config convention (e.g. script_name.yaml)
        potential_default_config = f"configs/{Path(config['script']).stem}.yaml"
        default_config = potential_default_config if potential_default_config in available_configs else "none"

        config["yaml_config"] = ask_question(
            "select",
            "Select a YAML config file for the script (passed as --config-file):",
            choices=available_configs,
            default=default_config,
        )

    # 3. SLURM Configuration
    print("\n--- SLURM Configuration ---")
    if 'cpus_per_task' not in config['slurm'] or not args.config_file:  # Ask if not in YAML or interactive mode
        config["slurm"]["cpus_per_task"] = int(ask_question(
            "text",
            "Number of CPUs per task?",
            default=str(config["slurm"]["cpus_per_task"]),
            validate=lambda text: text.isdigit() or "Please enter a valid number",
        ))

    if 'mem' not in config['slurm'] or not args.config_file:
        config["slurm"]["mem"] = ask_question(
            "text",
            "Memory required (e.g., 32G, 128G)?",
            default=config["slurm"]["mem"],
        )

    if 'gpu_type' not in config['slurm'] or not args.config_file:
        config["slurm"]["gpu_type"] = ask_question(
            "select",
            "Select GPU type:",
            choices=GPU_OPTIONS,
            default=config["slurm"]["gpu_type"],
        )
        if config["slurm"]["gpu_type"] == "other":
            config["slurm"]["gpu_type"] = ask_question(
                "text", "Enter custom GPU type (for --gres):"
            )
        elif config["slurm"]["gpu_type"] != "none":
            if 'num_gpus' not in config['slurm'] or not args.config_file:
                config["slurm"]["num_gpus"] = int(ask_question(
                    "text",
                    "Number of GPUs?",
                    default=str(config["slurm"]["num_gpus"]),
                    validate=lambda text: text.isdigit() or "Please enter a valid number",
                ))

    # 4. Environment Setup
    print("\n--- Environment Configuration ---")
    if 'install_python' not in config['env'] or not args.config_file:
        config["env"]["install_python"] = ask_question("confirm", "Install python for subprocesses?", default=True)

    print("\n--- Custom Script Arguments ---")
    if 'script_args' not in config or not args.config_file:
        config["script_args"] = ask_question(
            "text",
            "Enter any additional arguments for the python script:",
            default=config.get("script_args", ""),
        )

    sbatch_script_content = f"""#!/bin/bash
#SBATCH --mail-type=NONE # mail configuration: NONE, BEGIN, END, FAIL, REQUEUE, ALL
#SBATCH --output=/itet-stor/piroth/net_scratch/outputs/jobs/%j.out # Keep minimal SLURM logging
#SBATCH --error=/itet-stor/piroth/net_scratch/outputs/jobs/%j.err # Keep minimal SLURM logging
#SBATCH --nodes=1
#SBATCH --cpus-per-task={config['slurm']['cpus_per_task']}
#SBATCH --mem={config['slurm']['mem']}
"""
    # Add GPU resource request if a GPU is selected
    if config["slurm"]["gpu_type"] != "none":
        sbatch_script_content += f"#SBATCH --gres=gpu:{config['slurm']['num_gpus']}\n"
        sbatch_script_content += f"#SBATCH --constraint='{config['slurm']['gpu_type']}'\n"

    sbatch_script_content += f"""
ETH_USERNAME=piroth
PROJECT_NAME=rstar-arc
DIRECTORY=/home/${{ETH_USERNAME}}/${{PROJECT_NAME}}
CONDA_ENVIRONMENT=arc-solver
NET_SCRATCH_PATH=/itet-stor/${{ETH_USERNAME}}/net_scratch
"""

    if config["env"]["install_python"]:
        sbatch_script_content += f"""
# --- Configuration for Minimal Python Subprocess Environment ---
# Using the details provided by the user (Link validated ~ Mar 28, 2025)
TARGET_PYTHON_VERSION_BASE="3.12.9" # Base version for directory naming
TARGET_PYTHON_VERSION_FULL="3.12.9+20250317" # Full version string including build date
# Exact filename provided by user
PYTHON_FILENAME="cpython-${{TARGET_PYTHON_VERSION_FULL}}-x86_64-unknown-linux-gnu-install_only.tar.gz"
# Exact download URL provided by user
PYTHON_DOWNLOAD_URL="https://github.com/astral-sh/python-build-standalone/releases/download/20250317/${{PYTHON_FILENAME}}"
# Install location in persistent scratch
PYTHON_INSTALL_BASE_DIR="/scratch/${{ETH_USERNAME}}/minipython_install"
# Use base version for the directory name for simplicity
PYTHON_INSTALL_DIR="${{PYTHON_INSTALL_BASE_DIR}}/python-${{TARGET_PYTHON_VERSION_BASE}}"
MINIMAL_PYTHON_EXEC_PATH="${{PYTHON_INSTALL_DIR}}/bin/python" # Path the main script will use
"""
    sbatch_script_content += f"""
# Exit on errors
set -o errexit
set -o pipefail  # Fails if any command in a pipe fails
set -o nounset   # Fails if undefined variables are used

# --- Ensure Base Scratch Directory Exists ---
echo "Ensuring base scratch directory exists: /scratch/${{ETH_USERNAME}}"
mkdir -p "/scratch/${{ETH_USERNAME}}"
if [ $? -ne 0 ]; then
    echo "CRITICAL: Failed to create or access base scratch directory /scratch/${{ETH_USERNAME}}. Check permissions." >&2
    exit 1
fi
echo "Base scratch directory OK."

# Set up local scratch directories for job output and models
local_models_dir="/scratch/${{ETH_USERNAME}}/models"
local_job_dir="/scratch/${{ETH_USERNAME}}/job_${{SLURM_JOB_ID}}" # Node-local ephemeral job data
final_job_dir="/itet-stor/${{ETH_USERNAME}}/net_scratch/outputs/detailed_logs/job_${{SLURM_JOB_ID}}" # Persistent logs

# Create local scratch model directory
if ! mkdir -p "${{local_models_dir}}"; then
    echo "Failed to create local scratch model directory: ${{local_models_dir}}" >&2
    exit 1
fi

# Create local scratch job directory
if ! mkdir -p "${{local_job_dir}}"; then
    echo "Failed to create local scratch job directory: ${{local_job_dir}}" >&2
    exit 1
fi

# Create net scratch job directory for final logs
if ! mkdir -p "${{final_job_dir}}"; then
    echo "Failed to create net scratch directory: ${{final_job_dir}}" >&2
    exit 1
fi


# --- CORRECTED Cleanup Trap ---
# Set trap for termination signals to ensure exit code 1 is used
trap "exit 1" HUP INT TERM

# Define the cleanup actions command string first
# Note: Comments explaining the logic are moved *outside* the command string
# Note: We are NOT cleaning up PYTHON_INSTALL_DIR here to allow potential reuse.
# Add 'rm -rf "${{PYTHON_INSTALL_DIR}}"; \\' inside the single quotes below if you *always* want cleanup.
CLEANUP_COMMAND=' \\
echo "Transferring logs and cleaning up..."; \\
mkdir -p "${{final_job_dir}}"; \\
rsync -av --inplace "${{local_job_dir}}/" "${{final_job_dir}}/"; \\
\\
slurm_out_file="/itet-stor/piroth/net_scratch/outputs/jobs/${{SLURM_JOB_ID}}.out"; \\
slurm_err_file="/itet-stor/piroth/net_scratch/outputs/jobs/${{SLURM_JOB_ID}}.err"; \\
if [[ -f "$slurm_out_file" && -f "$slurm_err_file" ]]; then \\
    if cp "$slurm_out_file" "${{final_job_dir}}/slurm.out" && \\
       cp "$slurm_err_file" "${{final_job_dir}}/slurm.err"; then \\
        echo "SLURM logs copied successfully to ${{final_job_dir}}"; \\
        ln -sfn "${{final_job_dir}}" /home/${{ETH_USERNAME}}/latest_job; \\
        echo "Symlink created/updated for latest job directory"; \\
        rm -f "$slurm_out_file"; \\
        rm -f "$slurm_err_file"; \\
        echo "Original SLURM output and error files deleted"; \\
    else \\
        echo "WARNING: Failed to copy SLURM logs to detailed directory. Original files preserved."; \\
    fi; \\
else \\
    echo "WARNING: Original SLURM logs not found at expected location ($slurm_out_file / $slurm_err_file). Cannot copy or delete them."; \\
fi; \\
\\
echo "All local job files transferred to ${{final_job_dir}}"; \\
echo "Removing local job directory: ${{local_job_dir}}"; \\
rm -rf "${{local_job_dir}}"; \\
echo "Cleanup trap finished."; \\
'
# Set the trap using the command string variable for the EXIT signal
trap "${{CLEANUP_COMMAND}}" EXIT
# --- END CORRECTED Cleanup Trap ---

# Send noteworthy information to both SLURM log and our detailed log
{{
  echo "Running on node: $(hostname)"
  echo "In directory: $(pwd)"
  echo "Starting on: $(date)"
  echo "SLURM_JOB_ID: ${{SLURM_JOB_ID}}"
  echo "Detailed job data will be saved to: ${{final_job_dir}}"
  echo "Local job directory: ${{local_job_dir}}"
  echo "Minimal Python install dir target: ${{PYTHON_INSTALL_DIR}}"
  echo "Attempting to use Python version: ${{TARGET_PYTHON_VERSION_FULL}}"
}} | tee "${{local_job_dir}}/job_info.log"
"""

    if config["env"]["install_python"]:
        sbatch_script_content += f"""
# --- Setup Minimal Python for Subprocesses ---
echo "Checking for minimal Python environment..." | tee -a "${{local_job_dir}}/job_info.log"
INSTALL_PYTHON=true
if [ -f "$MINIMAL_PYTHON_EXEC_PATH" ]; then
    echo "Existing Python executable found at $MINIMAL_PYTHON_EXEC_PATH" | tee -a "${{local_job_dir}}/job_info.log"
    # Check if the version matches (allow for potential build metadata like +...)
    # We compare against the base version used for the directory name
    INSTALLED_VERSION_FULL=$("$MINIMAL_PYTHON_EXEC_PATH" --version 2>&1) # Get full version string (e.g., Python 3.12.9+20250317)
    INSTALLED_VERSION_NUM=$(echo "$INSTALLED_VERSION_FULL" | cut -d' ' -f2) # Extract number (e.g., 3.12.9+20250317)

    echo "Found version string: $INSTALLED_VERSION_FULL (extracted: $INSTALLED_VERSION_NUM)" | tee -a "${{local_job_dir}}/job_info.log"

    # Check if the installed version starts with the target base version
    if [[ "$INSTALLED_VERSION_NUM" == "$TARGET_PYTHON_VERSION_BASE"* ]]; then
        echo "Base version matches (${{TARGET_PYTHON_VERSION_BASE}}). Checking for NumPy." | tee -a "${{local_job_dir}}/job_info.log"
        # Check if numpy is installed
        if "$MINIMAL_PYTHON_EXEC_PATH" -m pip show numpy &> /dev/null; then
             echo "NumPy already installed. Skipping Python setup." | tee -a "${{local_job_dir}}/job_info.log"
             INSTALL_PYTHON=false
        else
             echo "NumPy not found. Will proceed with setup to install it." | tee -a "${{local_job_dir}}/job_info.log"
             # We will try to install numpy into the existing python if version matches
             INSTALL_PYTHON=true # Force setup path, but it might skip download/extract
        fi
    else
        echo "Version mismatch (Found: $INSTALLED_VERSION_NUM, Need base: $TARGET_PYTHON_VERSION_BASE). Re-installing." | tee -a "${{local_job_dir}}/job_info.log"
        # Clean up old version before installing new one
        rm -rf "$PYTHON_INSTALL_DIR"
        INSTALL_PYTHON=true
    fi
else
     echo "Minimal Python executable not found at $MINIMAL_PYTHON_EXEC_PATH. Installing." | tee -a "${{local_job_dir}}/job_info.log"
     INSTALL_PYTHON=true
fi

# Flag to track if download/extraction actually happens
DID_INSTALL_FRESH_PYTHON=false

if [ "$INSTALL_PYTHON" = true ]; then
    # Check if we need to download and extract, or just install numpy
    if [ ! -f "$MINIMAL_PYTHON_EXEC_PATH" ]; then
        DID_INSTALL_FRESH_PYTHON=true
        echo "Setting up minimal Python ${{TARGET_PYTHON_VERSION_FULL}} environment..." | tee -a "${{local_job_dir}}/job_info.log"
        # Ensure the base directory for *all* minimal python versions exists
        mkdir -p "$PYTHON_INSTALL_BASE_DIR"
        if [ $? -ne 0 ]; then
            echo "CRITICAL: Failed to create base install directory ${{PYTHON_INSTALL_BASE_DIR}}." | tee -a "${{local_job_dir}}/job_info.log" >&2
            exit 1
        fi
        # Clean up any potentially existing target directory before installing
        rm -rf "$PYTHON_INSTALL_DIR"

        # Temporary download location within the ephemeral job dir
        TEMP_DOWNLOAD_PATH="${{local_job_dir}}/${{PYTHON_FILENAME}}"

        echo "Downloading Python standalone from $PYTHON_DOWNLOAD_URL..." | tee -a "${{local_job_dir}}/job_info.log"
        # Use wget with timeout and retries for robustness
        wget --quiet --tries=3 --timeout=60 -O "$TEMP_DOWNLOAD_PATH" "$PYTHON_DOWNLOAD_URL"
        if [ $? -ne 0 ]; then
            echo "Error: Failed to download Python from $PYTHON_DOWNLOAD_URL." | tee -a "${{local_job_dir}}/job_info.log" >&2
            echo "Please double-check the URL and filename." >&2
            rm -f "$TEMP_DOWNLOAD_PATH" # Clean up partial download
            exit 1
        fi
        echo "Download successful." | tee -a "${{local_job_dir}}/job_info.log"

        echo "Extracting Python into $PYTHON_INSTALL_BASE_DIR..." | tee -a "${{local_job_dir}}/job_info.log"
        # Extract into the base directory. Expects it creates a 'python' subdir based on user info.
        tar -xzf "$TEMP_DOWNLOAD_PATH" -C "$PYTHON_INSTALL_BASE_DIR"
        EXTRACT_EXIT_CODE=$?
        # Clean up downloaded archive now that extraction is attempted
        rm -f "$TEMP_DOWNLOAD_PATH"
        # Check tar exit code
        if [ $EXTRACT_EXIT_CODE -ne 0 ]; then
            echo "Error: Failed to extract Python archive (tar exit code: $EXTRACT_EXIT_CODE). Download path: $TEMP_DOWNLOAD_PATH. Target base: $PYTHON_INSTALL_BASE_DIR" | tee -a "${{local_job_dir}}/job_info.log" >&2
            exit 1
        fi

        # Check if extraction created the expected 'python' directory in the base directory
        EXTRACTED_PYTHON_DIR="${{PYTHON_INSTALL_BASE_DIR}}/python"
        if [ ! -d "$EXTRACTED_PYTHON_DIR" ]; then
             echo "Error: Expected directory 'python' not found at $EXTRACTED_PYTHON_DIR after extraction." | tee -a "${{local_job_dir}}/job_info.log" >&2
             exit 1
        fi

        # Rename the extracted 'python' folder to the version-specific name (e.g., python-3.12.9)
        echo "Renaming $EXTRACTED_PYTHON_DIR to $PYTHON_INSTALL_DIR..." | tee -a "${{local_job_dir}}/job_info.log"
        mv "$EXTRACTED_PYTHON_DIR" "$PYTHON_INSTALL_DIR"
        if [ $? -ne 0 ]; then
            echo "Error: Failed to rename extracted directory $EXTRACTED_PYTHON_DIR to $PYTHON_INSTALL_DIR." | tee -a "${{local_job_dir}}/job_info.log" >&2
            # Attempt cleanup of the wrongly named directory if rename failed
            rm -rf "$EXTRACTED_PYTHON_DIR"
            exit 1
        fi
        echo "Extraction complete and directory renamed." | tee -a "${{local_job_dir}}/job_info.log"

        # Verify Python installation executable exists after rename
        if [ ! -f "$MINIMAL_PYTHON_EXEC_PATH" ]; then
            echo "Error: Python executable not found after setup at $MINIMAL_PYTHON_EXEC_PATH." | tee -a "${{local_job_dir}}/job_info.log" >&2
            exit 1
        fi
        echo "Python installed successfully at $PYTHON_INSTALL_DIR" | tee -a "${{local_job_dir}}/job_info.log"
        "$MINIMAL_PYTHON_EXEC_PATH" --version # Log the installed version
    else
         echo "Python executable already exists, proceeding directly to NumPy check/install." | tee -a "${{local_job_dir}}/job_info.log"
    fi

    # Install/Verify NumPy (runs if INSTALL_PYTHON is true, even if download was skipped)
    echo "Installing/Verifying NumPy using minimal Python's pip..." | tee -a "${{local_job_dir}}/job_info.log"
    "$MINIMAL_PYTHON_EXEC_PATH" -m pip install --no-cache-dir --upgrade pip # Upgrade pip first
    if [ $? -ne 0 ]; then
        echo "Warning: Failed to upgrade pip for $MINIMAL_PYTHON_EXEC_PATH. Continuing with NumPy install..." | tee -a "${{local_job_dir}}/job_info.log" >&2
    fi
    "$MINIMAL_PYTHON_EXEC_PATH" -m pip install --no-cache-dir numpy
    if [ $? -ne 0 ]; then
        echo "Error: Failed to install NumPy using $MINIMAL_PYTHON_EXEC_PATH." | tee -a "${{local_job_dir}}/job_info.log" >&2
        # If we just installed Python fresh and failed pip, exit. Otherwise maybe continue? Let's exit.
        exit 1
    fi
    echo "NumPy installed/verified." | tee -a "${{local_job_dir}}/job_info.log"
else
    echo "Using existing minimal Python environment found at $PYTHON_INSTALL_DIR" | tee -a "${{local_job_dir}}/job_info.log"
fi

# Export the path for the main application - must always be done after the check/install block
export SUBPROCESS_PYTHON_EXEC="$MINIMAL_PYTHON_EXEC_PATH"
# Final check that the executable exists before proceeding
if [ ! -x "$SUBPROCESS_PYTHON_EXEC" ]; then
    echo "CRITICAL Error: Subprocess Python executable is not found or not executable at $SUBPROCESS_PYTHON_EXEC after setup attempts." | tee -a "${{local_job_dir}}/job_info.log" >&2
    exit 1
fi
echo "Subprocess Python executable path set to: $SUBPROCESS_PYTHON_EXEC" | tee -a "${{local_job_dir}}/job_info.log"
"""

    sbatch_script_content += f"""
# --- Activate Main Conda Environment and Run ---
echo "Activating main Conda environment: ${{CONDA_ENVIRONMENT}}" | tee -a "${{local_job_dir}}/job_info.log"
# Ensure the conda path is correct for your setup
[[ -f /itet-stor/${{ETH_USERNAME}}/net_scratch/conda/bin/conda ]] && eval "$(/itet-stor/${{ETH_USERNAME}}/net_scratch/conda/bin/conda shell.bash hook)"
conda activate ${{CONDA_ENVIRONMENT}}
echo "Conda activated." | tee -a "${{local_job_dir}}/job_info.log"

echo "Changing to project directory: ${{DIRECTORY}}" | tee -a "${{local_job_dir}}/job_info.log"
cd ${{DIRECTORY}}

export HF_CACHE_DIR="${{NET_SCRATCH_PATH}}/.cache/huggingface"
export HF_HOME="${{NET_SCRATCH_PATH}}/.cache/huggingface"
export HF_DATASETS_CACHE="${{NET_SCRATCH_PATH}}/.cache/huggingface/datasets"

# Execute the Python application with output redirected to local scratch
echo "Running: python {config["script"]} {f"--config-file {config['yaml_config']}" if config['yaml_config'] != 'none' else ""} {config['script_args']} | tee -a "${{local_job_dir}}/job_info.log"

# Setting relevant environment variables for the main application
export VLLM_LOGGING_LEVEL=DEBUG
export TOKENIZERS_PARALLELISM=true

# Run the program with output going to local scratch
# The main program will use the SUBPROCESS_PYTHON_EXEC environment variable internally
python {config["script"]} {f"--config-file {config['yaml_config']}" if config['yaml_config'] != 'none' else ""} {config['script_args']} > "${{local_job_dir}}/program_output.log" 2> "${{local_job_dir}}/program_error.log"
EXIT_CODE=$?

# Send completion information to both SLURM log and our detailed log
{{
  echo "Program completed with exit code: ${{EXIT_CODE}}"
  if [ ${{EXIT_CODE}} -ne 0 ]; then
    echo "FAILURE: Program exited with non-zero code"
  fi
  echo "Finished at: $(date)"
  echo "Job data available at: ${{final_job_dir}}"
}} | tee -a "${{local_job_dir}}/job_info.log"

# The EXIT trap will automatically transfer logs and clean up local_job_dir
# End the script with the same exit code as the main program
exit ${{EXIT_CODE}}
"""

    # --- Submit the Job ---
    print("\n--- Generated sbatch script ---")
    print(sbatch_script_content)
    print("-----------------------------")

    submit_job = ask_question("confirm", "Submit this job to SLURM?", default=True)

    if submit_job:
        try:
            # Use a temporary file for the sbatch script
            with tempfile.NamedTemporaryFile(mode="w", suffix=".sh", delete=False) as temp_script:
                temp_script.write(sbatch_script_content)
                temp_script_path = temp_script.name

            print(f"Submitting job using temporary script: {temp_script_path}")
            result = subprocess.run(["sbatch", temp_script_path], capture_output=True, text=True, check=True)
            print("SLURM Output:\n", result.stdout)
            print(f"Job submitted successfully!")

        except FileNotFoundError:
            print("Error: 'sbatch' command not found. Make sure SLURM is installed and in your PATH.")
        except subprocess.CalledProcessError as e:
            print(f"Error submitting job to SLURM.")
            print("Command:", e.cmd)
            print("Return Code:", e.returncode)
            print("Output:\n", e.stdout)
            print("Error:\n", e.stderr)
        except Exception as e:
            print(f"An unexpected error occurred during submission: {e}")
        finally:
            # Clean up the temporary script file
            if 'temp_script_path' in locals() and os.path.exists(temp_script_path):
                try:
                    os.remove(temp_script_path)
                    print(f"Temporary script {temp_script_path} deleted.")
                except OSError as e:
                    print(f"Warning: Could not delete temporary script {temp_script_path}: {e}")
    else:
        print("Job submission cancelled.")


if __name__ == "__main__":
    main()
