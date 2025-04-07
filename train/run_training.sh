#!/bin/bash
#SBATCH --mail-type=NONE # mail configuration: NONE, BEGIN, END, FAIL, REQUEUE, ALL
#SBATCH --output=/itet-stor/piroth/net_scratch/outputs/jobs/%j.out # Keep minimal SLURM logging
#SBATCH --error=/itet-stor/piroth/net_scratch/outputs/jobs/%j.err # Keep minimal SLURM logging
#SBATCH --mem=64G
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --constraint='geforce_rtx_3090'


# Default application parameters
ETH_USERNAME=piroth
PROJECT_NAME=rstar-arc
DIRECTORY=/home/${ETH_USERNAME}/${PROJECT_NAME}/train
CONDA_ENVIRONMENT=arc-solver


# Exit on errors
set -o errexit
set -o pipefail  # Fails if any command in a pipe fails
set -o nounset   # Fails if undefined variables are used

# --- Ensure Base Scratch Directory Exists ---
echo "Ensuring base scratch directory exists: /scratch/${ETH_USERNAME}"
mkdir -p "/scratch/${ETH_USERNAME}"
if [ $? -ne 0 ]; then
    echo "CRITICAL: Failed to create or access base scratch directory /scratch/${ETH_USERNAME}. Check permissions." >&2
    exit 1
fi
echo "Base scratch directory OK."

# Set up local scratch directories for job output and models
local_job_dir="/scratch/${ETH_USERNAME}/job_${SLURM_JOB_ID}" # Node-local ephemeral job data
final_job_dir="/itet-stor/${ETH_USERNAME}/net_scratch/outputs/detailed_logs/job_${SLURM_JOB_ID}" # Persistent logs

# Create local scratch job directory
if ! mkdir -p "${local_job_dir}"; then
    echo "Failed to create local scratch job directory: ${local_job_dir}" >&2
    exit 1
fi

# Create net scratch job directory for final logs
if ! mkdir -p "${final_job_dir}"; then
    echo "Failed to create net scratch directory: ${final_job_dir}" >&2
    exit 1
fi


# --- CORRECTED Cleanup Trap ---
# Set trap for termination signals to ensure exit code 1 is used
trap "exit 1" HUP INT TERM

# Define the cleanup actions command string first
# Note: Comments explaining the logic are moved *outside* the command string
# Note: We are NOT cleaning up PYTHON_INSTALL_DIR here to allow potential reuse.
# Add 'rm -rf "${PYTHON_INSTALL_DIR}"; \' inside the single quotes below if you *always* want cleanup.
CLEANUP_COMMAND=' \
echo "Transferring logs and cleaning up..."; \
mkdir -p "${final_job_dir}"; \
rsync -av --inplace "${local_job_dir}/" "${final_job_dir}/"; \
\
slurm_out_file="/itet-stor/piroth/net_scratch/outputs/jobs/${SLURM_JOB_ID}.out"; \
slurm_err_file="/itet-stor/piroth/net_scratch/outputs/jobs/${SLURM_JOB_ID}.err"; \
if [[ -f "$slurm_out_file" && -f "$slurm_err_file" ]]; then \
    if cp "$slurm_out_file" "${final_job_dir}/slurm.out" && \
       cp "$slurm_err_file" "${final_job_dir}/slurm.err"; then \
        echo "SLURM logs copied successfully to ${final_job_dir}"; \
        ln -sfn "${final_job_dir}" /home/${ETH_USERNAME}/latest_job; \
        echo "Symlink created/updated for latest job directory"; \
        rm -f "$slurm_out_file"; \
        rm -f "$slurm_err_file"; \
        echo "Original SLURM output and error files deleted"; \
    else \
        echo "WARNING: Failed to copy SLURM logs to detailed directory. Original files preserved."; \
    fi; \
else \
    echo "WARNING: Original SLURM logs not found at expected location ($slurm_out_file / $slurm_err_file). Cannot copy or delete them."; \
fi; \
\
echo "All local job files transferred to ${final_job_dir}"; \
echo "Removing local job directory: ${local_job_dir}"; \
rm -rf "${local_job_dir}"; \
echo "Cleanup trap finished."; \
'

# Set the trap using the command string variable for the EXIT signal
trap "${CLEANUP_COMMAND}" EXIT
# --- END CORRECTED Cleanup Trap ---


# Send noteworthy information to both SLURM log and our detailed log
{
  echo "Running on node: $(hostname)"
  echo "In directory: $(pwd)"
  echo "Starting on: $(date)"
  echo "SLURM_JOB_ID: ${SLURM_JOB_ID}"
  echo "Detailed job data will be saved to: ${final_job_dir}"
  echo "Local job directory: ${local_job_dir}"
} | tee "${local_job_dir}/job_info.log"


# --- Activate Main Conda Environment and Run ---
echo "Activating main Conda environment: ${CONDA_ENVIRONMENT}" | tee -a "${local_job_dir}/job_info.log"
# Ensure the conda path is correct for your setup
[[ -f /itet-stor/${ETH_USERNAME}/net_scratch/conda/bin/conda ]] && eval "$(/itet-stor/${ETH_USERNAME}/net_scratch/conda/bin/conda shell.bash hook)"
conda activate ${CONDA_ENVIRONMENT}
echo "Conda activated." | tee -a "${local_job_dir}/job_info.log"

echo "Changing to project directory: ${DIRECTORY}" | tee -a "${local_job_dir}/job_info.log"
cd ${DIRECTORY}

# Execute the Python application with output redirected to local scratch
echo "Running: python train_policy.py" | tee -a "${local_job_dir}/job_info.log"

# Run the program with output going to local scratch
# The main program will use the SUBPROCESS_PYTHON_EXEC environment variable internally
python train_policy.py > "${local_job_dir}/program_output.log" 2> "${local_job_dir}/program_error.log"
EXIT_CODE=$?

# Send completion information to both SLURM log and our detailed log
{
  echo "Program completed with exit code: ${EXIT_CODE}"
  if [ ${EXIT_CODE} -ne 0 ]; then
    echo "FAILURE: Program exited with non-zero code"
  fi
  echo "Finished at: $(date)"
  echo "Job data available at: ${final_job_dir}"
} | tee -a "${local_job_dir}/job_info.log"

# The EXIT trap will automatically transfer logs and clean up local_job_dir
# End the script with the same exit code as the main program
exit ${EXIT_CODE}
