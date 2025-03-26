#!/bin/bash
#SBATCH --mail-type=NONE # mail configuration: NONE, BEGIN, END, FAIL, REQUEUE, ALL
#SBATCH --output=/itet-stor/piroth/net_scratch/outputs/jobs/%j.out # Keep minimal SLURM logging
#SBATCH --error=/itet-stor/piroth/net_scratch/outputs/jobs/%j.err # Keep minimal SLURM logging
#SBATCH --mem=32G
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
##SBATCH --constraint='geforce_rtx_3090'
##SBATCH --exclude=tikgpu10,tikgpu[06-09]
##SBATCH --nodelist=tikgpu01
##SBATCH --partition=gpu
##SBATCH --time=24:00:00

## GPU names: geforce_rtx_3090,rtx_a6000,a100

# Default application parameters
ETH_USERNAME=piroth
PROJECT_NAME=rstar-arc
DIRECTORY=/home/${ETH_USERNAME}/${PROJECT_NAME}
CONDA_ENVIRONMENT=arc-solver
CONFIG_FILE="basic_bs.yaml"

# Exit on errors
set -o errexit
set -o pipefail  # Fails if any command in a pipe fails
set -o nounset   # Fails if undefined variables are used

# Set up local scratch directories
local_models_dir="/scratch/${ETH_USERNAME}/models"
local_job_dir="/scratch/${ETH_USERNAME}/job_${SLURM_JOB_ID}"
final_job_dir="/itet-stor/${ETH_USERNAME}/net_scratch/outputs/detailed_logs/job_${SLURM_JOB_ID}"

# Create local scratch model directory
if ! mkdir -p "${local_models_dir}"; then
    echo "Failed to create local scratch model directory" >&2
    exit 1
fi

# Create local scratch job directory
if ! mkdir -p "${local_job_dir}"; then
    echo "Failed to create local scratch directory" >&2
    exit 1
fi

# Create net scratch job directory
if ! mkdir -p "${final_job_dir}"; then
    echo "Failed to create net scratch directory" >&2
    exit 1
fi

# Set up automatic cleanup when job ends
trap "exit 1" HUP INT TERM
trap 'echo "Transferring logs and cleaning up...";
      mkdir -p "${final_job_dir}";
      rsync -av --inplace "${local_job_dir}"/ "${final_job_dir}"/;

      # Copy SLURM output files to detailed logs and delete originals if copy succeeds
      if cp /itet-stor/piroth/net_scratch/outputs/jobs/${SLURM_JOB_ID}.out "${final_job_dir}/slurm.out" &&
         cp /itet-stor/piroth/net_scratch/outputs/jobs/${SLURM_JOB_ID}.err "${final_job_dir}/slurm.err"; then
          echo "SLURM logs copied successfully to ${final_job_dir}";

          ln -sfn "${final_job_dir}" /home/${ETH_USERNAME}/latest_job;
          echo "Symlink created for latest job directory";

          rm -f /itet-stor/piroth/net_scratch/outputs/jobs/${SLURM_JOB_ID}.out;
          rm -f /itet-stor/piroth/net_scratch/outputs/jobs/${SLURM_JOB_ID}.err;
          echo "Original SLURM output and error files deleted";
      else
          echo "WARNING: Failed to copy SLURM logs to detailed directory. Original files preserved.";
      fi;

      echo "All files transferred to ${final_job_dir}";
      rm -rf "${local_job_dir}"' EXIT

# Allow specifying a different config file as the only CLI argument
# Usage: ./run.sh [config_file]
if [ $# -eq 1 ]; then
  CONFIG_FILE=$1
fi

# Send noteworthy information to both SLURM log and our detailed log
{
  echo "Running on node: $(hostname)"
  echo "In directory: $(pwd)"
  echo "Starting on: $(date)"
  echo "SLURM_JOB_ID: ${SLURM_JOB_ID}"
  echo "Using config file: ${CONFIG_FILE}"
  echo "Detailed job data will be saved to: ${final_job_dir}"
} | tee "${local_job_dir}/job_info.log"


# Activate conda environment
[[ -f /itet-stor/${ETH_USERNAME}/net_scratch/conda/bin/conda ]] && eval "$(/itet-stor/${ETH_USERNAME}/net_scratch/conda/bin/conda shell.bash hook)"
conda activate ${CONDA_ENVIRONMENT}
echo "Conda activated" | tee -a "${local_job_dir}/job_info.log"
cd ${DIRECTORY}

# Execute the Python application with output redirected to local scratch
echo "Running: python main.py --config-file ${CONFIG_FILE}" | tee -a "${local_job_dir}/job_info.log"

# Setting relevant environment variables
export VLLM_LOGGING_LEVEL=DEBUG
export TOKENIZERS_PARALLELISM=true

# Run the program with output going to local scratch
python main.py --config-file "${CONFIG_FILE}" > "${local_job_dir}/program_output.log" 2> "${local_job_dir}/program_error.log"
EXIT_CODE=$?

# Send completion information to both SLURM log and our detailed log
{
  echo "Program completed with exit code: ${EXIT_CODE}"
  if [ ${EXIT_CODE} -ne 0 ]; then
    echo "FAILURE: Program exited with non-zero code"
  fi
  echo "Finished at: $(date)"
  echo "Job data will be available at: ${final_job_dir}"
} | tee -a "${local_job_dir}/job_info.log"

# The EXIT trap will automatically transfer logs and clean up
# End the script with the same exit code as the main program
exit ${EXIT_CODE}
