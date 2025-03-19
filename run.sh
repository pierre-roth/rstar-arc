#!/bin/bash
#SBATCH --mail-type=NONE # mail configuration: NONE, BEGIN, END, FAIL, REQUEUE, ALL
#SBATCH --output=/itet-stor/piroth/net_scratch/outputs/jobs/%j.out # Keep minimal SLURM logging
#SBATCH --error=/itet-stor/piroth/net_scratch/outputs/jobs/%j.err # Keep minimal SLURM logging
#SBATCH --mem=60G
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --constraint='geforce_rtx_3090'
##SBATCH --exclude=tikgpu10,tikgpu[06-09]
##SBATCH --nodelist=tikgpu01
##SBATCH --partition=gpu
##SBATCH --time=24:00:00

# Default application parameters
ETH_USERNAME=piroth
PROJECT_NAME=rstar-arc
DIRECTORY=/itet-stor/${ETH_USERNAME}/net_scratch/${PROJECT_NAME}
CONDA_ENVIRONMENT=arc-solver
CONFIG_FILE="basic_bs.yaml"

# Exit on errors
set -o errexit

# Set up local scratch for detailed logging
local_log_dir="/scratch/${ETH_USERNAME}/log_${SLURM_JOB_ID}"
final_log_dir="/itet-stor/${ETH_USERNAME}/net_scratch/outputs/detailed_logs/job_${SLURM_JOB_ID}"

# Create local scratch directory
if ! mkdir -p "${local_log_dir}"; then
    echo "Failed to create local scratch directory" >&2
    exit 1
fi

# Create net scratch directory
if ! mkdir -p "${final_log_dir}"; then
    echo "Failed to create net scratch directory" >&2
    exit 1
fi



# Set up automatic cleanup when job ends
trap "exit 1" HUP INT TERM
trap 'echo "Transferring logs and cleaning up...";
      mkdir -p "${final_log_dir}";
      rsync -av --inplace "${local_log_dir}"/ "${final_log_dir}"/;
      echo "Logs transferred to ${final_log_dir}";
      rm -rf "${local_log_dir}"' EXIT

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
  echo "Detailed logs will be saved to: ${final_log_dir}"
} | tee "${local_log_dir}/job_info.log"

# Activate conda environment
[[ -f /itet-stor/${ETH_USERNAME}/net_scratch/conda/bin/conda ]] && eval "$(/itet-stor/${ETH_USERNAME}/net_scratch/conda/bin/conda shell.bash hook)"
conda activate ${CONDA_ENVIRONMENT}
echo "Conda activated" | tee -a "${local_log_dir}/job_info.log"
cd ${DIRECTORY}

# Execute the Python application with output redirected to local scratch
echo "Running: python main.py --config-file ${CONFIG_FILE}" | tee -a "${local_log_dir}/job_info.log"

# setting relevant environment variables
export VLLM_LOGGING_LEVEL=DEBUG

# Run the program with output going to local scratch
python main.py --config-file ${CONFIG_FILE} > "${local_log_dir}/program_output.log" 2> "${local_log_dir}/program_error.log"

# Send completion information to both SLURM log and our detailed log
{
  echo "Program completed with exit code: $?"
  echo "Finished at: $(date)"
  echo "Logs will be available at: ${final_log_dir}"
} | tee -a "${local_log_dir}/job_info.log"

# The EXIT trap will automatically transfer logs and clean up
# End the script with exit code 0
exit 0
