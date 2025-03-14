#!/bin/bash
#SBATCH --mail-type=NONE # mail configuration: NONE, BEGIN, END, FAIL, REQUEUE, ALL
#SBATCH --output=/itet-stor/piroth/net_scratch/outputs/jobs/%j.out # where to store the output (%j is the JOBID)
#SBATCH --error=/itet-stor/piroth/net_scratch/outputs/jobs/%j.err # where to store error messages
#SBATCH --mem=48G
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

# Allow specifying a different config file as the only CLI argument
# Usage: ./simple_run.sh [config_file]
if [ $# -eq 1 ]; then
  CONFIG_FILE=$1
fi

# Send noteworthy information to the output log
echo "Running on node: $(hostname)"
echo "In directory: $(pwd)"
echo "Starting on: $(date)"
echo "SLURM_JOB_ID: ${SLURM_JOB_ID}"
echo "Using config file: ${CONFIG_FILE}"

# Activate conda environment
[[ -f /itet-stor/${ETH_USERNAME}/net_scratch/conda/bin/conda ]] && eval "$(/itet-stor/${ETH_USERNAME}/net_scratch/conda/bin/conda shell.bash hook)"
conda activate ${CONDA_ENVIRONMENT}
echo "Conda activated"
cd ${DIRECTORY}

# Execute the Python application
echo "Running: python main.py --config-file ${CONFIG_FILE}"
python main.py --config-file ${CONFIG_FILE}

# Send more noteworthy information to the output log
echo "Finished at: $(date)"

# End the script with exit code 0
exit 0
