#!/bin/bash

# Default SLURM parameters
MEM="32G"
CPUS=4
NODES=1
GPU=1
CONSTRAINT="geforce_rtx_3090"
EXCLUDE=""
NODELIST=""
PARTITION=""
TIME=""

# Default application parameters
ETH_USERNAME=piroth
PROJECT_NAME=rstar-arc
DIRECTORY=/itet-stor/${ETH_USERNAME}/net_scratch/${PROJECT_NAME}
CONDA_ENVIRONMENT=arc-solver
CONFIG_FILE="basic_bs.yaml"
PYTHON_ARGS=""

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --mem=*)
      MEM="${1#*=}"
      shift
      ;;
    --cpus=*)
      CPUS="${1#*=}"
      shift
      ;;
    --nodes=*)
      NODES="${1#*=}"
      shift
      ;;
    --gpu=*)
      GPU="${1#*=}"
      shift
      ;;
    --constraint=*)
      CONSTRAINT="${1#*=}"
      shift
      ;;
    --exclude=*)
      EXCLUDE="${1#*=}"
      shift
      ;;
    --nodelist=*)
      NODELIST="${1#*=}"
      shift
      ;;
    --partition=*)
      PARTITION="${1#*=}"
      shift
      ;;
    --time=*)
      TIME="${1#*=}"
      shift
      ;;
    --config=*)
      CONFIG_FILE="${1#*=}"
      shift
      ;;
    --python-args=*)
      PYTHON_ARGS="${1#*=}"
      shift
      ;;
    *)
      echo "Unknown parameter: $1"
      exit 1
      ;;
  esac
done

# Apply SLURM parameters
#SBATCH --mail-type=NONE # mail configuration: NONE, BEGIN, END, FAIL, REQUEUE, ALL
#SBATCH --output=/itet-stor/${ETH_USERNAME}/net_scratch/outputs/jobs/%j.out # where to store the output (%j is the JOBID)
#SBATCH --error=/itet-stor/${ETH_USERNAME}/net_scratch/outputs/jobs/%j.err # where to store error messages
#SBATCH --mem=${MEM}
#SBATCH --nodes=${NODES}
#SBATCH --cpus-per-task=${CPUS}
#SBATCH --gres=gpu:${GPU}
#SBATCH --constraint='${CONSTRAINT}'

# Conditionally add optional SLURM parameters
if [ ! -z "$EXCLUDE" ]; then
  #SBATCH --exclude=${EXCLUDE}
  SLURM_EXCLUDE="--exclude=${EXCLUDE}"
fi

if [ ! -z "$NODELIST" ]; then
  #SBATCH --nodelist=${NODELIST}
  SLURM_NODELIST="--nodelist=${NODELIST}"
fi

if [ ! -z "$PARTITION" ]; then
  #SBATCH --partition=${PARTITION}
  SLURM_PARTITION="--partition=${PARTITION}"
fi

if [ ! -z "$TIME" ]; then
  #SBATCH --time=${TIME}
  SLURM_TIME="--time=${TIME}"
fi

# Exit on errors
set -o errexit

# Send some noteworthy information to the output log
echo "Running on node: $(hostname)"
echo "In directory: $(pwd)"
echo "Starting on: $(date)"
echo "SLURM_JOB_ID: ${SLURM_JOB_ID}"

# Build SLURM parameter string to pass to Python
SLURM_PARAMS=""
[ ! -z "$MEM" ] && SLURM_PARAMS="$SLURM_PARAMS --mem=$MEM"
[ ! -z "$CPUS" ] && SLURM_PARAMS="$SLURM_PARAMS --cpus=$CPUS"
[ ! -z "$PARTITION" ] && SLURM_PARAMS="$SLURM_PARAMS --partition=$PARTITION"
[ ! -z "$EXCLUDE" ] && SLURM_PARAMS="$SLURM_PARAMS --exclude=$EXCLUDE"
[ ! -z "$NODELIST" ] && SLURM_PARAMS="$SLURM_PARAMS --nodelist=$NODELIST"
[ ! -z "$CONSTRAINT" ] && SLURM_PARAMS="$SLURM_PARAMS --constraint=$CONSTRAINT"
[ ! -z "$TIME" ] && SLURM_PARAMS="$SLURM_PARAMS --time=$TIME"

# Activate conda environment
[[ -f /itet-stor/${ETH_USERNAME}/net_scratch/conda/bin/conda ]] && eval "$(/itet-stor/${ETH_USERNAME}/net_scratch/conda/bin/conda shell.bash hook)"
conda activate ${CONDA_ENVIRONMENT}
echo "Conda activated"
cd ${DIRECTORY}

# Execute your code with all parameters
echo "Running: python main.py --config ${CONFIG_FILE} ${SLURM_PARAMS} ${PYTHON_ARGS}"
python main.py --config ${CONFIG_FILE} ${SLURM_PARAMS} ${PYTHON_ARGS}

# Send more noteworthy information to the output log
echo "Finished at: $(date)"

# End the script with exit code 0
exit 0
