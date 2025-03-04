#!/bin/bash

# Default SLURM resource values
MEM="20G"
CPUS=4
GPUS=2
PARTITION=""  # Default partition (empty means use the default)
EXCLUDE="tikgpu08,tikgpu10"  # Exclude these nodes by default
NODE_LIST=""  # No specific nodes by default
TIME_LIMIT="" # No time limit by default

# Default application parameters
VERSION=1  # Default to mark1.py
TASK_INDEX=1
MAX_ITERATIONS=5
MODEL="Qwen/Qwen2.5-Coder-1.5B-Instruct"
EVAL=false
HINT=""
VERBOSE=true
DTYPE="float16" # bfloat16 only supported in compute 8.0 and above otherwise use float16

# Parse named command line arguments
while [[ $# -gt 0 ]]; do
  case "$1" in
    --version=*)
      VERSION="${1#*=}"
      shift
      ;;
    --mem=*)
      MEM="${1#*=}"
      shift
      ;;
    --cpus=*)
      CPUS="${1#*=}"
      shift
      ;;
    --gpus=*)
      GPUS="${1#*=}"
      shift
      ;;
    --partition=*)
      PARTITION="${1#*=}"
      shift
      ;;
    --exclude=*)
      EXCLUDE="${1#*=}"
      shift
      ;;
    --nodelist=*)
      NODE_LIST="${1#*=}"
      shift
      ;;
    --time=*)
      TIME_LIMIT="${1#*=}"
      shift
      ;;
    --task=*)
      TASK_INDEX="${1#*=}"
      shift
      ;;
    --iter=*)
      MAX_ITERATIONS="${1#*=}"
      shift
      ;;
    --model=*)
      MODEL="${1#*=}"
      shift
      ;;
    --hint=*)
      HINT="${1#*=}"
      shift
      ;;
    --dtype=*)
      DTYPE="${1#*=}"
      shift
      ;;
    --eval)
      EVAL=true  # Corrected: Set to true when flag is present
      shift
      ;;
    --verbose)
      VERBOSE=true
      shift
      ;;
    *)
      echo "Unknown parameter: $1"
      echo "Usage: $0 [--version=1] [--mem=20G] [--cpus=4] [--gpus=1] [--partition=partition_name]"
      echo "  [--exclude=node1,node2] [--nodelist=node1,node2] [--time=HH:MM:SS]"
      echo "  [--task=1] [--iter=5] [--model=Qwen/Qwen2.5-Coder-1.5B-Instruct] [--hint=\"your hint\"] "
      echo "  [--eval] [--verbose] [--dtype=float16]"
      exit 1
      ;;
  esac
done

# Create temporary SBATCH script with the specified parameters
TEMP_SCRIPT=$(mktemp)
cat > "${TEMP_SCRIPT}" << EOL
#!/bin/bash
#SBATCH --mail-type=NONE
#SBATCH --output=/itet-stor/${USER}/net_scratch/outputs/jobs/%j.out
#SBATCH --error=/itet-stor/${USER}/net_scratch/outputs/jobs/%j.err
#SBATCH --mem=${MEM}
#SBATCH --nodes=1
#SBATCH --cpus-per-task=${CPUS}
#SBATCH --gres=gpu:${GPUS}

EOL

# Add optional SBATCH parameters if provided
if [[ ! -z "${PARTITION}" ]]; then
  echo "#SBATCH --partition=${PARTITION}" >> "${TEMP_SCRIPT}"
fi

if [[ ! -z "${EXCLUDE}" ]]; then
  echo "#SBATCH --exclude=${EXCLUDE}" >> "${TEMP_SCRIPT}"
fi

if [[ ! -z "${NODE_LIST}" ]]; then
  echo "#SBATCH --nodelist=${NODE_LIST}" >> "${TEMP_SCRIPT}"
fi

if [[ ! -z "${TIME_LIMIT}" ]]; then
  echo "#SBATCH --time=${TIME_LIMIT}" >> "${TEMP_SCRIPT}"
fi

# Add the rest of the script
cat >> "${TEMP_SCRIPT}" << EOL

# Set environment variables
ETH_USERNAME=${USER}
PROJECT_DIR=/itet-stor/\${ETH_USERNAME}/net_scratch/rstar-arc
CONDA_ENV=arc-solver
OUTPUT_DIR=/itet-stor/\${ETH_USERNAME}/net_scratch/outputs

# Exit on errors
set -o errexit

# Create jobs directory if it doesn't exist
mkdir -p /itet-stor/\${ETH_USERNAME}/net_scratch/outputs/jobs

# Log basic information and parameters
echo "===== Job Configuration ====="
echo "Running on node: \$(hostname)"
echo "In directory: \$(pwd)"
echo "Starting on: \$(date)"
echo "SLURM_JOB_ID: \${SLURM_JOB_ID}"
echo ""
echo "===== Resource Parameters ====="
echo "Memory: ${MEM}"
echo "CPUs: ${CPUS}"
echo "GPUs: ${GPUS}"
if [[ ! -z "${PARTITION}" ]]; then echo "Partition: ${PARTITION}"; fi
if [[ ! -z "${EXCLUDE}" ]]; then echo "Excluded nodes: ${EXCLUDE}"; fi
if [[ ! -z "${NODE_LIST}" ]]; then echo "Node list: ${NODE_LIST}"; fi
if [[ ! -z "${TIME_LIMIT}" ]]; then echo "Time limit: ${TIME_LIMIT}"; fi
echo ""
echo "===== Application Parameters ====="
echo "Version: mark${VERSION}.py"
echo "Task Index: ${TASK_INDEX}"
echo "Max Iterations: ${MAX_ITERATIONS}"
echo "Output Directory: \${OUTPUT_DIR}"
echo "LLM Model: ${MODEL}"
echo "Model dtype: ${DTYPE}"
if [[ ! -z "${HINT}" ]]; then echo "Hint: ${HINT}"; fi

# Activate conda
eval "\$(/itet-stor/\${ETH_USERNAME}/net_scratch/conda/bin/conda shell.bash hook)"
conda activate \${CONDA_ENV}
echo "Conda environment activated"

# Build command with all parameters
CMD="python \${PROJECT_DIR}/arc_baseline/mark${VERSION}.py --task-index=${TASK_INDEX} --max-iterations=${MAX_ITERATIONS} --output-dir=\${OUTPUT_DIR} --model='${MODEL}' --gpus=${GPUS} --dtype=${DTYPE}"

# Add optional parameters
if [ ! -z "${HINT}" ]; then
    CMD="\${CMD} --hint=\"${HINT}\""
fi

if ${EVAL}; then
    CMD="\${CMD} --eval"
fi

if ${VERBOSE}; then
    CMD="\${CMD} --verbose"
fi

# Execute the command
echo "Executing: \${CMD}"
cd \${PROJECT_DIR}
eval \${CMD}

# Cleanup
echo "Finished at: \$(date)"
exit 0
EOL

# Make the temporary script executable
chmod +x "${TEMP_SCRIPT}"

# Execute the script with sbatch and pass through any additional parameters
sbatch "${TEMP_SCRIPT}"

# Remove the temporary script
rm "${TEMP_SCRIPT}"
