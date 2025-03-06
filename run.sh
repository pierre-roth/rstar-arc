#!/bin/bash

# Default SLURM resource values
MEM="32G"
CPUS=4
GPUS=1
PARTITION=""  # Default partition (empty means use the default)
EXCLUDE=""  # Exclude these nodes by default
NODE_LIST=""  # No specific nodes by default
TIME_LIMIT="" # No time limit by default

# Default application parameters
TASK_INDEX=1
MAX_ITERATIONS=5
TASK_NAME=""
POLICY_MODEL="Qwen/Qwen2.5-Coder-7B-Instruct"
PP_MODEL="Qwen/Qwen2.5-Coder-7B-Instruct"
EVAL=false
HINT=""
VERBOSE=true
DTYPE="bfloat16" # bfloat16 only supported in compute 8.0 and above otherwise use float16
OUTPUT_DIR=""
SEARCH_MODE="beam_search"
MAX_DEPTH=10
TEMPERATURE=0.0
SEED=42
ALL_TASKS=false
CONFIG_FILE=""
MAX_TOKENS=2048
DATA_FOLDER=false
DETERMINISTIC=false
BEAM_WIDTH=3

# Parse named command line arguments
while [[ $# -gt 0 ]]; do
  case "$1" in
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
    --task-name=*)
      TASK_NAME="${1#*=}"
      shift
      ;;
    --iter=*)
      MAX_ITERATIONS="${1#*=}"
      shift
      ;;
    --policy-model=*)
      POLICY_MODEL="${1#*=}"
      shift
      ;;
    --pp-model=*)
      PP_MODEL="${1#*=}"
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
    --output-dir=*)
      OUTPUT_DIR="${1#*=}"
      shift
      ;;
    --search-mode=*)
      SEARCH_MODE="${1#*=}"
      shift
      ;;
    --max-depth=*)
      MAX_DEPTH="${1#*=}"
      shift
      ;;
    --temperature=*)
      TEMPERATURE="${1#*=}"
      shift
      ;;
    --seed=*)
      SEED="${1#*=}"
      shift
      ;;
    --max-tokens=*)
      MAX_TOKENS="${1#*=}"
      shift
      ;;
    --beam-width=*)
      BEAM_WIDTH="${1#*=}"
      shift
      ;;
    --config-file=*)
      CONFIG_FILE="${1#*=}"
      shift
      ;;
    --eval)
      EVAL=true
      shift
      ;;
    --verbose)
      VERBOSE=true
      shift
      ;;
    --all-tasks)
      ALL_TASKS=true
      shift
      ;;
    --data-folder)
      DATA_FOLDER=true
      shift
      ;;
    --deterministic)
      DETERMINISTIC=true
      shift
      ;;
    *)
      echo "Unknown parameter: $1"
      echo "Usage: $0 [--mem=32G] [--cpus=4] [--gpus=1] [--partition=partition_name]"
      echo "  [--exclude=node1,node2] [--nodelist=node1,node2] [--time=HH:MM:SS]"
      echo "  [--task=1] [--task-name=task_name] [--iter=5] [--max-depth=10]"
      echo "  [--policy-model=model_name] [--pp-model=model_name] [--hint=\"your hint\"]"
      echo "  [--search-mode=beam_search] [--temperature=0.0] [--seed=42] [--max-tokens=2048]"
      echo "  [--beam-width=3] [--output-dir=output_dir] [--config-file=config_file] [--all-tasks]"
      echo "  [--data-folder] [--eval] [--verbose] [--dtype=float16] [--deterministic]"
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
#SBATCH --constraint='geforce_rtx_3090'
EOL

# GPU names: geforce_rtx_3090,rtx_a6000,a100

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
CMD="python \${PROJECT_DIR}/main.py --task-index=${TASK_INDEX} --max-iterations=${MAX_ITERATIONS} --policy-model='${POLICY_MODEL}' --pp-model='${PP_MODEL}' --max-depth=${MAX_DEPTH} --gpus=${GPUS} --dtype=${DTYPE} --max-tokens=${MAX_TOKENS} --temperature=${TEMPERATURE} --seed=${SEED} --search-mode=${SEARCH_MODE} --beam-width=${BEAM_WIDTH}"

# Add optional parameters
if [ ! -z "${TASK_NAME}" ]; then
    CMD="\${CMD} --task-name=${TASK_NAME}"
fi

if [ ! -z "${HINT}" ]; then
    CMD="\${CMD} --hint=\"${HINT}\""
fi

if [ ! -z "${OUTPUT_DIR}" ]; then
    CMD="\${CMD} --output-dir=${OUTPUT_DIR}"
else
    CMD="\${CMD} --output-dir=\${OUTPUT_DIR}"
fi

if [ ! -z "${CONFIG_FILE}" ]; then
    CMD="\${CMD} --config-file=${CONFIG_FILE}"
fi

if ${EVAL}; then
    CMD="\${CMD} --eval"
fi

if ${VERBOSE}; then
    CMD="\${CMD} --verbose"
fi

if ${ALL_TASKS}; then
    CMD="\${CMD} --all-tasks"
fi

if ${DATA_FOLDER}; then
    CMD="\${CMD} --data-folder"
fi

if ${DETERMINISTIC}; then
    CMD="\${CMD} --deterministic"
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

