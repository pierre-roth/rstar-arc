#!/bin/bash

# Import parameter schema
source <(python3 -c "
import sys
from schema import PARAM_SCHEMA
# Generate bash parameter defaults
for param in PARAM_SCHEMA:
    name = param.name.upper()
    if param.is_flag:
        # Boolean flags need special handling
        print(f'{name}=false')
        if param.default:
            print(f'{name}=true')
    else:
        # Regular parameters
        if isinstance(param.default, str):
            print(f'{name}=\"{param.default}\"')
        else:
            print(f'{name}={param.default}')
")

# Parse named command line arguments
while [[ $# -gt 0 ]]; do
    # Extract argument name without dashes and equals
    ARG_NAME=$(echo "$1" | sed -E 's/^--([^=]+)(=.*)?$/\1/')
    # Convert dashes to underscores and to uppercase for bash variable
    BASH_VAR=$(echo "$ARG_NAME" | tr '-' '_' | tr '[:lower:]' '[:upper:]')
    
    # Handle the argument
    if [[ "$1" == "--verbose" || "$1" == "--all-tasks" || "$1" == "--deterministic" ]]; then
        # Boolean flags
        declare "${BASH_VAR}=true"
        shift
    elif [[ "$1" == *"="* ]]; then
        # Arguments with values
        VALUE="${1#*=}"
        declare "${BASH_VAR}=${VALUE}"
        shift
    else
        echo "Unknown parameter format: $1"
        echo "Usage: $0 [--param1=value1] [--param2=value2] [--flag]"
        exit 1
    fi
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
#SBATCH --gpus=gpu:${GPUS}
#SBATCH --constraint='geforce_rtx_3090'
EOL

# GPU names: geforce_rtx_3090,rtx_a6000,a100

# Add optional SBATCH parameters if provided
if [[ -n "${PARTITION}" ]]; then
  echo "#SBATCH --partition=${PARTITION}" >> "${TEMP_SCRIPT}"
fi

if [[ -n "${EXCLUDE}" ]]; then
  echo "#SBATCH --exclude=${EXCLUDE}" >> "${TEMP_SCRIPT}"
fi

if [[ -n "${NODELIST}" ]]; then
  echo "#SBATCH --nodelist=${NODELIST}" >> "${TEMP_SCRIPT}"
fi

if [[ -n "${TIME}" ]]; then
  echo "#SBATCH --time=${TIME}" >> "${TEMP_SCRIPT}"
fi

# Add the rest of the script
cat >> "${TEMP_SCRIPT}" << EOL

# Set environment variables
ETH_USERNAME=${USER}
PROJECT_DIR=/itet-stor/\${ETH_USERNAME}/net_scratch/rstar-arc
CONDA_ENV=arc-solver
CLUSTER_ENV="SLURM"
export CLUSTER_ENV
DEFAULT_OUTPUT_DIR=/itet-stor/\${ETH_USERNAME}/net_scratch/outputs

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
if [[ ! -z "${NODELIST}" ]]; then echo "Node list: ${NODELIST}"; fi
if [[ ! -z "${TIME}" ]]; then echo "Time limit: ${TIME}"; fi
echo ""
echo "===== Application Parameters ====="

echo "Task Index: ${TASK_INDEX}"
echo "Max Iterations: ${MAX_ITERATIONS}"
echo "Policy Model: ${POLICY_MODEL}"
echo "PP Model: ${PP_MODEL}"
echo "Model dtype: ${DTYPE}"
if [[ ! -z "${HINT}" ]]; then echo "Hint: ${HINT}"; fi

# Activate conda
eval "\$(/itet-stor/\${ETH_USERNAME}/net_scratch/conda/bin/conda shell.bash hook)"
conda activate \${CONDA_ENV}
echo "Conda environment activated"

# Pass through parameters from bash to Python
PYTHON_ARGS=()

# Add task parameters
PYTHON_ARGS+=(--task-index="${TASK_INDEX}")
if [[ ! -z "${TASK_NAME}" ]]; then
    PYTHON_ARGS+=(--task-name="${TASK_NAME}")
fi

# Add model parameters
PYTHON_ARGS+=(--policy-model="${POLICY_MODEL}")
PYTHON_ARGS+=(--pp-model="${PP_MODEL}")
PYTHON_ARGS+=(--max-tokens="${MAX_TOKENS}")

# Add search parameters
PYTHON_ARGS+=(--search-mode="${SEARCH_MODE}")
PYTHON_ARGS+=(--max-depth="${MAX_DEPTH}")
PYTHON_ARGS+=(--max-iterations="${MAX_ITERATIONS}")
PYTHON_ARGS+=(--beam-width="${BEAM_WIDTH}")
PYTHON_ARGS+=(--temperature="${TEMPERATURE}")
PYTHON_ARGS+=(--seed="${SEED}")

# Add hardware parameters
PYTHON_ARGS+=(--gpus="${GPUS}")
PYTHON_ARGS+=(--dtype="${DTYPE}")

# Add output parameters
if [ ! -z "${OUTPUT_DIR}" ]; then
    PYTHON_ARGS+=(--output-dir="${OUTPUT_DIR}")
else
    PYTHON_ARGS+=(--output-dir="\${DEFAULT_OUTPUT_DIR}")
fi

if [ ! -z "${HINT}" ]; then
    PYTHON_ARGS+=(--hint="${HINT}")
fi

# Add config file if specified
if [ ! -z "${CONFIG_FILE}" ]; then
    PYTHON_ARGS+=(--config-file="${CONFIG_FILE}")
fi

# Add data folder
PYTHON_ARGS+=(--data-folder="${DATA_FOLDER}")

# Add boolean flags
if [ "${VERBOSE}" = "true" ]; then
    PYTHON_ARGS+=(--verbose)
fi

if [ "${ALL_TASKS}" = "true" ]; then
    PYTHON_ARGS+=(--all-tasks)
fi

if [ "${DETERMINISTIC}" = "true" ]; then
    PYTHON_ARGS+=(--deterministic)
fi

# Build and execute the command
CMD="python \${PROJECT_DIR}/main.py \${PYTHON_ARGS[@]}"
echo "Executing: \${CMD}"
cd \${PROJECT_DIR}
eval \${CMD}

# Cleanup
echo "Finished at: \$(date)"
exit 0
EOL

# Make the temporary script executable
chmod +x "${TEMP_SCRIPT}"

# Execute the script with sbatch
sbatch "${TEMP_SCRIPT}"

# Remove the temporary script
rm "${TEMP_SCRIPT}"

