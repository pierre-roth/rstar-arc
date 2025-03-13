#!/bin/bash

# Create temporary SBATCH script with fixed parameters
TEMP_SCRIPT=$(mktemp)
cat > "${TEMP_SCRIPT}" << EOL
#!/bin/bash
#SBATCH --mail-type=NONE
#SBATCH --output=/itet-stor/${USER}/net_scratch/outputs/jobs/%j.out
#SBATCH --error=/itet-stor/${USER}/net_scratch/outputs/jobs/%j.err
#SBATCH --mem=32G
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=gpu:1
#CommentSBATCH --constraint='geforce_rtx_3090'

# Set environment variables
ETH_USERNAME=${USER}
PROJECT_DIR=/itet-stor/\${ETH_USERNAME}/net_scratch/rstar-arc
CONDA_ENV=arc-solver
DEFAULT_OUTPUT_DIR=/itet-stor/\${ETH_USERNAME}/net_scratch/outputs

# Exit on errors
set -o errexit

# Create jobs directory if it doesn't exist
mkdir -p /itet-stor/\${ETH_USERNAME}/net_scratch/outputs/jobs

# Log basic information
echo "Running on node: \$(hostname)"
echo "In directory: \$(pwd)"
echo "Starting on: \$(date)"
echo "SLURM_JOB_ID: \${SLURM_JOB_ID}"

# Activate conda
eval "\$(/itet-stor/\${ETH_USERNAME}/net_scratch/conda/bin/conda shell.bash hook)"
conda activate \${CONDA_ENV}
echo "Conda environment activated"

# Execute the Python script with fixed parameters
cd \${PROJECT_DIR}

python \${PROJECT_DIR}/main.py \\
  --task-index=1 \\
  --task-name="ac0a08a4" \\
  --policy-model="Qwen/Qwen2.5-Coder-7B-Instruct" \\
  --pp-model="Qwen/Qwen2.5-Coder-7B-Instruct" \\
  --max-tokens=2048 \\
  --search-mode="beam_search" \\
  --max-depth=10 \\
  --beam-width=3 \\
  --branching-factor=3 \\
  --temperature=0.3 \\
  --seed=42 \\
  --gpus=1 \\
  --dtype="float16" \\
  --output-dir="\${DEFAULT_OUTPUT_DIR}" \\
  --data-folder="data_sample/training" \\
  --verbose

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

