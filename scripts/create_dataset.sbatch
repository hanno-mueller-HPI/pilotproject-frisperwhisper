#!/bin/bash

# ==============================
# SLURM Job Configuration for Dataset Creation
# ==============================

#SBATCH --job-name=mk_dataset
#SBATCH --account=aisc                      # AISC account for resource access
#SBATCH --partition=aisc                    # AISC partition
#SBATCH --nodes=1                           # Single node job
#SBATCH --cpus-per-task=120                 # 120 CPU cores
#SBATCH --mem=1024G                         # 1TB memory
#SBATCH --time=48:00:00                     # 48 hours
#SBATCH --constraint=ARCH:X86               # X86 architecture constraint
#SBATCH --output=logs/dataset_creation/mk_dataset_%j.out
#SBATCH --error=logs/dataset_creation/mk_dataset_%j.err

# ==============================
# Environment Setup
# ==============================

# Create log directory
mkdir -p logs/dataset_creation

# Print job information
echo "===== Dataset Creation Job Information ====="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: $(hostname)"
echo "CPU Count: ${SLURM_CPUS_PER_TASK}"
echo "Memory: ${SLURM_MEM_PER_NODE}MB"
echo "Architecture: $(uname -m)"
echo "=============================================="

# Navigate to project directory
cd /sc/home/hanno.mueller/pilotproject-frisperwhisper/

# Activate virtual environment
echo "Activating virtual environment..."
source .venv/bin/activate

# Verify Python environment
echo "Python version: $(python --version)"
echo "Available CPU cores: $(nproc)"

# Set environment variables for multiprocessing
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

# ==============================
# Configuration with Environment Variables
# ==============================

# Set default values if environment variables are not provided
INPUT_FOLDER="${INPUT_FOLDER:-data/LangAge16kHz}"
OUTPUT_FOLDER="${OUTPUT_FOLDER:-data/LangAgeDataSet}"
NUM_PROCESSES="${NUM_PROCESSES:-120}"
BATCH_SIZE="${BATCH_SIZE:-500}"
AUDIO_BATCH_PROCESSES="${AUDIO_BATCH_PROCESSES:-8}"
CSV_FILE="${CSV_FILE:-}"  # Optional, empty by default

# ==============================
# Run Dataset Creation
# ==============================

echo "Starting dataset creation..."
echo "Configuration:"
echo "  Input folder: ${INPUT_FOLDER}"
echo "  Output folder: ${OUTPUT_FOLDER}"
echo "  Number of processes: ${NUM_PROCESSES}"
echo "  Batch size: ${BATCH_SIZE}"
echo "  Audio batch processes: ${AUDIO_BATCH_PROCESSES}"
if [ -n "${CSV_FILE}" ]; then
    echo "  CSV file: ${CSV_FILE}"
fi
echo "=============================================="

# Build command with optional CSV file parameter
CMD="python scripts/Textgrids2DatasetBatch.py -f ${INPUT_FOLDER} -o ${OUTPUT_FOLDER} -n ${NUM_PROCESSES} --batch_size ${BATCH_SIZE} --audio_batch_processes ${AUDIO_BATCH_PROCESSES}"
if [ -n "${CSV_FILE}" ]; then
    CMD="${CMD} -c ${CSV_FILE}"
fi

echo "Command: ${CMD}"
echo "=============================================="

# Execute the command
${CMD}

echo "=============================================="
echo "Dataset creation completed at $(date)"

# Show output directory structure
echo "Output directory contents:"
ls -la ${OUTPUT_FOLDER}/
