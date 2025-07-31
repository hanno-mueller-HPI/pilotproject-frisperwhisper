#!/bin/bash

# ==============================
# SLURM Job Configuration for Dataset Creation
# ==============================

#SBATCH --job-name=mk_dataset
#SBATCH --account=aisc                      # AISC account for resource access
#SBATCH --partition=aisc                    # AISC partition
#SBATCH --nodes=1                           # Single node job
#SBATCH --cpus-per-task=150                 # 150 CPU cores
#SBATCH --mem=1000G                         # 1TB memory
#SBATCH --time=24:00:00                     # 24 hours
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
cd /sc/home/hanno.mueller/TTS-LangAge/

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
# Run Dataset Creation
# ==============================

echo "Starting dataset creation..."
echo "Command: python scripts/Textgrids2DatasetBatch.py -f data -o LangAgeDataSetV2 -n 150 --batch_size 500 --audio_batch_processes 8"
echo "=============================================="

python scripts/Textgrids2DatasetBatch.py \
    -f data \
    -o LangAgeDataSetV2 \
    -n 150 \
    --batch_size 500 \
    --audio_batch_processes 8

echo "=============================================="
echo "Dataset creation completed at $(date)"

# Show output directory structure
echo "Output directory contents:"
ls -la LangAgeDataSetV2/
