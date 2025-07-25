#!/bin/bash

# ==============================
# SLURM Job Configuration for Dataset Preprocessing
# ==============================

#SBATCH --job-name=dataset_preprocessing
#SBATCH --account=aisc                      # AISC account for resource access
#SBATCH --partition=aisc                    # AISC partition (CPU-only job)
#SBATCH --nodes=1-4                         # Allow 1-4 nodes for flexibility
#SBATCH --ntasks=1                          # Single task (shared memory processing)
#SBATCH --cpus-per-task=200                 # 200 CPU cores total across nodes
#SBATCH --time=12:00:00                     # 12 hours for preprocessing
#SBATCH --exclusive                         # Request exclusive node access for better performance
#SBATCH --output=logs/preprocessing/dataset_preprocess_%j.out
#SBATCH --error=logs/preprocessing/dataset_preprocess_%j.err

# ==============================
# Environment Setup
# ==============================

# Create log directory
mkdir -p logs/preprocessing

# Print job information
echo "===== Dataset Preprocessing Job Information ====="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: $(hostname)"
echo "CPU Count: ${SLURM_CPUS_PER_TASK}"
echo "Memory: ${SLURM_MEM_PER_NODE}MB"
echo "=================================================="

# Navigate to project directory
cd /sc/home/hanno.mueller/TTS-LangAge/

# Activate virtual environment
echo "Activating virtual environment..."
source .venv/bin/activate

# Set HuggingFace cache to avoid downloading issues
export HF_HOME="/sc/home/hanno.mueller/.huggingface"
export TRANSFORMERS_CACHE="/sc/home/hanno.mueller/.huggingface/transformers"

# ==============================
# Run Dataset Preprocessing
# ==============================

echo "Starting dataset preprocessing..."
echo "Command: python scripts/Dataset2LogMelSpec.py -i LangAgeDataSet -o LangAgeDataSet_preprocessed --num_cpus 200 --model_size large"
echo "=================================================="

python scripts/Dataset2LogMelSpec.py \
    -i LangAgeDataSet \
    -o LangAgeDataSet_preprocessed \
    --num_cpus 200 \
    --model_size large

echo "=================================================="
echo "Dataset preprocessing completed at $(date)"
