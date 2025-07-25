#!/bin/bash

# ==============================
# SLURM Job Configuration for Whisper Fine-tuning
# ==============================

#SBATCH --job-name=whisper_finetune
#SBATCH --account=aisc                      # AISC account for H100 access
#SBATCH --partition=aisc                    # Partition with H100 GPUs
#SBATCH --nodes=1                           # Single node job
#SBATCH --ntasks-per-node=1                 # One task per node
#SBATCH --gpus-per-node=h100:4              # 4 H100 GPUs 
#SBATCH --cpus-per-task=200                 # 200 CPU cores
#SBATCH --mem=1800G                         # Total memory: 4×50GB + 200×8GB = 1800GB
#SBATCH --time=24:00:00                     # 24 hours as requested
#SBATCH --output=logs/integrated/whisper_finetune_%j.out
#SBATCH --error=logs/integrated/whisper_finetune_%j.err

# ==============================
# Environment Setup
# ==============================

# Create log directory
mkdir -p logs/integrated

# Print job information
echo "===== Job Information ====="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: $(hostname)"
echo "GPU Count: ${SLURM_GPUS_PER_NODE}"
echo "CPU Count: ${SLURM_CPUS_PER_TASK}"
echo "Memory: ${SLURM_MEM_PER_NODE}MB"
echo "==========================="

# Set CUDA environment
export CUDA_VISIBLE_DEVICES=0,1,2,3

# Navigate to project directory
cd /sc/home/hanno.mueller/TTS-LangAge/

# Activate virtual environment
echo "Activating virtual environment..."
source .venv/bin/activate

# Verify GPU availability
echo "Checking GPU availability..."
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU count: {torch.cuda.device_count()}')"

# Set HuggingFace cache to avoid downloading issues
export HF_HOME="/sc/home/hanno.mueller/.huggingface"
export TRANSFORMERS_CACHE="/sc/home/hanno.mueller/.huggingface/transformers"

# ==============================
# Run Training
# ==============================

echo "Starting Whisper fine-tuning..."
echo "Command: python scripts/finetune_whisper.py -d LangAgeDataSet -o ./output_test --num_cpus 200 --num_gpus 4"
echo "==========================="

python scripts/finetune_whisper.py \
    -d LangAgeDataSet \
    -o ./output_test \
    --num_cpus 200 \
    --num_gpus 4

echo "==========================="
echo "Job completed at $(date)"
