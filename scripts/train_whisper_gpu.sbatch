#!/bin/bash

# ==============================
# SLURM Job Configuration for GPU Whisper Training
# ==============================

#SBATCH --job-name=whisper_gpu_training
#SBATCH --account=aisc                      # AISC account for H100 access
#SBATCH --partition=aisc                    # Partition with H100 GPUs
#SBATCH --nodes=1                           # Single node job
#SBATCH --ntasks-per-node=1                 # One task per node
#SBATCH --gpus-per-node=h100:4              # 4 H100 GPUs 
#SBATCH --cpus-per-task=100                  # 100 CPU cores for data loading
#SBATCH --mem=1000G                         # 1000G memory for training
#SBATCH --time=48:00:00                     # 48 hours for training
#SBATCH --output=logs/training/whisper_gpu_train_%j.out
#SBATCH --error=logs/training/whisper_gpu_train_%j.err

# ==============================
# Environment Setup
# ==============================

# Create log directory
mkdir -p logs/training

# Print job information
echo "===== GPU Training Job Information ====="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: $(hostname)"
echo "GPU Count: ${SLURM_GPUS_PER_NODE}"
echo "CPU Count: ${SLURM_CPUS_PER_TASK}"
echo "Memory: ${SLURM_MEM_PER_NODE}MB"
echo "========================================="

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
# Run GPU Training
# ==============================

echo "Starting Whisper GPU training..."
echo "Command: python scripts/finetune_whisper_gpu.py -i data/LangAgeLogMelSpec -o ./FrisperWhisper -v v2 --num_gpus 4 --num_cpus 100 --model_size large"
echo "========================================="

python scripts/finetune_whisper_gpu.py \
    -i data/LangAgeLogMelSpec \
    -o ./FrisperWhisper \
    -v v2 \
    --num_gpus 4 \
    --num_cpus 100 \
    --model_size large \
    --dataloader_workers 8

echo "========================================="
echo "GPU training completed at $(date)"
