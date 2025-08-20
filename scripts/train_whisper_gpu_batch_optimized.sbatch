#!/bin/bash

# ==============================
# SLURM Job Configuration for Optimized GPU Whisper Training
# ==============================

#SBATCH --job-name=whisper_gpu_batch_optimized
#SBATCH --account=aisc                      # AISC account for H100 access
#SBATCH --partition=aisc                    # Partition with H100 GPUs
#SBATCH --nodes=1                           # Single node job
#SBATCH --ntasks-per-node=1                 # One task per node
#SBATCH --gpus-per-node=h100:4              # 4 H100 GPUs 
#SBATCH --cpus-per-task=60                  # 60 CPU cores for data loading (15 per GPU)
#SBATCH --mem=500G                          # 500G memory - reduced from 1000G since we use smaller batches
#SBATCH --time=48:00:00                     # 48 hours for training
#SBATCH --output=logs/training/whisper_gpu_batch_optimized_%j.out
#SBATCH --error=logs/training/whisper_gpu_batch_optimized_%j.err

# ==============================
# Environment Setup
# ==============================

# Create log directory
mkdir -p logs/training

# Print job information
echo "===== Optimized GPU Training Job Information ====="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: $(hostname)"
echo "GPU Count: ${SLURM_GPUS_PER_NODE}"
echo "CPU Count: ${SLURM_CPUS_PER_TASK}"
echo "Memory: ${SLURM_MEM_PER_NODE}MB"
echo "Start Time: $(date)"
echo "======================================================="

# Set CUDA environment and memory optimization
export CUDA_VISIBLE_DEVICES=0,1,2,3
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Navigate to project directory
cd /sc/home/hanno.mueller/TTS-LangAge/

# Activate virtual environment
echo "Activating virtual environment..."
source .venv/bin/activate

# Verify GPU availability
echo "Checking GPU availability..."
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU count: {torch.cuda.device_count()}'); [print(f'GPU {i}: {torch.cuda.get_device_name(i)} ({torch.cuda.get_device_properties(i).total_memory / 1024**3:.1f} GB)') for i in range(torch.cuda.device_count())]"

# Set HuggingFace cache to avoid downloading issues
export HF_HOME="/sc/home/hanno.mueller/.huggingface"
export TRANSFORMERS_CACHE="/sc/home/hanno.mueller/.huggingface/transformers"

# ==============================
# Run Optimized GPU Training
# ==============================

echo "Starting Whisper GPU training with optimized memory settings..."
echo "Memory optimizations applied based on previous OOM debugging:"
echo "  - Batch size per device: 4 (down from 32)"
echo "  - Eval batch size per device: 2 (down from 16)" 
echo "  - Gradient accumulation steps: 8 (to maintain effective batch size)"
echo "  - PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
echo "======================================================="

echo "Command: python scripts/finetune_whisper_gpu.py -i data/LangAgeLogMelSpec -o ./FrisperWhisper -v batch-optimised --num_gpus 4 --num_cpus 60 --model_size large --dataloader_workers 8 --per_device_train_batch_size 4 --per_device_eval_batch_size 2 --gradient_accumulation_steps 8"
echo "======================================================="

python scripts/finetune_whisper_gpu.py \
    -i data/LangAgeLogMelSpec \
    -o ./FrisperWhisper \
    -v batch-optimised \
    --num_gpus 4 \
    --num_cpus 60 \
    --model_size large \
    --dataloader_workers 8 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 2 \
    --gradient_accumulation_steps 8

echo "======================================================="
echo "GPU training completed at $(date)"
echo "Model saved to: ./FrisperWhisper/batch-optimised"

# Optional: Print final model directory contents
echo "Final model directory contents:"
ls -la ./FrisperWhisper/batch-optimised/ || echo "Model directory not found"
