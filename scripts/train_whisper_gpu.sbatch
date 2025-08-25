#!/bin/bash

# ==============================
# SLURM Job Configuration for GPU Whisrun torchrun \
    --nproc_per_node=8 \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=localhost \
    --master_port=${MASTER_PORT} \
    scripts/finetune_whisper_gpu.py \
    -i data/LangAgeLogMelSpec \
    -o ./FrisperWhisper \
    -v gpu_21 \
    --num_gpus 8 \
    --num_cpus 120 \g
# ==============================

#SBATCH --job-name=whisper_gpu_training
#SBATCH --account=aisc                      # AISC account for H100 access
#SBATCH --partition=aisc                    # Partition with H100 GPUs
#SBATCH --nodes=1                           # Single node job
#SBATCH --ntasks-per-node=6                 # Eight tasks for 8 GPUs
#SBATCH --gpus-per-node=h100:6              # 8 H100 GPUs 
#SBATCH --cpus-per-task=15                  # 15 CPU cores per task (120 total)
#SBATCH --mem=1500G                          # 800G memory for training
#SBATCH --time=96:00:00                     # 48 hours for training
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

# Set CUDA environment and memory optimization
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:128

# Navigate to project directory
cd /sc/home/hanno.mueller/TTS-LangAge/

# Activate virtual environment
echo "Activating virtual environment..."
source .venv/bin/activate

# Verify GPU availability
echo "Checking GPU availability..."
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU count: {torch.cuda.device_count()}'); [print(f'GPU {i}: {torch.cuda.get_device_name(i)} ({torch.cuda.get_device_properties(i).total_memory / 1024**3:.1f} GB)') for i in range(torch.cuda.device_count())]"

# Set HuggingFace cache to correct location where models are cached
export HF_HOME="/sc/home/hanno.mueller/.cache/huggingface"
export HF_HUB_CACHE="/sc/home/hanno.mueller/.cache/huggingface/hub"
# Remove deprecated TRANSFORMERS_CACHE to avoid conflicts
unset TRANSFORMERS_CACHE
# Force offline mode to prevent network downloads
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1

# Set distributed training environment variables with unique port per job
export MASTER_ADDR=localhost
export MASTER_PORT=$((29500 + ${SLURM_JOB_ID} % 1000))
export WORLD_SIZE=6
export NCCL_DEBUG=WARN
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1

echo "Using port: ${MASTER_PORT} for job ${SLURM_JOB_ID}"

# ==============================
# Run GPU Training
# ==============================

echo "Starting Whisper GPU training with 8 GPUs using torchrun..."
echo "Configuration:"
echo "  - Method: torchrun distributed training"
echo "  - GPUs: 6 H100"
echo "  - Process per node: 6"
echo "  - Batch size per device: 1"
echo "  - Gradient accumulation: 16"
echo "  - Total effective batch size: 128"
echo "Command: srun torchrun --nproc_per_node=6 scripts/finetune_whisper_gpu.py -i data/LangAgeLogMelSpec -o ./FrisperWhisper -v gpu_23_96h_6gpu --num_gpus 6 --num_cpus 120 --model_size large --per_device_train_batch_size 1 --per_device_eval_batch_size 1 --gradient_accumulation_steps 16"
echo "========================================="

srun torchrun \
    --nproc_per_node=6 \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=localhost \
    --master_port=${MASTER_PORT} \
    scripts/finetune_whisper_gpu.py \
    -i data/LangAgeLogMelSpec \
    -o ./FrisperWhisper \
    -v gpu_23_72h_extensive \
    --num_gpus 6 \
    --num_cpus 60 \
    --model_size large \
    --dataloader_workers 8 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 16

echo "========================================="
echo "GPU training completed at $(date)"

# Print final GPU memory usage
echo "Final GPU memory usage:"
python -c "import torch; [print(f'GPU {i}: {torch.cuda.memory_allocated(i) / 1024**3:.2f}GB allocated') for i in range(torch.cuda.device_count()) if torch.cuda.is_available()]" || echo "GPU memory check failed"
