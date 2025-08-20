#!/bin/bash

# ==============================
# SLURM Job Configuration for Multi-Node GPU Whisper Training
# ==============================

#SBATCH --job-name=whisper_multinode_training
#SBATCH --account=aisc                      # AISC account for H100 access
#SBATCH --partition=aisc                    # Partition with H100 GPUs
#SBATCH --nodes=4                           # 4 nodes for distributed training
#SBATCH --ntasks-per-node=3                 # 3 tasks per node (for 12 GPUs total)
#SBATCH --gpus-per-node=h100:3              # 3 H100 GPUs per node
#SBATCH --cpus-per-task=25                  # 25 CPU cores per task (100 total / 4 tasks)
#SBATCH --mem=375G                          # 375G memory per node (1500G total / 4 nodes)
#SBATCH --time=72:00:00                     # 72 hours for distributed training
#SBATCH --output=logs/training/whisper_multinode_train_%j.out
#SBATCH --error=logs/training/whisper_multinode_train_%j.err

# ==============================
# Environment Setup
# ==============================

# Create log directory
mkdir -p logs/training

# Print job information
echo "===== Multi-Node GPU Training Job Information ====="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: $(hostname)"
echo "Node List: ${SLURM_JOB_NODELIST}"
echo "Total Nodes: ${SLURM_NNODES}"
echo "Tasks per Node: ${SLURM_NTASKS_PER_NODE}"
echo "Total Tasks: ${SLURM_NTASKS}"
echo "GPU Count per Node: ${SLURM_GPUS_PER_NODE}"
echo "CPU Count per Task: ${SLURM_CPUS_PER_TASK}"
echo "Memory per Node: ${SLURM_MEM_PER_NODE}MB"
echo "World Size: $(( ${SLURM_NNODES} * ${SLURM_NTASKS_PER_NODE} ))"
echo "========================================="

# Set up distributed training environment variables
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=29500
export WORLD_SIZE=$(( ${SLURM_NNODES} * ${SLURM_NTASKS_PER_NODE} ))

# Navigate to project directory
cd /sc/home/hanno.mueller/TTS-LangAge/

# Activate virtual environment
echo "Activating virtual environment..."
source .venv/bin/activate

# Verify GPU availability
echo "Checking GPU availability on node $(hostname)..."
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU count: {torch.cuda.device_count()}')"

# Set HuggingFace cache to avoid downloading issues
export HF_HOME="/sc/home/hanno.mueller/.huggingface"
export TRANSFORMERS_CACHE="/sc/home/hanno.mueller/.huggingface/transformers"

# ==============================
# Run Multi-Node Distributed Training
# ==============================

echo "Starting Whisper Multi-Node GPU training..."
echo "Command: srun torchrun scripts/finetune_whisper_gpu.py -i data/LangAgeLogMelSpec -o ./FrisperWhisper -v multinode --num_gpus 3 --num_cpus 25 --model_size large"
echo "========================================="

# Use srun with torchrun for proper multi-node distributed training
srun torchrun \
    --nproc_per_node=3 \
    --nnodes=4 \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
    scripts/finetune_whisper_gpu.py \
    -i data/LangAgeLogMelSpec \
    -o ./FrisperWhisper \
    -v multinode \
    --num_gpus 3 \
    --num_cpus 25 \
    --model_size large \
    --dataloader_workers 2

echo "========================================="
echo "Multi-node GPU training completed at $(date)"
