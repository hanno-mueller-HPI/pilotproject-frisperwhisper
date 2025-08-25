#!/bin/bash
#SBATCH --job-name=whisper_hf_training
#SBATCH --output=logs/whisper_hf_gpu_%j.out
#SBATCH --error=logs/whisper_hf_gpu_%j.err
#SBATCH --partition=aisc
#SBATCH --account=aisc                      # AISC account for H100 access
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:4
#SBATCH --mem=1000G
#SBATCH --time=96:00:00

# Load necessary modules
module load CUDA/11.8
module load Python/3.11.3

# Activate virtual environment
cd /sc/home/hanno.mueller/TTS-LangAge
source .venv/bin/activate

# Set CUDA environment variables
export CUDA_VISIBLE_DEVICES=0,1,2,3

echo "Starting Whisper HF GPU training with 4 GPUs..."
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Available GPUs: $CUDA_VISIBLE_DEVICES"

# First training: LangAgeDataSetSubsample1 -> FrisperWhisper/hf1
echo "========================================="
echo "Training 1/3: LangAgeDataSetSubsample1"
echo "========================================="
python scripts/finetune_whisper_HF_gpu.py \
    --dataset_path data/LangAgeDataSetSubsample1 \
    --output_dir FrisperWhisper/hf1 \
    --model_size large-v3 \
    --num_gpus 4

if [ $? -ne 0 ]; then
    echo "Training 1/3 failed, exiting..."
    exit 1
fi

# Second training: LangAgeDataSetSubsample10 -> FrisperWhisper/hf10
echo "========================================="
echo "Training 2/3: LangAgeDataSetSubsample10"
echo "========================================="
python scripts/finetune_whisper_HF_gpu.py \
    --dataset_path data/LangAgeDataSetSubsample10 \
    --output_dir FrisperWhisper/hf10 \
    --model_size large-v3 \
    --num_gpus 4

if [ $? -ne 0 ]; then
    echo "Training 2/3 failed, exiting..."
    exit 1
fi

# Third training: LangAgeDataSet -> FrisperWhisper/hf
echo "========================================="
echo "Training 3/3: LangAgeDataSet (full)"
echo "========================================="
python scripts/finetune_whisper_HF_gpu.py \
    --dataset_path data/LangAgeDataSet \
    --output_dir FrisperWhisper/hf \
    --model_size large-v3 \
    --num_gpus 4

if [ $? -ne 0 ]; then
    echo "Training 3/3 failed, exiting..."
    exit 1
fi

echo "========================================="
echo "All three training jobs completed successfully!"
echo "========================================="
