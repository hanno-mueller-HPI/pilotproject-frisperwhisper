#!/bin/bash
#SBATCH --job-name=whisper_training
#SBATCH --output=logs/whisper_training_%j.out
#SBATCH --error=logs/whisper_training_%j.err
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

echo "Starting Whisper training with finetune_whisper.py..."
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Available GPUs: $CUDA_VISIBLE_DEVICES"

# Default parameters - modify as needed
DATASET_PATH="${DATASET_PATH:-data/LangAgeDataSet}"
OUTPUT_DIR="${OUTPUT_DIR:-FrisperWhisper}"
VERSION="${VERSION:-v1}"
MODEL_SIZE="${MODEL_SIZE:-large}"
NUM_GPUS="${NUM_GPUS:-4}"
NUM_CPUS="${NUM_CPUS:-16}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-1}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-1}"
GRADIENT_ACCUMULATION="${GRADIENT_ACCUMULATION:-16}"
LEARNING_RATE="${LEARNING_RATE:-1e-5}"
MAX_STEPS="${MAX_STEPS:-20000}"
RESUME_CHECKPOINT="${RESUME_CHECKPOINT:-}"

echo "========================================="
echo "Training Configuration:"
echo "Dataset: $DATASET_PATH"
echo "Output: $OUTPUT_DIR/$VERSION"
echo "Model: $MODEL_SIZE"
echo "GPUs: $NUM_GPUS"
echo "CPUs: $NUM_CPUS"
echo "Batch size (train/eval): $TRAIN_BATCH_SIZE/$EVAL_BATCH_SIZE"
echo "Gradient accumulation: $GRADIENT_ACCUMULATION"
echo "Learning rate: $LEARNING_RATE"
echo "Max steps: $MAX_STEPS"
if [ -n "$RESUME_CHECKPOINT" ]; then
    echo "Resume from checkpoint: $RESUME_CHECKPOINT"
fi
echo "========================================="

# Build command arguments
TRAIN_ARGS=(
    --input_dataset "$DATASET_PATH"
    --output_dir "$OUTPUT_DIR"
    --version "$VERSION"
    --model_size "$MODEL_SIZE"
    --num_gpus "$NUM_GPUS"
    --num_cpus "$NUM_CPUS"
    --per_device_train_batch_size "$TRAIN_BATCH_SIZE"
    --per_device_eval_batch_size "$EVAL_BATCH_SIZE"
    --gradient_accumulation_steps "$GRADIENT_ACCUMULATION"
    --learning_rate "$LEARNING_RATE"
    --max_steps "$MAX_STEPS"
    --save_steps 1000
    --eval_steps 1000
    --logging_steps 25
    --warmup_steps 2000
)

# Add checkpoint resumption if specified
if [ -n "$RESUME_CHECKPOINT" ]; then
    TRAIN_ARGS+=(--resume_from_checkpoint "$RESUME_CHECKPOINT")
fi

# Run training with the unified script
python scripts/finetune_whisper.py "${TRAIN_ARGS[@]}"

if [ $? -eq 0 ]; then
    echo "========================================="
    echo "Training completed successfully!"
    echo "Model saved to: $OUTPUT_DIR/$VERSION"
    echo "========================================="
else
    echo "========================================="
    echo "Training failed with exit code $?"
    echo "========================================="
    exit 1
fi
