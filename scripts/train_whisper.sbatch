#!/bin/bash
#SBATCH --job-name=whisper_large_v3_training
#SBATCH --output=logs/whisper_large_v3_training_%j.out
#SBATCH --error=logs/whisper_large_v3_training_%j.err
#SBATCH --partition=aisc
#SBATCH --account=aisc                      # AISC account for H100 access
#SBATCH --qos=aisc
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24                  # Increased for 20 dataloader workers
#SBATCH --gres=gpu:4
#SBATCH --mem=1000G
#SBATCH --time=96:00:00

# Load necessary modules
module load CUDA/11.8
module load Python/3.11.3

# Activate virtual environment
cd /sc/home/hanno.mueller/pilotproject-frisperwhisper
source .venv/bin/activate

# Set CUDA environment variables for 4 GPUs
export CUDA_VISIBLE_DEVICES=0,1,2,3

echo "Starting Whisper large-v3 training with finetune_whisper_from_LogMEL.py..."
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Available GPUs: $CUDA_VISIBLE_DEVICES"

# Configuration for large-v3 training
DATASET_PATH="data/LangAgeLogMelSpec"
OUTPUT_DIR="FrisperWhisper/largeV3.2"
MODEL_SIZE="large-v3"
NUM_GPUS=4
DATALOADER_WORKERS=20
TRAIN_BATCH_SIZE=1                    # Conservative for large-v3
GRADIENT_ACCUMULATION=16              # Effective batch size = 1×4×16 = 64
LEARNING_RATE=1.5e-5                  # Increased from 3e-6 for better convergence
MAX_STEPS=10000                       # Sufficient for convergence
WARMUP_STEPS=1000                     # 10% of max steps
SAVE_STEPS=500
EVAL_STEPS=500
LOGGING_STEPS=50
WEIGHT_DECAY=0.05                     # Increased from 0.02 for better regularization
LR_SCHEDULER_TYPE="linear"            # Changed from cosine for more gradual decay

echo "========================================="
echo "Training Configuration:"
echo "Dataset: $DATASET_PATH"
echo "Output: $OUTPUT_DIR"
echo "Model: $MODEL_SIZE"
echo "GPUs: $NUM_GPUS"
echo "Dataloader workers: $DATALOADER_WORKERS"
echo "Batch size per GPU: $TRAIN_BATCH_SIZE"
echo "Gradient accumulation: $GRADIENT_ACCUMULATION"
echo "Effective batch size: $((TRAIN_BATCH_SIZE * NUM_GPUS * GRADIENT_ACCUMULATION))"
echo "Learning rate: $LEARNING_RATE"
echo "Max steps: $MAX_STEPS"
echo "Warmup steps: $WARMUP_STEPS"
echo "========================================="

# Build command arguments for finetune_whisper_from_LogMEL.py
TRAIN_ARGS=(
    --dataset_path "$DATASET_PATH"
    --output_dir "$OUTPUT_DIR"
    --model_size "$MODEL_SIZE"
    --num_gpus "$NUM_GPUS"
    --dataloader_num_workers "$DATALOADER_WORKERS"
    --per_device_train_batch_size "$TRAIN_BATCH_SIZE"
    --gradient_accumulation_steps "$GRADIENT_ACCUMULATION"
    --learning_rate "$LEARNING_RATE"
    --max_steps "$MAX_STEPS"
    --warmup_steps "$WARMUP_STEPS"
    --save_steps "$SAVE_STEPS"
    --eval_steps "$EVAL_STEPS"
    --logging_steps "$LOGGING_STEPS"
    --max_grad_norm 0.5
    --weight_decay "$WEIGHT_DECAY"
    --lr_scheduler_type "$LR_SCHEDULER_TYPE"
    --bf16
    --report_to "tensorboard"
)

# Run training with the LogMEL script
python scripts/finetune_whisper_from_LogMEL.py "${TRAIN_ARGS[@]}"

if [ $? -eq 0 ]; then
    echo "========================================="
    echo "Training completed successfully!"
    echo "Model saved to: $OUTPUT_DIR"
    echo "TensorBoard logs: $OUTPUT_DIR/runs/"
    echo "========================================="
else
    echo "========================================="
    echo "Training failed with exit code $?"
    echo "========================================="
    exit 1
fi
