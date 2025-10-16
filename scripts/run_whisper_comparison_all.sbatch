#!/bin/bash
#SBATCH --job-name=whisper_all_3models
#SBATCH --account=aisc                      # AISC account for GPU access
#SBATCH --partition=aisc                    # Partition with GPUs
#SBATCH --qos=aisc                          # QoS setting for AISC
#SBATCH --output=logs/whisper_comparison_all_%j.out
#SBATCH --error=logs/whisper_comparison_all_%j.err
#SBATCH --time=120:00:00                    # 5 days (partition limit)
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=56                  # 56 CPUs
#SBATCH --gres=gpu:4                        # 2 GPUs (adjust as needed)
#SBATCH --mem=600G                          # 600GB memory

# This script compares THREE Whisper models:
# 1. Whisper Large V3 (default OpenAI model)
# 2. Fine-tuned model (previously trained)
# 3. New model (your newly trained version)

echo "Using virtual environment CUDA setup directly"

echo "=== SLURM Job Information ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURMD_NODENAME"
echo "Start time: $(date)"
echo "CPUs allocated: $SLURM_CPUS_PER_TASK"
echo "GPUs allocated: $SLURM_GPUS"
echo "Memory allocated: $SLURM_MEM_PER_NODE MB"
echo "=============================="

# Navigate to the project directory
cd /sc/home/hanno.mueller/pilotproject-frisperwhisper

# Activate virtual environment
echo "Activating virtual environment..."
source .venv/bin/activate

# Verify environment
echo "Python path: $(which python)"
echo "Working directory: $(pwd)"

# Set CUDA environment variables
export CUDA_VISIBLE_DEVICES=0,1,2,3
echo "Available GPUs: $CUDA_VISIBLE_DEVICES"

# Create logs directory if it doesn't exist
mkdir -p logs

# ========================================
# CONFIGURATION - ADJUST THESE VARIABLES
# ========================================

# Input directory: Combined LangAge + ESLO data (16kHz audio + TextGrid files)
INPUT_DIR="data/LangAgeESLOcombined16kHz"

# Model paths
FINE_TUNED_MODEL="./FrisperWhisper/largeV3.2"
NEW_MODEL="./FrisperWhisper/largeV3_LangAgeESLO"  # UPDATE THIS!

# Checkpoints (optional - leave empty to use final model)
CHECKPOINT_FINE_TUNED="checkpoint-2000"  # Use checkpoint-2000 for fine-tuned
CHECKPOINT_NEW="checkpoint-3500"         # Using checkpoint-3000 for new model

# Dataset path for train/test tracking (optional)
DATASET_PATH="data/LangAgeESLODataSet"   # Optional: adds train/test columns

# Automatically generate output directory name
OUTPUT_DIR="results/comparison_all_3models_v2"

# Processing configuration
NUM_CPUS=56
NUM_GPUS=4
BATCH_SIZE=64                           # Batch size for transcription
TRANSCRIPTION_PROCESSES=8               # Number of batch processes

echo "========================================="
echo "Whisper 3-Model Comparison Configuration:"
echo "Input directory: $INPUT_DIR"
echo "Output directory: $OUTPUT_DIR"
echo "Fine-tuned model: $FINE_TUNED_MODEL"
if [ -n "$CHECKPOINT_FINE_TUNED" ]; then
    echo "  └─ Checkpoint: $CHECKPOINT_FINE_TUNED"
else
    echo "  └─ Checkpoint: final model"
fi
echo "New model: $NEW_MODEL"
if [ -n "$CHECKPOINT_NEW" ]; then
    echo "  └─ Checkpoint: $CHECKPOINT_NEW"
else
    echo "  └─ Checkpoint: final model"
fi
echo "CPUs: $NUM_CPUS"
echo "GPUs: $NUM_GPUS"
echo "Batch size: $BATCH_SIZE"
echo "Transcription processes: $TRANSCRIPTION_PROCESSES"
echo "========================================="

# Validate input directory
if [ ! -d "$INPUT_DIR" ]; then
    echo "ERROR: Input directory '$INPUT_DIR' does not exist!"
    exit 1
fi

# Validate model directories
if [ ! -d "$FINE_TUNED_MODEL" ]; then
    echo "ERROR: Fine-tuned model directory '$FINE_TUNED_MODEL' does not exist!"
    exit 1
fi

if [ ! -d "$NEW_MODEL" ]; then
    echo "ERROR: New model directory '$NEW_MODEL' does not exist!"
    echo "Please update the NEW_MODEL variable in this script."
    exit 1
fi

# Build command arguments for run_whisper_comparison_all.py
COMPARISON_ARGS=(
    --input "$INPUT_DIR"
    --output "$OUTPUT_DIR"
    --fine_tuned_model "$FINE_TUNED_MODEL"
    --new_model "$NEW_MODEL"
    --cpus "$NUM_CPUS"
    --gpus "$NUM_GPUS"
    --batch_size "$BATCH_SIZE"
    --transcription_batch_processes "$TRANSCRIPTION_PROCESSES"
    --steps "all"
)

# Add checkpoint for fine-tuned model if specified
if [ -n "$CHECKPOINT_FINE_TUNED" ]; then
    COMPARISON_ARGS+=(--checkpoint_fine_tuned "$CHECKPOINT_FINE_TUNED")
fi

# Add checkpoint for new model if specified
if [ -n "$CHECKPOINT_NEW" ]; then
    COMPARISON_ARGS+=(--checkpoint_new "$CHECKPOINT_NEW")
fi

# Add dataset path if specified
if [ -n "$DATASET_PATH" ] && [ -d "$DATASET_PATH" ]; then
    COMPARISON_ARGS+=(--dataset_path "$DATASET_PATH")
fi

echo "Starting Whisper 3-model comparison pipeline..."
echo "Command: python scripts/run_whisper_comparison_all.py ${COMPARISON_ARGS[@]}"

# Run the comparison pipeline
python scripts/run_whisper_comparison_all.py "${COMPARISON_ARGS[@]}"

EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo "========================================="
    echo "Whisper 3-model comparison completed successfully!"
    echo "Results saved to: $OUTPUT_DIR"
    echo "Main CSV: $OUTPUT_DIR/whisper_comparison_all_results.csv"
    echo "Sample CSV: $OUTPUT_DIR/whisper_comparison_all_results_sample.csv"
    echo "Documentation: $OUTPUT_DIR/README.md"
    echo "Intermediate files: $OUTPUT_DIR/whisper_comparison_all_results_intermediate/"
    echo "========================================="
else
    echo "========================================="
    echo "Whisper 3-model comparison failed with exit code $EXIT_CODE"
    echo "Check logs for details"
    echo "========================================="
    exit $EXIT_CODE
fi

echo "Job completed at: $(date)"
echo "Total runtime: $SECONDS seconds"
