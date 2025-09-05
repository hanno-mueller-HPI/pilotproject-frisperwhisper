#!/bin/bash
#SBATCH --job-name=whisper_comparison
#SBATCH --account=aisc                      # AISC account for GPU access
#SBATCH --partition=aisc                    # Partition with GPUs
#SBATCH --qos=aisc                          # QoS setting for AISC
#SBATCH --output=logs/whisper_comparison_%j.out
#SBATCH --error=logs/whisper_comparison_%j.err
#SBATCH --time=120:00:00                    # (5 days)
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=112                 # 112 CPUs
#SBATCH --gres=gpu:4                        # 4 GPUs
#SBATCH --mem=600G                          # 600GB memory

# Load necessary modules
module load CUDA/11.8
module load Python/3.11.3

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
cd /sc/home/hanno.mueller/TTS-LangAge

# Activate virtual environment
echo "Activating virtual environment..."
source .venv/bin/activate

# Verify environment
echo "Python path: $(which python)"
echo "Working directory: $(pwd)"

# Set CUDA environment variables for 4 GPUs
export CUDA_VISIBLE_DEVICES=0,1,2,3
echo "Available GPUs: $CUDA_VISIBLE_DEVICES"

# Create logs directory if it doesn't exist
mkdir -p logs

# Configuration
INPUT_DIR="data/LangAge16kHz"
FINE_TUNED_MODEL="./FrisperWhisper/largeV3.1"
CHECKPOINT="checkpoint-5500"  # Use checkpoint-5500

# Automatically generate output directory name based on model and checkpoint
MODEL_NAME="largeV3.1"
CHECKPOINT_NUM="5500"
OUTPUT_DIR="results/${MODEL_NAME}-check${CHECKPOINT_NUM}"

# Processing configuration
NUM_CPUS=112
NUM_GPUS=4
BATCH_SIZE=64                           # Batch size for transcription
TRANSCRIPTION_PROCESSES=8               # Number of batch processes for transcription

echo "========================================="
echo "Whisper Comparison Configuration:"
echo "Input directory: $INPUT_DIR"
echo "Output directory: $OUTPUT_DIR"
echo "Fine-tuned model: $FINE_TUNED_MODEL"
if [ -n "$CHECKPOINT" ]; then
    echo "Checkpoint: $CHECKPOINT"
else
    echo "Checkpoint: final model"
fi
echo "CPUs: $NUM_CPUS"
echo "GPUs: $NUM_GPUS"
echo "Batch size: $BATCH_SIZE"
echo "Transcription processes: $TRANSCRIPTION_PROCESSES"
echo "========================================="

# Build command arguments for run_whisper_comparison_pipeline.py
COMPARISON_ARGS=(
    --input "$INPUT_DIR"
    --output "$OUTPUT_DIR"
    --fine_tuned_model "$FINE_TUNED_MODEL"
    --cpus "$NUM_CPUS"
    --gpus "$NUM_GPUS"
    --batch_size "$BATCH_SIZE"
    --transcription_batch_processes "$TRANSCRIPTION_PROCESSES"
    --steps "all"
)

# Add checkpoint if specified
if [ -n "$CHECKPOINT" ]; then
    COMPARISON_ARGS+=(--checkpoint "$CHECKPOINT")
fi

echo "Starting Whisper model comparison pipeline..."
echo "Command: python scripts/run_whisper_comparison_pipeline.py ${COMPARISON_ARGS[@]}"

# Run the comparison pipeline
python scripts/run_whisper_comparison_pipeline.py "${COMPARISON_ARGS[@]}"

EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo "========================================="
    echo "Whisper comparison completed successfully!"
    echo "Results saved to: $OUTPUT_DIR"
    echo "Main CSV: $OUTPUT_DIR/whisper_comparison_results.csv"
    echo "Sample CSV: $OUTPUT_DIR/whisper_comparison_results_sample.csv"
    echo "Documentation: $OUTPUT_DIR/README.md"
    echo "Intermediate files: $OUTPUT_DIR/whisper_comparison_results_intermediate/"
    echo "========================================="
else
    echo "========================================="
    echo "Whisper comparison failed with exit code $EXIT_CODE"
    echo "Check logs for details"
    echo "========================================="
    exit $EXIT_CODE
fi

echo "Job completed at: $(date)"
echo "Total runtime: $SECONDS seconds"
