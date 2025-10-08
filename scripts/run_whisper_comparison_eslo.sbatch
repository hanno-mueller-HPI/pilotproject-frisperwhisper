#!/bin/bash
#SBATCH --job-name=whisper_eslo
#SBATCH --account=aisc-staff                      # AISC account for GPU access
#SBATCH --partition=aisc                    # Partition with GPUs
#SBATCH --output=logs/whisper_eslo_%j.out
#SBATCH --error=logs/whisper_eslo_%j.err
#SBATCH --time=120:00:00                    # (5 days)
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=56                 # 56 CPUs
#SBATCH --gres=gpu:2                        # 2 GPUs
#SBATCH --mem=600G                          # 600GB memory

# Parse command line arguments
INPUT_DIR=${1:-"data/sampleESLO30-39"}
OUTPUT_DIR=${2:-"results/ESLO30-39"}

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

# Set CUDA environment variables for 2 GPUs
export CUDA_VISIBLE_DEVICES=0,1
echo "Available GPUs: $CUDA_VISIBLE_DEVICES"

# Create logs directory if it doesn't exist
mkdir -p logs

# Configuration
FINE_TUNED_MODEL="./FrisperWhisper/largeV3.2"
CHECKPOINT="checkpoint-2000"  # Use checkpoint-2000

# Processing configuration
NUM_CPUS=56
NUM_GPUS=2
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

# Validate input directory exists
if [ ! -d "$INPUT_DIR" ]; then
    echo "ERROR: Input directory '$INPUT_DIR' does not exist!"
    exit 1
fi

# Build command arguments for run_whisper_comparison.py
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
echo "Command: python scripts/run_whisper_comparison.py ${COMPARISON_ARGS[@]}"

# Run the comparison pipeline
python scripts/run_whisper_comparison.py "${COMPARISON_ARGS[@]}"

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
