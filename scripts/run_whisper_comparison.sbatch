#!/bin/bash

#############################################################################
# Script Name: run_whisper_comparison.sbatch                               #
# Description: SLURM batch script for Whisper model comparison pipeline   #
# Author: Hanno Müller                                                     #
# Date: 2025-10-30                                                         #
#                                                                          #
# Features:                                                                #
# - Multi-GPU SLURM job execution                                         #
# - Environment variable passthrough to Python script                     #
# - Flexible resource allocation                                          #
# - Comprehensive logging and error handling                              #
#############################################################################

# SLURM configuration
#SBATCH --job-name=whisper-comparison-full
#SBATCH --output=logs/whisper_comparison_%j.out
#SBATCH --error=logs/whisper_comparison_%j.err
#SBATCH --time=60:00:00
#SBATCH --mem=400G
#SBATCH --cpus-per-task=16
#SBATCH --gpus=4
#SBATCH --partition=aisc
#SBATCH --account=aisc

# Default values (can be overridden by environment variables)
INPUT_DIR="${INPUT_DIR:-data/ESLOLangAgeCombined16kHz}"
OUTPUT_DIR="${OUTPUT_DIR:-results/whisper_comparison_$(date +%Y%m%d_%H%M%S)}"
FINE_TUNED_MODEL="${FINE_TUNED_MODEL:-FrisperWhisper/ESLOLangAgeCombined/checkpoint-4000}"
DATASET_PATH="${DATASET_PATH:-data/ESLOLangAgeDataSet}"
CPUS="${CPUS:-16}"
GPUS="${GPUS:-4}"
BATCH_SIZE="${BATCH_SIZE:-32}"
TRANSCRIPTION_BATCH_PROCESSES="${TRANSCRIPTION_BATCH_PROCESSES:-8}"
STEPS="${STEPS:-all}"
FILE_LIMIT="${FILE_LIMIT:-}"
RESUME_FROM_TRANSCRIPTIONS="${RESUME_FROM_TRANSCRIPTIONS:-}"

# Create logs directory if it doesn't exist
mkdir -p logs

echo "============================================================================"
echo "SLURM Whisper Model Comparison Pipeline"
echo "============================================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Started at: $(date)"
echo "Working directory: $(pwd)"
echo ""
echo "Configuration:"
echo "  Input directory: $INPUT_DIR"
echo "  Output directory: $OUTPUT_DIR" 
echo "  Fine-tuned model: $FINE_TUNED_MODEL"
echo "  Dataset path: $DATASET_PATH"
echo "  CPUs: $CPUS"
echo "  GPUs: $GPUS"
echo "  Batch size: $BATCH_SIZE"
echo "  Transcription processes: $TRANSCRIPTION_BATCH_PROCESSES"
echo "  Pipeline steps: $STEPS"
if [ -n "$FILE_LIMIT" ]; then
    echo "  File limit: $FILE_LIMIT"
fi
if [ -n "$RESUME_FROM_TRANSCRIPTIONS" ]; then
    echo "  Resume from: $RESUME_FROM_TRANSCRIPTIONS"
fi
echo "============================================================================"

# Activate Python environment
echo "Activating Python environment..."
source .venv/bin/activate

# Verify GPU availability
echo ""
echo "GPU Information:"
nvidia-smi --query-gpu=index,name,memory.total,memory.used --format=csv,noheader,nounits
echo ""

# Build Python command
PYTHON_CMD="python scripts/run_whisper_comparison.py"
PYTHON_CMD="$PYTHON_CMD --input \"$INPUT_DIR\""
PYTHON_CMD="$PYTHON_CMD --output \"$OUTPUT_DIR\""
PYTHON_CMD="$PYTHON_CMD --fine_tuned_model \"$FINE_TUNED_MODEL\""
PYTHON_CMD="$PYTHON_CMD --dataset \"$DATASET_PATH\""
PYTHON_CMD="$PYTHON_CMD --cpus $CPUS"
PYTHON_CMD="$PYTHON_CMD --gpus $GPUS"
PYTHON_CMD="$PYTHON_CMD --batch_size $BATCH_SIZE"
PYTHON_CMD="$PYTHON_CMD --transcription_batch_processes $TRANSCRIPTION_BATCH_PROCESSES"
PYTHON_CMD="$PYTHON_CMD --steps \"$STEPS\""

if [ -n "$FILE_LIMIT" ]; then
    PYTHON_CMD="$PYTHON_CMD --file_limit $FILE_LIMIT"
fi

if [ -n "$RESUME_FROM_TRANSCRIPTIONS" ]; then
    PYTHON_CMD="$PYTHON_CMD --resume_from_transcriptions \"$RESUME_FROM_TRANSCRIPTIONS\""
fi

echo "Executing command:"
echo "$PYTHON_CMD"
echo ""

# Record start time
START_TIME=$(date +%s)

# Execute the Python script
eval $PYTHON_CMD
EXIT_CODE=$?

# Record end time and calculate duration
END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
HOURS=$((DURATION / 3600))
MINUTES=$(((DURATION % 3600) / 60))
SECONDS=$((DURATION % 60))

echo ""
echo "============================================================================"
echo "Pipeline completed at: $(date)"
echo "Exit code: $EXIT_CODE"
echo "Total duration: ${HOURS}h ${MINUTES}m ${SECONDS}s"

if [ $EXIT_CODE -eq 0 ]; then
    echo "✓ Pipeline completed successfully"
    if [ -d "$OUTPUT_DIR" ]; then
        echo "✓ Results saved to: $OUTPUT_DIR"
        echo ""
        echo "Output files:"
        ls -la "$OUTPUT_DIR"
    fi
else
    echo "✗ Pipeline failed with exit code $EXIT_CODE"
    echo "Check error logs for details"
fi

echo "============================================================================"

# Deactivate environment
deactivate

exit $EXIT_CODE