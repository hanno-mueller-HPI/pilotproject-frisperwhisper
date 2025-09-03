#!/bin/bash
#############################################################################
# Test script for full Whisper comparison pipeline with GPU acceleration   #
# Description: Test complete pipeline utilizing available GPUs and CPUs     #
# Author: Hanno Müller                                                      #
# Date: 2025-09-03                                                          #
#############################################################################

echo "Testing Full Whisper Comparison Pipeline (GPU Accelerated)"
echo "=========================================================="

# Check SLURM allocation
echo "SLURM Resource Allocation:"
echo "  Job ID: $SLURM_JOB_ID"
echo "  CPUs allocated: $SLURM_CPUS_PER_TASK"
echo "  GPUs allocated: $SLURM_GPUS_PER_TASK"
echo "  Nodes: $SLURM_JOB_NODELIST"
echo ""

# Activate virtual environment
source .venv/bin/activate

# Set paths
INPUT_DIR="tmp/LangAge16kHz"
OUTPUT_FILE="test_full_gpu_results_v2.csv"
FINE_TUNED_MODEL="FrisperWhisper/largeV3"
CHECKPOINT="${1:-checkpoint-6000}"  # Default to checkpoint-6000

# GPU Configuration
GPUS=4  # Use all 4 available GPUs
CPUS=32  # Use most of available CPUs, leaving some for system
BATCH_SIZE=8  # More conservative batch size for stability
TRANSCRIPTION_PROCESSES=1  # Single process to avoid conflicts

echo "Input directory: $INPUT_DIR"
echo "Output file: $OUTPUT_FILE"
echo "Fine-tuned model: $FINE_TUNED_MODEL"
echo "Checkpoint: $CHECKPOINT"
echo "Configuration: $GPUS GPUs, $CPUS CPUs, batch size $BATCH_SIZE"
echo "Transcription processes: $TRANSCRIPTION_PROCESSES"
echo ""

# Check if GPUs are available
if command -v nvidia-smi >/dev/null 2>&1; then
    echo "GPU Status:"
    nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits
    echo ""
fi

# Run the complete pipeline with GPU acceleration
echo "Starting GPU-accelerated pipeline..."
python scripts/run_whisper_comparison_pipeline.py \
    --input "$INPUT_DIR" \
    --output "$OUTPUT_FILE" \
    --fine_tuned_model "$FINE_TUNED_MODEL" \
    --checkpoint "$CHECKPOINT" \
    --cpus "$CPUS" \
    --gpus "$GPUS" \
    --batch_size "$BATCH_SIZE" \
    --transcription_batch_processes "$TRANSCRIPTION_PROCESSES" \
    --steps all

echo ""
echo "Test Results:"
echo "============"

if [ -f "$OUTPUT_FILE" ]; then
    echo "✅ Output file created: $OUTPUT_FILE"
    
    # Count rows (excluding header)
    TOTAL_ROWS=$(tail -n +2 "$OUTPUT_FILE" | wc -l)
    echo "📊 Total data rows: $TOTAL_ROWS"
    
    # Show processing time statistics
    echo ""
    echo "📈 Performance Summary:"
    echo "   GPU Configuration: $GPUS GPUs, batch size $BATCH_SIZE"
    echo "   CPU Configuration: $CPUS CPUs, $TRANSCRIPTION_PROCESSES processes"
    
    # Show column headers
    echo ""
    echo "📋 Columns:"
    head -n 1 "$OUTPUT_FILE" | tr ',' '\n' | nl
    
    echo ""
    echo "📄 First few rows:"
    head -n 3 "$OUTPUT_FILE"
    
    echo ""
    echo "🎯 Sample metrics from first data row:"
    if [ $TOTAL_ROWS -gt 0 ]; then
        FIRST_ROW=$(tail -n +2 "$OUTPUT_FILE" | head -n 1)
        echo "   Filename: $(echo "$FIRST_ROW" | cut -d',' -f1)"
        echo "   Speaker: $(echo "$FIRST_ROW" | cut -d',' -f2)"
        echo "   Gender: $(echo "$FIRST_ROW" | cut -d',' -f3)"
        echo "   Duration: $(echo "$FIRST_ROW" | cut -d',' -f6) seconds"
        echo "   WER (Large V3 vs Original): $(echo "$FIRST_ROW" | cut -d',' -f11)"
        echo "   WER (Fine-tuned vs Original): $(echo "$FIRST_ROW" | cut -d',' -f12)"
        echo "   BLEU (Large V3 vs Original): $(echo "$FIRST_ROW" | cut -d',' -f17)"
        echo "   BLEU (Fine-tuned vs Original): $(echo "$FIRST_ROW" | cut -d',' -f18)"
    fi
    
    echo ""
    echo "🔍 Files processed:"
    tail -n +2 "$OUTPUT_FILE" | cut -d',' -f1 | sort | uniq -c
    
else
    echo "❌ Output file not created"
fi

# Check intermediate directory
if [ -d "test_full_gpu_results_intermediate" ]; then
    echo ""
    echo "📁 Intermediate files:"
    ls -la test_full_gpu_results_intermediate/
else
    echo ""
    echo "📁 No intermediate directory found"
fi

echo ""
echo "🚀 GPU-accelerated pipeline test completed!"
echo ""
echo "Usage:"
echo "   ./scripts/test_full_pipeline_gpu.sh                    # Test with checkpoint-6000"
echo "   ./scripts/test_full_pipeline_gpu.sh checkpoint-8000    # Test with specific checkpoint"
