#!/bin/bash

#############################################################################
# Script Name: run_distributed_training.sh                                 #
# Description: Launch multi-GPU distributed Whisper fine-tuning            #
# Author: Hanno Müller                                                      #
# Date: 2025-07-30                                                          #
#############################################################################

# Set environment variables for distributed training
export CUDA_VISIBLE_DEVICES=0,1,2,3
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:512

# Number of GPUs to use
NUM_GPUS=4

# Training arguments
INPUT_DATASET="${1:-LangAgeDataSet_preprocessed_test/}"
OUTPUT_DIR="${2:-whisper_refined_4gpu}"
MODEL_SIZE="${3:-tiny}"
MAX_STEPS="${4:-100}"
BATCH_SIZE="${5:-1}"  # Further reduced batch size for memory optimization
LOGGING_STEPS="${6:-10}"

echo "=== Multi-GPU Distributed Training Setup ==="
echo "Number of GPUs: $NUM_GPUS"
echo "Input Dataset: $INPUT_DATASET"
echo "Output Directory: $OUTPUT_DIR"
echo "Model Size: $MODEL_SIZE"
echo "Max Steps: $MAX_STEPS"
echo "Batch Size per GPU: $BATCH_SIZE"
echo "Logging Steps: $LOGGING_STEPS"
echo "============================================="

# Run distributed training using torchrun
torchrun \
    --nproc_per_node=$NUM_GPUS \
    --master_port=29500 \
    scripts/finetune_whisper_gpu.py \
    --input_dataset "$INPUT_DATASET" \
    --output_dir "$OUTPUT_DIR" \
    --num_gpus $NUM_GPUS \
    --num_cpus 20 \
    --model_size "$MODEL_SIZE" \
    --per_device_train_batch_size $BATCH_SIZE \
    --max_steps $MAX_STEPS \
    --logging_steps $LOGGING_STEPS \
    --eval_steps 50 \
    --save_steps 50 \
    --dataloader_workers 2

echo "Multi-GPU training completed!"
