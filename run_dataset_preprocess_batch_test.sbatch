#!/bin/bash
#SBATCH --job-name=dataset_preprocess_batch_test
#SBATCH --output=logs/Dataset2LogMelSpec/dataset_preprocess_batch_test_%j.out
#SBATCH --error=logs/Dataset2LogMelSpec/dataset_preprocess_batch_test_%j.err
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --constraint=ARCH:X86

echo "=== SLURM Job Information ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURMD_NODENAME"
echo "Start time: $(date)"
echo "CPUs allocated: $SLURM_CPUS_PER_TASK"
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

# Create logs directory if it doesn't exist
mkdir -p logs/Dataset2LogMelSpec

# Run the dataset preprocessing with memory optimization
echo "Starting memory-optimized dataset preprocessing..."
echo "Command: python scripts/Dataset2LogMelSpecBatch.py -i LangAgeDataSet -o LangAgeDataSet_preprocessed_batch_test --max_samples 20 --model_size large --batch_size 100 --max_memory_per_worker 2.0"

python scripts/Dataset2LogMelSpecBatch.py \
    -i LangAgeDataSet \
    -o LangAgeDataSet_preprocessed_batch_test \
    --max_samples 20 \
    --model_size large \
    --batch_size 100 \
    --max_memory_per_worker 2.0

echo "Job completed at: $(date)"
echo "Exit code: $?"
