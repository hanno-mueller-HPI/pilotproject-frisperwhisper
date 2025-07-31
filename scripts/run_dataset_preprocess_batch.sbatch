#!/bin/bash
#SBATCH --job-name=dataset_preprocess_batch_full
#SBATCH --account=aisc                      # AISC account for H100 access
#SBATCH --partition=aisc                    # Partition with H100 GPUs
#SBATCH --output=logs/Dataset2LogMelSpec/dataset_preprocess_batch_full_%j.out
#SBATCH --error=logs/Dataset2LogMelSpec/dataset_preprocess_batch_full_%j.err
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=98
#SBATCH --mem=512G
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

# Run the full dataset preprocessing with memory optimization
echo "Starting memory-optimized dataset preprocessing (full dataset)..."
echo "Command: python scripts/Dataset2LogMelSpecBatch.py -i LangAgeDataSet -o LangAgeDataSet_preprocessed_batch --num_cpus 98 --model_size large --batch_size 500 --max_memory_per_worker 3.0"

python scripts/Dataset2LogMelSpecBatch.py \
    -i LangAgeDataSet \
    -o LangAgeDataSet_preprocessed_batch \
    --num_cpus 98 \
    --model_size large \
    --batch_size 500 \
    --max_memory_per_worker 3.0

echo "Job completed at: $(date)"
echo "Exit code: $?"
