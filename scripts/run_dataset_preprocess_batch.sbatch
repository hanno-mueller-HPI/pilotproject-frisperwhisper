#!/bin/bash
#SBATCH --job-name=dataset_preprocess_batch_full
#SBATCH --account=aisc                      # AISC account for H100 access
#SBATCH --partition=aisc                    # Partition with H100 GPUs
#SBATCH --output=logs/Dataset2LogMelSpec/dataset_preprocess_batch_full_%j.out
#SBATCH --error=logs/Dataset2LogMelSpec/dataset_preprocess_batch_full_%j.err
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=100
#SBATCH --mem=800G
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
cd /sc/home/hanno.mueller/pilotproject-frisperwhisper

# Activate virtual environment
echo "Activating virtual environment..."
source .venv/bin/activate

# Verify environment
echo "Python path: $(which python)"
echo "Working directory: $(pwd)"

# Create logs directory if it doesn't exist
mkdir -p logs/Dataset2LogMelSpec

# ==============================
# Configuration with Environment Variables
# ==============================

# Set default values if environment variables are not provided
INPUT_DATASET="${INPUT_DATASET:-data/LangAgeDataSet}"
OUTPUT_DATASET="${OUTPUT_DATASET:-data/LangAgeLogMelSpec}"
NUM_CPUS="${NUM_CPUS:-100}"
BATCH_SIZE="${BATCH_SIZE:-400}"
MODEL_SIZE="${MODEL_SIZE:-large-v3}"
WRITER_BATCH_SIZE="${WRITER_BATCH_SIZE:-100}"
MAX_MEMORY_PER_WORKER="${MAX_MEMORY_PER_WORKER:-4.0}"
SHUFFLE_SEED="${SHUFFLE_SEED:-42}"
MAX_SAMPLES="${MAX_SAMPLES:-}"  # Optional, empty by default

# ==============================
# Run Dataset Preprocessing
# ==============================

echo "Starting memory-optimized dataset preprocessing..."
echo "Configuration:"
echo "  Input dataset: ${INPUT_DATASET}"
echo "  Output dataset: ${OUTPUT_DATASET}"
echo "  Number of CPUs: ${NUM_CPUS}"
echo "  Batch size: ${BATCH_SIZE}"
echo "  Model size: ${MODEL_SIZE}"
echo "  Writer batch size: ${WRITER_BATCH_SIZE}"
echo "  Max memory per worker: ${MAX_MEMORY_PER_WORKER} GB"
echo "  Shuffle seed: ${SHUFFLE_SEED}"
if [ -n "${MAX_SAMPLES}" ]; then
    echo "  Max samples: ${MAX_SAMPLES}"
fi
echo "=============================="

# Build command with optional max_samples parameter
CMD="python scripts/Dataset2LogMelSpecBatch.py -i ${INPUT_DATASET} -o ${OUTPUT_DATASET} --num_cpus ${NUM_CPUS} --batch_size ${BATCH_SIZE} --model_size ${MODEL_SIZE} --writer_batch_size ${WRITER_BATCH_SIZE} --max_memory_per_worker ${MAX_MEMORY_PER_WORKER} --shuffle_seed ${SHUFFLE_SEED}"
if [ -n "${MAX_SAMPLES}" ]; then
    CMD="${CMD} --max_samples ${MAX_SAMPLES}"
fi

echo "Command: ${CMD}"
echo "=============================="

# Execute the command
${CMD}

echo "Job completed at: $(date)"
echo "Exit code: $?"
