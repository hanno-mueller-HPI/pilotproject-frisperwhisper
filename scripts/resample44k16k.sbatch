#!/bin/bash
#SBATCH --job-name=resample_44k_to_16k
#SBATCH --account=aisc                      # AISC account for compute access
#SBATCH --partition=aisc                    # Partition for compute jobs
#SBATCH --output=logs/resample_44k_to_16k_%j.out
#SBATCH --error=logs/resample_44k_to_16k_%j.err
#SBATCH --time=24:00:00                     # Default 24 hours
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=60                  # Default 60 CPUs
#SBATCH --mem=200G                           # Default 200GB memory
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
mkdir -p logs

# ==============================
# Configuration with Environment Variables
# ==============================

# Set default values if environment variables are not provided
INPUT_FOLDER="${INPUT_FOLDER:-}"            # Required parameter
OUTPUT_FOLDER="${OUTPUT_FOLDER:-}"          # Not used in this script (files are overwritten)
NUM_PROCESSES="${NUM_PROCESSES:-$SLURM_CPUS_PER_TASK}"  # Use allocated CPUs by default

# ==============================
# Validate Parameters
# ==============================

if [ -z "${INPUT_FOLDER}" ]; then
    echo "ERROR: INPUT_FOLDER environment variable is required!"
    echo ""
    echo "Usage examples:"
    echo "  # Set parameters via environment variables:"
    echo "  export INPUT_FOLDER='data_eslo'"
    echo "  export NUM_PROCESSES=8"
    echo "  sbatch scripts/resample44k16k.sbatch"
    echo ""
    echo "  # Or set them inline:"
    echo "  INPUT_FOLDER='data_eslo' NUM_PROCESSES=8 sbatch scripts/resample44k16k.sbatch"
    echo ""
    echo "  # Or modify SLURM parameters and use defaults:"
    echo "  sbatch --cpus-per-task=8 --mem=16G --time=01:00:00 scripts/resample44k16k.sbatch"
    echo ""
    exit 1
fi

if [ ! -d "${INPUT_FOLDER}" ]; then
    echo "ERROR: Input folder '${INPUT_FOLDER}' does not exist!"
    exit 1
fi

# ==============================
# Run Audio Resampling
# ==============================

echo "Starting audio resampling from 44kHz to 16kHz..."
echo "Configuration:"
echo "  Input folder: ${INPUT_FOLDER}"
echo "  Number of processes: ${NUM_PROCESSES}"
echo "  Files will be overwritten with 16kHz versions"
echo "=============================="

# Count wav files before processing
WAV_COUNT=$(find "${INPUT_FOLDER}" -name "*.wav" | wc -l)
echo "Found ${WAV_COUNT} .wav files to process"

if [ ${WAV_COUNT} -eq 0 ]; then
    echo "No .wav files found in '${INPUT_FOLDER}'. Nothing to process."
    exit 0
fi

echo "=============================="

# Execute the resampling script
python scripts/resample44k16k.py \
    --input_folder "${INPUT_FOLDER}" \
    --processes "${NUM_PROCESSES}"

EXIT_CODE=$?

echo "=============================="
echo "Job completed at: $(date)"
echo "Exit code: ${EXIT_CODE}"

if [ ${EXIT_CODE} -eq 0 ]; then
    echo "Audio resampling completed successfully!"
    echo "All .wav files in '${INPUT_FOLDER}' have been resampled to 16kHz"
else
    echo "Audio resampling failed with exit code ${EXIT_CODE}"
    exit ${EXIT_CODE}
fi