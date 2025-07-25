#!/bin/bash

#############################################################################
# Example script to run Whisper fine-tuning with hardware configuration    #
#############################################################################

# Make sure virtual environment is activated
source .venv/bin/activate

# Example 1: Auto-detect resources (CPU-only, half cores)
echo "=== Example 1: Auto-detect resources (CPU-only, half cores) ==="
echo "python scripts/finetune_whisper.py -d LangAgeDataSet -o ./output_auto"
echo ""

# Example 2: Specify your exact hardware (2 CPUs, 2 GPUs)
echo "=== Example 2: Your hardware setup (2 CPUs, 2 GPUs) ==="
echo "python scripts/finetune_whisper.py -d LangAgeDataSet -o ./output_2cpu_2gpu \\"
echo "       --num_cpus 2 --num_gpus 2 --dataloader_workers 4"
echo ""

# Example 3: Conservative resource usage
echo "=== Example 3: Conservative resource usage ==="
echo "python scripts/finetune_whisper.py -d LangAgeDataSet -o ./output_conservative \\"
echo "       --num_cpus 1 --num_gpus 1 --dataloader_workers 2"
echo ""

# Example 4: CPU-only training
echo "=== Example 4: CPU-only training ==="
echo "python scripts/finetune_whisper.py -d LangAgeDataSet -o ./output_cpu_only \\"
echo "       --num_cpus 4 --num_gpus 0 --dataloader_workers 2"
echo ""

# Example 5: Maximum resources (use with caution)
echo "=== Example 5: Maximum resources ==="
echo "python scripts/finetune_whisper.py -d LangAgeDataSet -o ./output_max \\"
echo "       --num_cpus 8 --num_gpus 2 --dataloader_workers 8"
echo ""

echo "Choose the example that best fits your needs and hardware setup!"
echo ""
echo "For help on all available options:"
echo "python scripts/finetune_whisper.py --help"
