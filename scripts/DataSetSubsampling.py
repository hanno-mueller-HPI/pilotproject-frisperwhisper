#!/usr/bin/env python3

#############################################################################
# Script Name: DataSetSubsampling.py                                       #
# Description: Create a subsampled version of a dataset                    #
# Author: Hanno Müller                                                      #
# Date: 2025-08-25                                                          #
#############################################################################

import os
import argparse
import multiprocessing
from datasets import load_from_disk, DatasetDict
import random


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Create a subsampled version of a dataset."
    )
    
    parser.add_argument(
        "-i", "--input_dataset",
        type=str,
        required=True,
        help="Path to the input dataset folder"
    )
    
    parser.add_argument(
        "-o", "--output_dataset",
        type=str,
        required=True,
        help="Base path for the output dataset folder (subset size will be appended)"
    )
    
    parser.add_argument(
        "--num_cpus",
        type=int,
        default=max(1, int(multiprocessing.cpu_count() * 2 / 3)),
        help="Number of CPUs to use for multiprocessing (default: 2/3 of available CPUs)"
    )
    
    parser.add_argument(
        "--shuffle-seed",
        type=int,
        default=42,
        help="Seed for randomization (default: 42)"
    )
    
    parser.add_argument(
        "--subset-size",
        type=int,
        default=10,
        choices=range(1, 101),
        metavar="[1-100]",
        help="Percentage of original dataset to keep (1-100, default: 10)"
    )
    
    args = parser.parse_args()
    
    # Ensure num_cpus doesn't exceed 2/3 of available CPUs
    max_allowed_cpus = max(1, int(multiprocessing.cpu_count() * 2 / 3))
    if args.num_cpus > max_allowed_cpus:
        print(f"Warning: Requested {args.num_cpus} CPUs, but limiting to {max_allowed_cpus} (2/3 of {multiprocessing.cpu_count()} available CPUs)")
        args.num_cpus = max_allowed_cpus
    
    return args


def load_dataset(input_path):
    """Load the dataset from disk."""
    print(f"Loading dataset from: {input_path}")
    
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input dataset not found: {input_path}")
    
    dataset = load_from_disk(input_path)
    
    print(f"Dataset loaded successfully!")
    print(f"Dataset structure: {dataset}")
    
    return dataset


def subsample_split(split_data, subset_percentage, seed, split_name):
    """Subsample a single split of the dataset."""
    print(f"Subsampling {split_name} split...")
    
    original_size = len(split_data)
    target_size = max(1, int(original_size * subset_percentage / 100))
    
    print(f"  Original {split_name} size: {original_size}")
    print(f"  Target {split_name} size: {target_size} ({subset_percentage}%)")
    
    # Shuffle and subsample
    shuffled_data = split_data.shuffle(seed=seed)
    subsampled_data = shuffled_data.select(range(target_size))
    
    print(f"  Final {split_name} size: {len(subsampled_data)}")
    
    return subsampled_data


def create_subsampled_dataset(dataset, subset_percentage, seed, num_cpus):
    """Create a subsampled version of the dataset."""
    print(f"Creating subsampled dataset ({subset_percentage}%)...")
    
    subsampled_dataset = DatasetDict()
    
    # Process each split
    for split_name, split_data in dataset.items():
        print(f"\nProcessing {split_name} split...")
        subsampled_split = subsample_split(
            split_data, 
            subset_percentage, 
            seed, 
            split_name
        )
        subsampled_dataset[split_name] = subsampled_split
    
    return subsampled_dataset


def save_dataset(dataset, output_path, num_cpus):
    """Save the subsampled dataset to disk."""
    print(f"Saving subsampled dataset to: {output_path}")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_path, exist_ok=True)
    
    # Calculate appropriate number of processes based on dataset size
    min_dataset_size = min(len(split) for split in dataset.values())
    effective_num_cpus = min(num_cpus, min_dataset_size, multiprocessing.cpu_count())
    
    if effective_num_cpus != num_cpus:
        print(f"  Reducing number of processes from {num_cpus} to {effective_num_cpus} due to small dataset size")
    
    # Save dataset
    if effective_num_cpus <= 1:
        dataset.save_to_disk(output_path)
    else:
        dataset.save_to_disk(output_path, num_proc=effective_num_cpus)
    
    print(f"Dataset saved successfully!")


def print_dataset_summary(dataset, title):
    """Print a summary of the dataset."""
    print(f"\n=== {title} ===")
    print(f"Dataset structure: {dataset}")
    
    total_samples = 0
    for split_name, split_data in dataset.items():
        split_size = len(split_data)
        total_samples += split_size
        print(f"  {split_name}: {split_size} samples")
        
        # Show column names for the first split
        if split_name == list(dataset.keys())[0]:
            print(f"  Columns: {list(split_data.features.keys())}")
    
    print(f"  Total: {total_samples} samples")
    print("=" * (len(title) + 8))


def main():
    """Main function."""
    args = parse_arguments()
    
    # Set random seed for reproducibility
    random.seed(args.shuffle_seed)
    
    # Create output path with subset size appended
    output_path = f"{args.output_dataset}{args.subset_size}"
    
    print("=== Dataset Subsampling Tool ===")
    print(f"Input dataset: {args.input_dataset}")
    print(f"Output dataset: {output_path}")
    print(f"Subset size: {args.subset_size}%")
    print(f"Shuffle seed: {args.shuffle_seed}")
    print(f"Number of CPUs: {args.num_cpus}")
    print("=" * 40)
    
    try:
        # Load original dataset
        dataset = load_dataset(args.input_dataset)
        print_dataset_summary(dataset, "Original Dataset")
        
        # Create subsampled dataset
        subsampled_dataset = create_subsampled_dataset(
            dataset, 
            args.subset_size, 
            args.shuffle_seed, 
            args.num_cpus
        )
        print_dataset_summary(subsampled_dataset, "Subsampled Dataset")
        
        # Save subsampled dataset
        save_dataset(subsampled_dataset, output_path, args.num_cpus)
        
        print(f"\n✅ Subsampling completed successfully!")
        print(f"📁 Subsampled dataset saved to: {output_path}")
        print(f"📊 Dataset reduced to {args.subset_size}% of original size")
        
        # Final verification
        print(f"\n=== Verification ===")
        verification_dataset = load_from_disk(output_path)
        print_dataset_summary(verification_dataset, "Saved Dataset")
        
    except Exception as e:
        print(f"\n❌ Error during subsampling: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
