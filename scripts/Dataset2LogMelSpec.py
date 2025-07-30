#############################################################################
# Script Name: Dataset2LogMelSpec.py                                        #
# Description: Preprocess LangAge dataset to log-Mel spectrograms           #
# Author: Hanno Müller                                                      #
# Date: 2025-07-25                                                          #
#############################################################################

### Required Libraries ######################################################
import os
import argparse
import multiprocessing
from typing import Dict, Any

from transformers import (
    WhisperFeatureExtractor,
    WhisperTokenizer
)
from datasets import DatasetDict, Dataset, load_from_disk


### Function Definitions #####################################################

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Preprocess LangAge dataset to log-Mel spectrograms for Whisper training.")
    parser.add_argument(
        "-i", "--input_dataset",
        type=str,
        required=True,
        help="Path to the input LangAgeDataSet folder (HuggingFace dataset)"
    )
    parser.add_argument(
        "-o", "--output_dataset", 
        type=str,
        required=True,
        help="Path where the preprocessed dataset will be saved"
    )
    parser.add_argument(
        "--num_cpus",
        type=int,
        default=None,
        help="Number of CPU cores to use for preprocessing. If not specified, will use all available cores."
    )
    parser.add_argument(
        "--model_size",
        type=str,
        default="large",
        choices=["tiny", "base", "small", "medium", "large"],
        help="Whisper model size for feature extraction (default: large)"
    )
    parser.add_argument(
        "--shuffle_seed",
        type=int,
        default=42,
        help="Random seed for shuffling datasets (default: 42)"
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Maximum number of samples to process per split (for testing). If not specified, processes all samples."
    )
    return parser.parse_args()


def configure_cpus(args):
    """Configure CPU resources based on user specifications and SLURM environment."""
    print("=== CPU Configuration ===")
    
    # Check if running under SLURM
    slurm_cpus = os.environ.get('SLURM_CPUS_PER_TASK')
    slurm_job_id = os.environ.get('SLURM_JOB_ID')
    
    if slurm_job_id:
        print(f"Running under SLURM (Job ID: {slurm_job_id})")
        if slurm_cpus:
            slurm_cpu_count = int(slurm_cpus)
            print(f"SLURM allocated CPUs: {slurm_cpu_count}")
        else:
            slurm_cpu_count = None
    else:
        print("Not running under SLURM")
        slurm_cpu_count = None
    
    available_cpus = multiprocessing.cpu_count()
    print(f"Available CPUs on this node: {available_cpus}")
    
    if args.num_cpus is not None:
        requested_cpus = args.num_cpus
        print(f"User requested CPUs: {requested_cpus}")
        
        # Determine the actual limit
        if slurm_cpu_count:
            max_cpus = min(slurm_cpu_count, available_cpus)
            limit_source = "SLURM allocation"
        else:
            max_cpus = available_cpus
            limit_source = "available CPUs"
        
        if requested_cpus > max_cpus:
            print(f"Warning: Requested {requested_cpus} CPUs, but only {max_cpus} {limit_source}. Using {max_cpus}.")
            num_cpus = max_cpus
        elif requested_cpus <= 0:
            print("Error: Number of CPUs must be positive.")
            raise ValueError("Invalid number of CPUs specified")
        else:
            num_cpus = requested_cpus
    else:
        # Auto-detect mode: use SLURM allocation if available, otherwise all available
        if slurm_cpu_count:
            num_cpus = min(slurm_cpu_count, available_cpus)
            print(f"Auto-detect mode: using {num_cpus} CPUs (SLURM allocation)")
        else:
            num_cpus = available_cpus
            print(f"Auto-detect mode: using {num_cpus} CPUs (all available)")
    
    print(f"Final configuration: using {num_cpus} CPU cores for preprocessing")
    print("=" * 30)
    
    return num_cpus


def prepare_dataset(batch, feature_extractor, tokenizer):
    """
    Prepare dataset batch for Whisper training.
    Converts audio to log-Mel spectrograms and text to token IDs.
    """
    # Ensure audio is at 16kHz (should already be from LangAge dataset)
    audio = batch["audio"]
    
    # Compute log-Mel spectrogram input features from audio array
    batch["input_features"] = feature_extractor(
        audio["array"], 
        sampling_rate=audio["sampling_rate"]
    ).input_features[0]
    
    # Encode target text to label ids
    batch["labels"] = tokenizer(batch["sentence"]).input_ids
    
    return batch


### main ######################################################################

if __name__ == "__main__":
    print("Starting dataset preprocessing to log-Mel spectrograms...")
    
    # Parse command line arguments
    args = parse_arguments()
    
    # Configure CPU resources
    num_cpus = configure_cpus(args)
    
    # Load the LangAge dataset from disk
    print(f"Loading dataset from {args.input_dataset}")
    try:
        # Try loading as DatasetDict first
        try:
            dataset_dict = DatasetDict.load_from_disk(args.input_dataset)
            print(f"Loaded as DatasetDict. Splits: {list(dataset_dict.keys())}")
        except Exception:
            dataset_obj = load_from_disk(args.input_dataset)
            if isinstance(dataset_obj, Dataset):
                print("Loaded as single Dataset.")
                dataset_dict = DatasetDict({"train": dataset_obj})
            else:
                raise ValueError("Could not load dataset as Dataset or DatasetDict.")
        
        print(f"Dataset loaded successfully!")
        print(f"Available splits: {list(dataset_dict.keys())}")
        
        # Check for required splits
        required_splits = ["train", "test"]
        for split in required_splits:
            if split not in dataset_dict:
                print(f"Warning: No '{split}' split found in dataset!")
        
        # Display dataset sizes
        for split_name, split_data in dataset_dict.items():
            print(f"{str(split_name).capitalize()} dataset size: {len(split_data)} samples")
        
    except Exception as e:
        print(f"Error loading dataset: {e}")
        raise
    
    # Initialize Whisper components for preprocessing
    model_name = f"openai/whisper-{args.model_size}"
    print(f"Loading Whisper components for model: {model_name}")
    
    feature_extractor = WhisperFeatureExtractor.from_pretrained(model_name)
    tokenizer = WhisperTokenizer.from_pretrained(model_name, language="French", task="transcribe")
    
    print("Whisper components loaded successfully!")
    
    # Prepare preprocessing function
    def prepare_dataset_with_processors(batch):
        return prepare_dataset(batch, feature_extractor, tokenizer)
    
    # Process each split in the dataset
    processed_dataset_dict = DatasetDict()
    
    for split_name, split_dataset in dataset_dict.items():
        print(f"\nProcessing {split_name} split...")
        print(f"Original columns: {split_dataset.column_names}")
        print(f"Original dataset size: {len(split_dataset)} samples")
        
        # Limit samples if max_samples is specified
        if args.max_samples is not None and len(split_dataset) > args.max_samples:
            print(f"Limiting to {args.max_samples} samples for testing...")
            split_dataset = split_dataset.select(range(args.max_samples))
            print(f"Limited dataset size: {len(split_dataset)} samples")
        
        # Shuffle dataset if it's the train split
        if split_name == "train":
            print(f"Shuffling {split_name} split with seed {args.shuffle_seed}")
            split_dataset = split_dataset.shuffle(seed=args.shuffle_seed)
        
        # Apply preprocessing - use single process for small datasets
        if len(split_dataset) <= 100:
            print(f"Small dataset detected. Using single process...")
            processed_split = split_dataset.map(
                prepare_dataset_with_processors,
                remove_columns=split_dataset.column_names,
                desc=f"Processing {split_name} split"
            )
        else:
            print(f"Processing with {num_cpus} processes...")
            processed_split = split_dataset.map(
                prepare_dataset_with_processors,
                remove_columns=split_dataset.column_names,
                num_proc=num_cpus,
                desc=f"Processing {split_name} split"
            )
        
        print(f"Processed {split_name} split: {len(processed_split)} samples")
        print(f"New columns: {processed_split.column_names}")
        
        # Validate processed data
        sample = processed_split[0]
        input_features_shape = sample["input_features"].shape if hasattr(sample["input_features"], 'shape') else len(sample["input_features"])
        labels_length = len(sample["labels"])
        print(f"Sample input_features shape: {input_features_shape}")
        print(f"Sample labels length: {labels_length}")
        
        processed_dataset_dict[split_name] = processed_split
    
    # Save the preprocessed dataset
    print(f"\nSaving preprocessed dataset to {args.output_dataset}")
    processed_dataset_dict.save_to_disk(args.output_dataset)
    
    print("Dataset preprocessing completed successfully!")
    print(f"Preprocessed dataset saved to: {args.output_dataset}")
    
    # Final summary
    print("\n" + "=" * 50)
    print("PREPROCESSING SUMMARY")
    print("=" * 50)
    print(f"Input dataset: {args.input_dataset}")
    print(f"Output dataset: {args.output_dataset}")
    print(f"Whisper model: {model_name}")
    print(f"CPU cores used: {num_cpus}")
    print(f"Processed splits: {list(processed_dataset_dict.keys())}")
    for split_name, split_data in processed_dataset_dict.items():
        print(f"  {split_name}: {len(split_data)} samples")
    print("=" * 50)
