#!/usr/bin/env python3
#############################################################################
# Script Name: update_comparison.py                                        #
# Description: Update whisper comparison CSV with HuggingFace dataset      #
#              train/test splits and add dataset type columns              #
# Author: Hanno Müller                                                      #
# Date: 2025-11-03                                                          #
#############################################################################

import os
import sys
import pandas as pd
import argparse
from pathlib import Path
import re
import chardet
from collections import defaultdict
from multiprocessing import Pool, cpu_count
from functools import partial
import numpy as np

# Add modules to path for TextGrid parsing
sys.path.append(os.path.join(os.path.dirname(__file__), 'modules'))
from metadata_extraction import TextGrid
from speaker_mapping import create_speaker_mapping, get_speaker_info

try:
    from datasets import load_from_disk
    HF_DATASETS_AVAILABLE = True
except ImportError:
    print("Warning: HuggingFace datasets not available. Install with: pip install datasets")
    HF_DATASETS_AVAILABLE = False


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Update CSV with HuggingFace dataset train/test splits and add dataset type columns"
    )
    
    parser.add_argument(
        "--input_csv", "-i",
        type=str,
        required=True,
        help="Path to input CSV file (e.g., whisper_comparison_results.csv)"
    )
    parser.add_argument(
        "--input_dataset", "-d",
        type=str,
        required=True,
        help="Path to HuggingFace dataset directory (e.g., data/ESLOLangAgeDataSet)"
    )
    parser.add_argument(
        "--output_csv", "-o", 
        type=str,
        required=True,
        help="Path to output CSV file"
    )
    parser.add_argument(
        "--workers", "-w",
        type=int,
        default=cpu_count(),
        help=f"Number of workers for parallel processing (default: {cpu_count()})"
    )
    parser.add_argument(
        "--textgrid_dir",
        type=str,
        default="data_original",
        help="Directory containing TextGrid files for empty intervals (default: data_original)"
    )
    parser.add_argument(
        "--populate_empty",
        action="store_true",
        help="Also populate with empty intervals from TextGrid files"
    )
    
    return parser.parse_args()


def load_hf_dataset(dataset_path):
    """Load HuggingFace dataset and extract train/test file mappings."""
    if not HF_DATASETS_AVAILABLE:
        raise ImportError("HuggingFace datasets not available")
    
    print(f"Loading HuggingFace dataset from: {dataset_path}")
    dataset = load_from_disk(dataset_path)
    
    print(f"Dataset splits: {list(dataset.keys())}")
    
    # Extract file mappings from train and test splits
    train_files = set()
    test_files = set()
    
    print(f"Train split size: {len(dataset['train'])}")
    print(f"Test split size: {len(dataset['test'])}")
    
    # Process train split
    print("Processing train split...")
    for example in dataset['train']:
        # Extract filename from path
        path = example['path']
        # Path format: 'data/ESLOLangAgeCombined16kHz/FILENAME.TextGrid'
        filename = Path(path).stem  # Remove .TextGrid extension
        train_files.add(filename)
    
    # Process test split
    print("Processing test split...")
    for example in dataset['test']:
        # Extract filename from path
        path = example['path']
        filename = Path(path).stem
        test_files.add(filename)
    
    print(f"Unique files in train: {len(train_files)}")
    print(f"Unique files in test: {len(test_files)}")
    
    # Check for overlap
    overlap = train_files.intersection(test_files)
    if overlap:
        print(f"Warning: {len(overlap)} files appear in both train and test splits")
        print("Examples:", list(overlap)[:10])
    
    return train_files, test_files


def add_dataset_columns(df):
    """Add ESLO and LangAge columns based on filename patterns."""
    print("Adding dataset type columns...")
    
    # Create ESLO column (1 if filename starts with ESLO, 0 otherwise)
    df['ESLO'] = df['filename'].str.startswith('ESLO').astype(int)
    
    # Create LangAge column (0 if filename starts with ESLO, 1 otherwise)
    df['LangAge'] = (~df['filename'].str.startswith('ESLO')).astype(int)
    
    eslo_count = df['ESLO'].sum()
    langage_count = df['LangAge'].sum()
    
    print(f"Added dataset columns:")
    print(f"  ESLO files: {eslo_count} rows")
    print(f"  LangAge files: {langage_count} rows")
    
    return df


def update_train_test_splits(df, train_files, test_files):
    """Update train and test columns based on HuggingFace dataset splits."""
    print("Updating train/test splits...")
    
    # Initialize columns
    df['train'] = 0
    df['test'] = 0
    
    # Set train=1 for files in train split
    train_mask = df['filename'].isin(train_files)
    df.loc[train_mask, 'train'] = 1
    
    # Set test=1 for files in test split
    test_mask = df['filename'].isin(test_files)
    df.loc[test_mask, 'test'] = 1
    
    # Report statistics
    train_rows = df['train'].sum()
    test_rows = df['test'].sum()
    neither_rows = len(df) - train_rows - test_rows
    overlap_rows = ((df['train'] == 1) & (df['test'] == 1)).sum()
    
    print(f"Updated train/test splits:")
    print(f"  Train rows: {train_rows}")
    print(f"  Test rows: {test_rows}")
    print(f"  Neither rows: {neither_rows}")
    print(f"  Overlap rows: {overlap_rows}")
    
    return df


def find_textgrid_files(textgrid_dir):
    """Find all TextGrid files in the directory."""
    textgrid_files = {}
    for file_path in Path(textgrid_dir).rglob("*.TextGrid"):
        filename = file_path.stem  # Remove .TextGrid extension
        textgrid_files[filename] = str(file_path)
    return textgrid_files


def get_all_intervals_from_textgrid(textgrid_path):
    """Extract ALL intervals (including empty ones) from a TextGrid file."""
    try:
        tg = TextGrid.load_textgrid(textgrid_path)
        if not tg:
            return []
        
        all_intervals = []
        for item in tg.items:
            speaker = item['name']
            for idx, interval in enumerate(item['intervals'], 1):
                # Include ALL intervals, both empty and non-empty
                interval_data = {
                    'speaker': speaker,
                    'start_time': interval['xmin'],
                    'end_time': interval['xmax'],
                    'duration': interval['xmax'] - interval['xmin'],
                    'text': interval['text'],
                    'interval': idx,
                    'is_empty': not interval['text'].strip()
                }
                all_intervals.append(interval_data)
        
        return all_intervals
    except Exception as e:
        print(f"Error processing {textgrid_path}: {e}")
        return []


def process_file_for_empty_intervals(args):
    """Process a single file for empty intervals (for parallel processing)."""
    filename, existing_data, textgrid_files, speaker_mapping = args
    
    if filename not in textgrid_files:
        return []
    
    # Get all intervals from TextGrid
    all_intervals = get_all_intervals_from_textgrid(textgrid_files[filename])
    
    if not all_intervals:
        return []
    
    new_rows = []
    
    # Group intervals by speaker
    intervals_by_speaker = defaultdict(list)
    for interval in all_intervals:
        intervals_by_speaker[interval['speaker']].append(interval)
    
    # For each speaker, create complete sequence
    for speaker, intervals in intervals_by_speaker.items():
        # Sort intervals by interval number
        intervals.sort(key=lambda x: x['interval'])
        
        for interval_data in intervals:
            interval_num = interval_data['interval']
            
            # Check if this interval already exists in CSV
            if interval_num in existing_data.get(filename, {}).get(speaker, {}):
                # Use existing row
                existing_row = existing_data[filename][speaker][interval_num]
                new_rows.append(existing_row)
            else:
                # Create new row for empty interval
                if existing_data.get(filename, {}).get(speaker, {}):
                    # Use first existing row as template
                    template_row = list(existing_data[filename][speaker].values())[0]
                    new_row = template_row.copy()
                else:
                    # Create minimal row structure
                    new_row = pd.Series(dtype=object, index=[
                        'filename', 'speaker_id', 'interview_number', 'startTime', 
                        'endTime', 'interval', 'gender', 'dialect', 'segment_duration',
                        'train', 'test', 'transcript_original', 'transcript_large_v3', 
                        'transcript_fine_tuned'
                    ])
                    new_row['filename'] = filename
                    new_row['speaker_id'] = speaker
                    new_row['interview_number'] = filename
                    
                    # Try to get speaker info from mapping
                    mapping_key = f"{filename}_{speaker}"
                    if mapping_key in speaker_mapping:
                        speaker_info = speaker_mapping[mapping_key]
                        new_row['gender'] = speaker_info.get('gender', '')
                        new_row['dialect'] = speaker_info.get('dialect', '')
                    else:
                        new_row['gender'] = ''
                        new_row['dialect'] = ''
                
                # Update with interval-specific data
                new_row['startTime'] = interval_data['start_time']
                new_row['endTime'] = interval_data['end_time']
                new_row['interval'] = interval_num
                new_row['segment_duration'] = interval_data['duration']
                
                # Clear transcript columns for empty intervals
                transcript_columns = ['transcript_original', 'transcript_large_v3', 'transcript_fine_tuned']
                for col in transcript_columns:
                    if col in new_row.index:
                        new_row[col] = ""
                
                # Clear metric columns
                metric_columns = [col for col in new_row.index if any(metric in str(col).lower() 
                                 for metric in ['wer', 'cer', 'bleu'])]
                for col in metric_columns:
                    new_row[col] = np.nan
                
                new_rows.append(new_row)
    
    return new_rows


def populate_with_empty_intervals(df, textgrid_dir, workers=1):
    """Populate CSV with empty intervals from TextGrid files."""
    print(f"Populating with empty intervals using {workers} workers...")
    
    # Find TextGrid files
    textgrid_files = find_textgrid_files(textgrid_dir)
    print(f"Found {len(textgrid_files)} TextGrid files")
    
    # Create speaker mapping for metadata
    try:
        speaker_mapping = create_speaker_mapping(textgrid_dir)
        print(f"Created speaker mapping with {len(speaker_mapping)} entries")
    except Exception as e:
        print(f"Warning: Could not create speaker mapping: {e}")
        speaker_mapping = {}
    
    # Group existing data by filename and speaker
    existing_data = defaultdict(lambda: defaultdict(dict))
    for _, row in df.iterrows():
        filename = row['filename']
        speaker = row['speaker_id'] 
        interval = int(row['interval'])
        existing_data[filename][speaker][interval] = row
    
    # Prepare arguments for parallel processing
    process_args = []
    for filename in existing_data.keys():
        process_args.append((filename, existing_data, textgrid_files, speaker_mapping))
    
    # Process files in parallel
    all_new_rows = []
    if workers > 1:
        with Pool(workers) as pool:
            results = pool.map(process_file_for_empty_intervals, process_args)
        
        for result in results:
            all_new_rows.extend(result)
    else:
        for args in process_args:
            result = process_file_for_empty_intervals(args)
            all_new_rows.extend(result)
    
    # Create new DataFrame
    if all_new_rows:
        result_df = pd.DataFrame(all_new_rows)
        
        # Sort by filename, speaker, and interval
        result_df = result_df.sort_values(['filename', 'speaker_id', 'interval']).reset_index(drop=True)
        
        print(f"Added {len(result_df) - len(df)} empty intervals")
        return result_df
    else:
        print("No new intervals added")
        return df


def main():
    """Main execution function."""
    args = parse_arguments()
    
    # Validate input files
    if not os.path.exists(args.input_csv):
        print(f"Error: Input CSV file does not exist: {args.input_csv}")
        sys.exit(1)
    
    if not os.path.exists(args.input_dataset):
        print(f"Error: Input dataset directory does not exist: {args.input_dataset}")
        sys.exit(1)
    
    # Create output directory if needed
    output_dir = os.path.dirname(args.output_csv)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")
    
    try:
        # Load input CSV
        print(f"Loading CSV file: {args.input_csv}")
        df = pd.read_csv(args.input_csv, low_memory=False)
        print(f"Loaded CSV with {len(df)} rows")
        
        # Load HuggingFace dataset and get train/test splits
        train_files, test_files = load_hf_dataset(args.input_dataset)
        
        # Add dataset type columns
        df = add_dataset_columns(df)
        
        # Update train/test splits
        df = update_train_test_splits(df, train_files, test_files)
        
        # Populate with empty intervals if requested
        if args.populate_empty:
            if not os.path.exists(args.textgrid_dir):
                print(f"Warning: TextGrid directory does not exist: {args.textgrid_dir}")
                print("Skipping empty interval population")
            else:
                df = populate_with_empty_intervals(df, args.textgrid_dir, args.workers)
        
        # Save result
        print(f"Saving updated CSV to: {args.output_csv}")
        df.to_csv(args.output_csv, index=False)
        
        # Print final statistics
        print(f"\nFinal statistics:")
        print(f"  Total rows: {len(df)}")
        print(f"  ESLO rows: {df['ESLO'].sum()}")
        print(f"  LangAge rows: {df['LangAge'].sum()}")
        print(f"  Train rows: {df['train'].sum()}")
        print(f"  Test rows: {df['test'].sum()}")
        print(f"  Neither train nor test: {((df['train'] == 0) & (df['test'] == 0)).sum()}")
        
        print("Script completed successfully!")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()