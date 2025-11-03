#!/usr/bin/env python3

#############################################################################
# Script Name: run_whisper_comparison.py                                   #
# Description: Consolidated script to compare Whisper Large V3 vs          #
#              fine-tuned model with comprehensive CSV output               #
# Author: Hanno Müller                                                      #
# Date: 2025-10-30                                                          #
#                                                                           #
# Features:                                                                 #
# - Compare 2 models: Whisper Large V3 vs Fine-tuned model                #
# - Multi-GPU parallel processing                                          #
# - Train/test split lookup from HuggingFace dataset                       #
# - Comprehensive CSV output with all metrics and marker columns           #
# - Resume functionality and intermediate file saving                      #
#############################################################################

import os
import sys
import json
import pandas as pd
import argparse
from pathlib import Path
import time

# Add modules to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'modules'))

from speaker_mapping import create_speaker_mapping, get_speaker_info, print_speaker_statistics
from metadata_extraction import extract_segments_from_folder
from transcription import transcribe_with_both_models
from metrics_calculation import calculate_metrics_for_segments, print_metrics_summary


def load_dataset_splits(dataset_path):
    """
    Load dataset and create a lookup dictionary for train/test membership.
    Uses Method 2: extract filename from audio.path, use client_id and segment.
    
    Args:
        dataset_path (str): Path to HuggingFace dataset directory
        
    Returns:
        dict: Lookup dictionary with keys created from audio path, client_id, and segment
              mapped to {'train': 0/1, 'test': 0/1}
    """
    if not dataset_path or not os.path.exists(dataset_path):
        return None
    
    try:
        from datasets import load_from_disk
        
        print(f"\nLoading dataset from {dataset_path}...")
        dataset_dict = load_from_disk(dataset_path)
        
        lookup = {}
        
        # Process train split
        if 'train' in dataset_dict:
            print(f"   Processing train split ({len(dataset_dict['train'])} samples)...")
            for item in dataset_dict['train']:
                # Extract filename from audio path (e.g., 'data/.../ESLO1_ENT_019.wav' -> 'ESLO1_ENT_019')
                audio_path = item.get('audio', {}).get('path', '')
                if audio_path:
                    filename = os.path.splitext(os.path.basename(audio_path))[0]
                else:
                    filename = ''
                
                speaker = item.get('client_id', '')
                interval = item.get('segment', '')
                
                key = f"{filename}_{speaker}_{interval}"
                lookup[key] = {'train': 1, 'test': 0}
        
        # Process test split
        if 'test' in dataset_dict:
            print(f"   Processing test split ({len(dataset_dict['test'])} samples)...")
            for item in dataset_dict['test']:
                # Extract filename from audio path
                audio_path = item.get('audio', {}).get('path', '')
                if audio_path:
                    filename = os.path.splitext(os.path.basename(audio_path))[0]
                else:
                    filename = ''
                
                speaker = item.get('client_id', '')
                interval = item.get('segment', '')
                
                key = f"{filename}_{speaker}_{interval}"
                if key in lookup:
                    lookup[key] = {'train': 1, 'test': 1}
                else:
                    lookup[key] = {'train': 0, 'test': 1}
        
        print(f"   Dataset lookup created: {len(lookup)} entries")
        return lookup
        
    except Exception as e:
        print(f"Warning: Could not load dataset from {dataset_path}: {e}")
        import traceback
        traceback.print_exc()
        return None


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Compare Whisper Large V3 vs fine-tuned model with comprehensive analysis"
    )
    
    # Required arguments
    parser.add_argument(
        "--input", "-i",
        type=str,
        default="data/ESLOLangAgeCombined16kHz",
        help="Path to input audio directory (default: data/ESLOLangAgeCombined16kHz)"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        required=True,
        help="Path to output directory (e.g., results/comparison_output)"
    )
    parser.add_argument(
        "--fine_tuned_model",
        type=str,
        required=True,
        help="Full path to fine-tuned Whisper model including checkpoint (e.g., FrisperWhisper/ESLOLangAgeCombined/checkpoint-4000)"
    )
    
    # Dataset for train/test lookup
    parser.add_argument(
        "--dataset",
        type=str,
        default="data/ESLOLangAgeDataSet",
        help="Path to HuggingFace dataset for train/test lookup (default: data/ESLOLangAgeDataSet)"
    )
    
    # Processing arguments
    parser.add_argument(
        "--cpus",
        type=int,
        default=8,
        help="Number of CPU cores to use (default: 8)"
    )
    parser.add_argument(
        "--gpus",
        type=int,
        default=2,
        help="Number of GPUs to use (default: 2)"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for transcription (default: 32)"
    )
    parser.add_argument(
        "--transcription_batch_processes",
        type=int,
        default=4,
        help="Number of batch processes for transcription (default: 4)"
    )
    
    # Pipeline control
    parser.add_argument(
        "--steps",
        type=str,
        default="all",
        choices=["all", "metadata", "transcription", "metrics"],
        help="Pipeline steps to run (default: all)"
    )
    parser.add_argument(
        "--file_limit",
        type=int,
        help="Limit number of files for testing (optional)"
    )
    parser.add_argument(
        "--resume_from_transcriptions",
        type=str,
        help="Resume from existing transcriptions JSON file (optional)"
    )
    
    return parser.parse_args()


def save_intermediate_results(data, filepath):
    """Save intermediate results to JSON file."""
    print(f"Saving intermediate results to {filepath}")
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_intermediate_results(filepath):
    """Load intermediate results from JSON file."""
    print(f"Loading intermediate results from {filepath}")
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def progress_callback(current, total):
    """Progress callback for transcription."""
    percentage = (current / total) * 100
    print(f"   Progress: {current}/{total} batches ({percentage:.1f}%)")


def run_metadata_extraction(args):
    """Run metadata extraction step."""
    print("\n" + "="*60)
    print("STEP 1: METADATA EXTRACTION")
    print("="*60)
    
    # Create speaker mapping
    speaker_mapping = create_speaker_mapping(args.input)
    print_speaker_statistics(speaker_mapping)
    
    # Extract segments
    segments = extract_segments_from_folder(args.input, args.file_limit)
    
    # Add speaker information to each segment
    print("Adding speaker information to segments...")
    for segment in segments:
        speaker_info = get_speaker_info(
            speaker_mapping, 
            segment['filename'], 
            segment['speaker']
        )
        
        # Preserve existing gender information from TRS parsing
        if 'gender' in segment and segment['gender']:
            speaker_info['gender'] = segment['gender']
        if 'dialect' in segment and segment['dialect']:
            speaker_info['dialect'] = segment['dialect']
        if 'accent' in segment and segment['accent']:
            speaker_info['accent'] = segment['accent']
            
        segment.update(speaker_info)
    
    return segments


def run_transcription(segments, args):
    """Run transcription step."""
    print("\n" + "="*60)
    print("STEP 2: TRANSCRIPTION")
    print("="*60)
    
    # Validate that fine-tuned model path exists
    if not os.path.exists(args.fine_tuned_model):
        print(f"Error: Fine-tuned model not found: {args.fine_tuned_model}")
        sys.exit(1)
    
    print(f"Using fine-tuned model: {args.fine_tuned_model}")
    
    # Transcribe with both models
    large_v3_results, fine_tuned_results = transcribe_with_both_models(
        segments,
        args.fine_tuned_model,
        batch_size=args.batch_size,
        num_workers=args.transcription_batch_processes,
        progress_callback=progress_callback,
        num_gpus=args.gpus
    )
    
    # Add transcriptions to segments
    print("Combining transcription results...")
    for i, segment in enumerate(segments):
        segment['transcript_original'] = segment['text']
        segment['transcript_large_v3'] = large_v3_results[i]['text']
        segment['transcript_fine_tuned'] = fine_tuned_results[i]['text']
    
    return segments


def run_metrics_calculation(segments, args):
    """Run metrics calculation step."""
    print("\n" + "="*60)
    print("STEP 3: METRICS CALCULATION")
    print("="*60)
    
    # Calculate all metrics
    segments_with_metrics = calculate_metrics_for_segments(segments)
    
    # Print summary
    print_metrics_summary(segments_with_metrics)
    
    return segments_with_metrics


def create_final_dataframe(segments, dataset_lookup=None):
    """Create final DataFrame with all results including individual marker columns and train/test indicators.
    
    Args:
        segments (list): List of segment dictionaries with transcriptions and metrics
        dataset_lookup (dict, optional): Lookup dictionary for train/test membership
        
    Returns:
        pd.DataFrame: Final dataframe with all columns
    """
    print("\n" + "="*60)
    print("CREATING FINAL DATAFRAME")
    print("="*60)
    
    # Define marker categories
    interjection_markers = ['ah', 'bah', 'beh', 'ben', 'chh', 'eh', 'euh', 'ha', 'hé', 
                           'hein', 'hop', 'hum', 'm-hm', 'mmh', 'mm', 'oh', 'ouf', 'pff', 'youh']
    
    parenthetical_markers = ['(bru)', '(bou)', '(exp)', '(ges)', '(ins)', '(ono)', '(pau)', 
                            '(rac)', '(rir)', '(tou)', '(tss)', '(sou)', '(buzz)']
    
    special_markers = ['XXX']  # For XX, XXX, XXXX patterns
    
    all_markers = interjection_markers + parenthetical_markers + special_markers
    
    # Define base columns in desired order
    base_columns = [
        'filename',
        'speaker_id', 
        'interview_number',
        'startTime',
        'endTime',
        'interval',
        'gender',
        'dialect',
        'segment_duration',
        'train',
        'test',
        'transcript_original',
        'transcript_large_v3',
        'transcript_fine_tuned',
        'WER_large_v3_vs_original',
        'WER_fine_tuned_vs_original',
        'WER_large_v3_vs_fine_tuned',
        'CER_large_v3_vs_original',
        'CER_fine_tuned_vs_original',
        'CER_large_v3_vs_fine_tuned',
        'BLEU_large_v3_vs_original',
        'BLEU_fine_tuned_vs_original',
        'BLEU_large_v3_vs_fine_tuned'
    ]
    
    # Final column order: base columns + marker columns
    columns = base_columns + all_markers
    
    # Convert to DataFrame
    df_data = []
    for segment in segments:
        row = {}
        
        # Fill base columns
        for col in base_columns:
            if col == 'segment_duration':
                row[col] = segment.get('duration', 0.0)
            elif col == 'startTime':
                row[col] = segment.get('start_time', 0.0)
            elif col == 'endTime':
                row[col] = segment.get('end_time', 0.0)
            elif col == 'interval':
                row[col] = segment.get('interval', 0)
            elif col == 'train' or col == 'test':
                # Determine train/test membership from dataset lookup
                if dataset_lookup is not None:
                    filename = segment.get('filename', '')
                    speaker = segment.get('speaker', '')
                    interval = segment.get('interval', 0)
                    key = f"{filename}_{speaker}_{interval}"
                    
                    if key in dataset_lookup:
                        row[col] = dataset_lookup[key].get(col, 0)
                    else:
                        row[col] = 0  # Not in dataset
                else:
                    row[col] = 0  # No dataset provided
            else:
                row[col] = segment.get(col, '')
        
        # Initialize all marker columns to 0
        for marker in all_markers:
            row[marker] = 0
        
        # Fill marker columns based on extracted markers
        markers_list = segment.get('markers', [])
        transcript_original = segment.get('text', '')
        
        # Convert markers list to string for pattern matching
        markers_str = '; '.join(markers_list) if markers_list else ''
        
        # Check interjection markers (look in both markers and original text)
        for marker in interjection_markers:
            # Check in extracted markers (case-insensitive)
            found_in_markers = marker.lower() in markers_str.lower()
            
            # Also check in original text as word boundaries (case-insensitive)
            import re
            found_in_text = bool(re.search(r'\b' + re.escape(marker) + r'\b', transcript_original, re.IGNORECASE))
            
            if found_in_markers or found_in_text:
                row[marker] = 1
        
        # Check parenthetical markers (only in extracted markers)
        for marker in parenthetical_markers:
            if marker in markers_str:
                row[marker] = 1
        
        # Check for XXX pattern (XX or more X's in original transcript)
        if 'XX' in transcript_original:
            row['XXX'] = 1
        
        df_data.append(row)
    
    df = pd.DataFrame(df_data, columns=columns)
    
    print(f"DataFrame created:")
    print(f"   Rows: {len(df)}")
    print(f"   Columns: {len(df.columns)}")
    print(f"   Marker columns added: {len(all_markers)}")
    
    return df


def main():
    """Main pipeline execution."""
    start_time = time.time()
    
    # Parse arguments
    args = parse_arguments()
    
    print("WHISPER MODEL COMPARISON PIPELINE (CONSOLIDATED)")
    print("="*60)
    print(f"Input directory: {args.input}")
    print(f"Output directory: {args.output}")
    print(f"Fine-tuned model: {args.fine_tuned_model}")
    print(f"Dataset path: {args.dataset}")
    print(f"Configuration: {args.cpus} CPUs, {args.gpus} GPUs, batch size {args.batch_size}")
    print(f"Steps to run: {args.steps}")
    
    # Validate inputs
    if not os.path.exists(args.input):
        print(f"Input directory not found: {args.input}")
        sys.exit(1)
    
    if not os.path.exists(args.fine_tuned_model):
        print(f"Fine-tuned model directory not found: {args.fine_tuned_model}")
        sys.exit(1)
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Define output file paths
    csv_file = output_dir / "whisper_comparison_results.csv"
    sample_csv_file = output_dir / "whisper_comparison_results_sample.csv"
    
    # Define intermediate file paths
    intermediate_dir = output_dir / "whisper_comparison_results_intermediate"
    intermediate_dir.mkdir(exist_ok=True)
    
    segments_file = intermediate_dir / "segments_with_metadata.json"
    transcriptions_file = intermediate_dir / "segments_with_transcriptions.json"
    
    # Load dataset for train/test membership
    dataset_lookup = load_dataset_splits(args.dataset)
    if dataset_lookup:
        print(f"✓ Dataset loaded successfully for train/test tracking")
    else:
        print(f"✗ Could not load dataset, train/test columns will be 0")
    
    try:
        # Step 1: Metadata extraction
        if args.steps in ["all", "metadata"]:
            segments = run_metadata_extraction(args)
            save_intermediate_results(segments, segments_file)
        else:
            # Load existing metadata
            if segments_file.exists():
                segments = load_intermediate_results(segments_file)
            else:
                print("No existing metadata found. Run metadata step first.")
                sys.exit(1)
        
        # Step 2: Transcription
        if args.steps in ["all", "transcription"]:
            if args.resume_from_transcriptions:
                segments = load_intermediate_results(args.resume_from_transcriptions)
            else:
                segments = run_transcription(segments, args)
                save_intermediate_results(segments, transcriptions_file)
        else:
            # Load existing transcriptions
            if transcriptions_file.exists():
                segments = load_intermediate_results(transcriptions_file)
            elif args.resume_from_transcriptions:
                segments = load_intermediate_results(args.resume_from_transcriptions)
            else:
                print("No existing transcriptions found. Run transcription step first.")
                sys.exit(1)
        
        # Step 3: Metrics calculation
        if args.steps in ["all", "metrics"]:
            segments = run_metrics_calculation(segments, args)
        
        # Create final DataFrame and save
        df = None
        if args.steps in ["all", "metrics"]:
            df = create_final_dataframe(segments, dataset_lookup)
            
            print(f"\nSaving final results to {csv_file}")
            df.to_csv(csv_file, index=False, encoding='utf-8')
            
            # Save a sample for quick inspection
            df.head(100).to_csv(sample_csv_file, index=False, encoding='utf-8')
            print(f"Sample (100 rows) saved to {sample_csv_file}")
        
        # Final statistics
        end_time = time.time()
        duration = end_time - start_time
        
        print(f"\n" + "="*60)
        print("PIPELINE COMPLETED SUCCESSFULLY")
        print("="*60)
        print(f"Total time: {duration/60:.1f} minutes")
        if df is not None:
            print(f"Total segments processed: {len(df)}")
            print(f"Main results: {csv_file}")
            print(f"Sample results: {sample_csv_file}")
        print(f"Intermediate files: {intermediate_dir}")
        
    except KeyboardInterrupt:
        print(f"\nPipeline interrupted by user")
        print(f"Intermediate files saved in: {intermediate_dir}")
        sys.exit(1)
    except Exception as e:
        print(f"\nPipeline failed with error: {e}")
        import traceback
        traceback.print_exc()
        print(f"Check intermediate files in: {intermediate_dir}")
        sys.exit(1)


if __name__ == "__main__":
    main()
