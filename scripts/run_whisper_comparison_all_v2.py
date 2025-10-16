#############################################################################
# Script Name: run_whisper_comparison_all.py                               #
# Description: Compare 3 Whisper models: V3 default, fine-tuned, new model #
# Author: Hanno Müller                                                      #
# Date: 2025-10-14                                                          #
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
from transcription import transcribe_with_multiple_models
from metrics_calculation_all import calculate_metrics_for_segments_three_models, print_metrics_summary_three_models


def load_dataset_splits(dataset_path):
    """
    Load dataset and create a lookup dictionary for train/test membership.
    
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
        description="Compare 3 Whisper models: Large V3 default, fine-tuned, and new model"
    )
    
    # Required arguments
    parser.add_argument(
        "--input", "-i",
        type=str,
        required=True,
        help="Path to input directory (e.g., data/LangAgeESLOcombined16kHz)"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        required=True,
        help="Path to output directory (e.g., results/comparison_all)"
    )
    parser.add_argument(
        "--fine_tuned_model",
        type=str,
        required=True,
        help="Path to fine-tuned Whisper model directory"
    )
    parser.add_argument(
        "--new_model",
        type=str,
        required=True,
        help="Path to new Whisper model directory (your newly trained version)"
    )
    parser.add_argument(
        "--checkpoint_fine_tuned",
        type=str,
        help="Specific checkpoint to use for fine-tuned model (e.g., 'checkpoint-2000')"
    )
    parser.add_argument(
        "--checkpoint_new",
        type=str,
        help="Specific checkpoint to use for new model (e.g., 'checkpoint-6000')"
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        help="Path to the HuggingFace dataset (optional). If provided, adds 'train' and 'test' columns to output CSV."
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
        default=1,
        help="Number of GPUs to use (default: 1)"
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


def create_readme(output_dir, args, start_time):
    """Create README.md with execution details."""
    readme_path = output_dir / "README.md"
    
    from datetime import datetime
    
    content = f"""# Whisper Model Comparison Results (3 Models)

## Execution Details

**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  
**Script:** run_whisper_comparison_all.py  
**Duration:** Started at {datetime.fromtimestamp(start_time).strftime('%Y-%m-%d %H:%M:%S')}  

## Command Line Arguments

```bash
python scripts/run_whisper_comparison_all.py \\
  --input "{args.input}" \\
  --output "{args.output}" \\
  --fine_tuned_model "{args.fine_tuned_model}" \\
  --new_model "{args.new_model}" \\"""

    if args.checkpoint_fine_tuned:
        content += f"""
  --checkpoint_fine_tuned "{args.checkpoint_fine_tuned}" \\"""
    
    if args.checkpoint_new:
        content += f"""
  --checkpoint_new "{args.checkpoint_new}" \\"""
    
    content += f"""
  --cpus {args.cpus} \\
  --gpus {args.gpus} \\
  --batch_size {args.batch_size} \\
  --transcription_batch_processes {args.transcription_batch_processes} \\
  --steps "{args.steps}"\\"""

    if args.file_limit:
        content += f"""
  --file_limit {args.file_limit} \\"""
    
    if args.resume_from_transcriptions:
        content += f"""
  --resume_from_transcriptions "{args.resume_from_transcriptions}" \\"""
    
    content = content.rstrip(" \\")
    
    content += f"""
```

## Configuration Summary

- **Input Directory:** `{args.input}`
- **Output Directory:** `{args.output}`
- **Whisper Large V3:** Default model from OpenAI
- **Fine-tuned Model:** `{args.fine_tuned_model}`
  - Checkpoint: `{args.checkpoint_fine_tuned if args.checkpoint_fine_tuned else 'final model'}`
- **New Model:** `{args.new_model}`
  - Checkpoint: `{args.checkpoint_new if args.checkpoint_new else 'final model'}`
- **Processing:** {args.cpus} CPUs, {args.gpus} GPUs{' (Multi-GPU Support)' if args.gpus > 1 else ''}
- **Batch Size:** {args.batch_size}
- **Transcription Processes:** {args.transcription_batch_processes}
- **Pipeline Steps:** {args.steps}
{f"- **File Limit:** {args.file_limit}" if args.file_limit else ""}
{f"- **Resume From:** `{args.resume_from_transcriptions}`" if args.resume_from_transcriptions else ""}

## Models Compared

This comparison evaluates three Whisper models:

1. **Whisper Large V3 (Default)**: OpenAI's base model without fine-tuning
2. **Fine-tuned Model**: Previously trained model with specific checkpoint
3. **New Model**: Your newly trained Whisper version

## Multi-GPU Support

{f'This pipeline includes multi-GPU parallel processing support. With {args.gpus} GPUs configured, the workload is automatically distributed across all available GPUs for faster processing.' if args.gpus > 1 else 'Single-GPU processing mode.'}

## Output Files

- **Main Results:** `whisper_comparison_all_results.csv`
- **Sample Results:** `whisper_comparison_all_results_sample.csv` (first 100 rows)
- **Intermediate Files:** `whisper_comparison_all_results_intermediate/`
  - `segments_with_metadata.json`
  - `segments_with_transcriptions.json`

## Pipeline Steps

1. **Metadata Extraction:** Extract audio segments and speaker information
2. **Transcription:** Transcribe with all three Whisper models
3. **Metrics Calculation:** Calculate WER, CER, and BLEU scores
4. **Results Export:** Create comprehensive CSV with all comparisons

## Metrics Included

- **WER (Word Error Rate):** Lower is better
- **CER (Character Error Rate):** Lower is better  
- **BLEU Score:** Higher is better

Comparisons are made between:
- Large V3 vs Original transcripts
- Fine-tuned vs Original transcripts
- New Model vs Original transcripts
- Large V3 vs Fine-tuned transcripts
- Large V3 vs New Model transcripts
- Fine-tuned vs New Model transcripts
"""
    
    print(f"Creating README.md: {readme_path}")
    with open(readme_path, 'w', encoding='utf-8') as f:
        f.write(content)


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
    """Run transcription step with all three models."""
    print("\n" + "="*60)
    print("STEP 2: TRANSCRIPTION (3 MODELS)")
    print("="*60)
    
    # Determine model paths (with checkpoints if specified)
    if args.checkpoint_fine_tuned:
        fine_tuned_model_path = os.path.join(args.fine_tuned_model, args.checkpoint_fine_tuned)
        if not os.path.exists(fine_tuned_model_path):
            print(f"Fine-tuned checkpoint not found: {fine_tuned_model_path}")
            sys.exit(1)
        print(f"Fine-tuned model using checkpoint: {args.checkpoint_fine_tuned}")
    else:
        fine_tuned_model_path = args.fine_tuned_model
        print(f"Fine-tuned model using final model (no checkpoint specified)")
    
    if args.checkpoint_new:
        new_model_path = os.path.join(args.new_model, args.checkpoint_new)
        if not os.path.exists(new_model_path):
            print(f"New model checkpoint not found: {new_model_path}")
            sys.exit(1)
        print(f"New model using checkpoint: {args.checkpoint_new}")
    else:
        new_model_path = args.new_model
        print(f"New model using final model (no checkpoint specified)")
    
    # Separate empty and non-empty segments
    non_empty_segments = []
    empty_segments_indices = []
    
    for i, segment in enumerate(segments):
        original_text = segment.get('text', '').strip()
        if original_text:  # Non-empty
            non_empty_segments.append(segment)
        else:  # Empty interval
            empty_segments_indices.append(i)
            # Mark as empty for later processing
            segment['is_empty_interval'] = True
    
    print(f"Total segments: {len(segments)}")
    print(f"  - Non-empty segments to transcribe: {len(non_empty_segments)}")
    print(f"  - Empty intervals (will be skipped): {len(empty_segments_indices)}")
    
    # Transcribe only non-empty segments with all three models
    if non_empty_segments:
        model_paths = ["openai/whisper-large-v3", fine_tuned_model_path, new_model_path]
        large_v3_results, fine_tuned_results, new_model_results = transcribe_with_multiple_models(
            non_empty_segments,
            model_paths,
            batch_size=args.batch_size,
            num_workers=args.transcription_batch_processes,
            progress_callback=progress_callback,
            num_gpus=args.gpus
        )
    else:
        large_v3_results, fine_tuned_results, new_model_results = [], [], []
    
    # Add transcriptions to segments
    print("Combining transcription results...")
    non_empty_idx = 0
    for i, segment in enumerate(segments):
        segment['transcript_original'] = segment.get('text', '')
        
        if segment.get('is_empty_interval', False):
            # Empty interval: set all transcriptions to empty
            segment['transcript_large_v3'] = ''
            segment['transcript_fine_tuned'] = ''
            segment['transcript_new_model'] = ''
        else:
            # Non-empty: use transcription results
            segment['transcript_large_v3'] = large_v3_results[non_empty_idx]['text']
            segment['transcript_fine_tuned'] = fine_tuned_results[non_empty_idx]['text']
            segment['transcript_new_model'] = new_model_results[non_empty_idx]['text']
            non_empty_idx += 1
    
    return segments


def run_metrics_calculation(segments, args):
    """Run metrics calculation step for all three models."""
    print("\n" + "="*60)
    print("STEP 3: METRICS CALCULATION (3 MODELS)")
    print("="*60)
    
    # Separate empty and non-empty segments for metric calculation
    empty_intervals_count = sum(1 for seg in segments if seg.get('is_empty_interval', False))
    non_empty_count = len(segments) - empty_intervals_count
    
    print(f"Total segments for metrics: {len(segments)}")
    print(f"  - Non-empty segments: {non_empty_count}")
    print(f"  - Empty intervals (will get default metrics): {empty_intervals_count}")
    
    # Calculate metrics for non-empty segments
    # Empty segments will get default metrics assigned by the calculation function
    segments_with_metrics = calculate_metrics_for_segments_three_models(segments)
    
    # Print summary
    print_metrics_summary_three_models(segments_with_metrics)
    
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
    
    special_markers = ['XXX']
    
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
        'LangAge',
        'ESLO',
        'transcript_original',
        'transcript_large_v3',
        'transcript_fine_tuned',
        'transcript_new_model',
        # WER metrics (6 comparisons)
        'WER_large_v3_vs_original',
        'WER_fine_tuned_vs_original',
        'WER_new_model_vs_original',
        'WER_large_v3_vs_fine_tuned',
        'WER_large_v3_vs_new_model',
        'WER_fine_tuned_vs_new_model',
        # CER metrics (6 comparisons)
        'CER_large_v3_vs_original',
        'CER_fine_tuned_vs_original',
        'CER_new_model_vs_original',
        'CER_large_v3_vs_fine_tuned',
        'CER_large_v3_vs_new_model',
        'CER_fine_tuned_vs_new_model',
        # BLEU metrics (6 comparisons)
        'BLEU_large_v3_vs_original',
        'BLEU_fine_tuned_vs_original',
        'BLEU_new_model_vs_original',
        'BLEU_large_v3_vs_fine_tuned',
        'BLEU_large_v3_vs_new_model',
        'BLEU_fine_tuned_vs_new_model'
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
                        row[col] = 0
                else:
                    row[col] = 0
            elif col == 'LangAge' or col == 'ESLO':
                # Determine corpus membership based on filename
                filename = segment.get('filename', '')
                if col == 'ESLO':
                    # ESLO files start with 'ESLO' prefix
                    row[col] = 1 if filename.startswith('ESLO') else 0
                else:  # LangAge
                    # LangAge files are those that don't start with 'ESLO'
                    row[col] = 0 if filename.startswith('ESLO') else 1
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
        
        # Check interjection markers
        for marker in interjection_markers:
            found_in_markers = marker.lower() in markers_str.lower()
            
            import re
            found_in_text = bool(re.search(r'\b' + re.escape(marker) + r'\b', transcript_original, re.IGNORECASE))
            
            if found_in_markers or found_in_text:
                row[marker] = 1
        
        # Check parenthetical markers
        for marker in parenthetical_markers:
            if marker in markers_str:
                row[marker] = 1
        
        # Check for XXX pattern
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
    
    print("WHISPER MODEL COMPARISON PIPELINE (3 MODELS)")
    print("="*60)
    print(f"Input directory: {args.input}")
    print(f"Output directory: {args.output}")
    print(f"Fine-tuned model: {args.fine_tuned_model}")
    if args.checkpoint_fine_tuned:
        print(f"  └─ Checkpoint: {args.checkpoint_fine_tuned}")
    print(f"New model: {args.new_model}")
    if args.checkpoint_new:
        print(f"  └─ Checkpoint: {args.checkpoint_new}")
    print(f"Configuration: {args.cpus} CPUs, {args.gpus} GPUs, batch size {args.batch_size}")
    print(f"Steps to run: {args.steps}")
    
    # Validate inputs
    if not os.path.exists(args.input):
        print(f"Input directory not found: {args.input}")
        sys.exit(1)
    
    if not os.path.exists(args.fine_tuned_model):
        print(f"Fine-tuned model directory not found: {args.fine_tuned_model}")
        sys.exit(1)
    
    if not os.path.exists(args.new_model):
        print(f"New model directory not found: {args.new_model}")
        sys.exit(1)
    
    # Validate checkpoints if specified
    if args.checkpoint_fine_tuned:
        checkpoint_path = os.path.join(args.fine_tuned_model, args.checkpoint_fine_tuned)
        if not os.path.exists(checkpoint_path):
            print(f"Fine-tuned checkpoint not found: {checkpoint_path}")
            sys.exit(1)
    
    if args.checkpoint_new:
        checkpoint_path = os.path.join(args.new_model, args.checkpoint_new)
        if not os.path.exists(checkpoint_path):
            print(f"New model checkpoint not found: {checkpoint_path}")
            sys.exit(1)
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Define output file paths
    csv_file = output_dir / "whisper_comparison_all_results.csv"
    sample_csv_file = output_dir / "whisper_comparison_all_results_sample.csv"
    
    # Define intermediate file paths
    intermediate_dir = output_dir / "whisper_comparison_all_results_intermediate"
    intermediate_dir.mkdir(exist_ok=True)
    
    segments_file = intermediate_dir / "segments_with_metadata.json"
    transcriptions_file = intermediate_dir / "segments_with_transcriptions.json"
    
    # Create README.md with execution details
    create_readme(output_dir, args, start_time)
    
    # Load dataset for train/test membership (if provided)
    dataset_lookup = None
    if args.dataset_path:
        dataset_lookup = load_dataset_splits(args.dataset_path)
        if dataset_lookup:
            print(f"✓ Dataset loaded successfully for train/test tracking")
        else:
            print(f"✗ Could not load dataset, train/test columns will be 0")
    
    try:
        # Step 1: Metadata extraction
        if args.steps in ["all", "metadata"]:
            segments = run_metadata_extraction(args)
            save_intermediate_results(segments, segments_file)
            
            # If only metadata was requested, exit successfully here
            if args.steps == "metadata":
                print(f"\n" + "="*60)
                print("METADATA EXTRACTION COMPLETED SUCCESSFULLY")
                print("="*60)
                print(f"Total segments extracted: {len(segments)}")
                print(f"Results saved to: {segments_file}")
                print(f"\nTo continue with transcription, run with --steps transcription")
                return
        else:
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
            
            # If only transcription was requested, exit successfully here
            if args.steps == "transcription":
                print(f"\n" + "="*60)
                print("TRANSCRIPTION COMPLETED SUCCESSFULLY")
                print("="*60)
                print(f"Total segments transcribed: {len(segments)}")
                print(f"Results saved to: {transcriptions_file}")
                print(f"\nTo continue with metrics, run with --steps metrics")
                return
        else:
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
        print(f"Documentation: {output_dir / 'README.md'}")
        
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
