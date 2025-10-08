#############################################################################
# Script Name: run_whisper_comparison_pipeline.py                          #
# Description: Pipeline for comparing Whisper Large V3 vs fine-tuned model #
# Author: Hanno Müller                                                      #
# Date: 2025-09-03                                                          #
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


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Compare Whisper Large V3 vs fine-tuned model on LangAge dataset"
    )
    
    # Required arguments
    parser.add_argument(
        "--input", "-i",
        type=str,
        required=True,
        help="Path to input directory (data/LangAge16kHz)"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        required=True,
        help="Path to output directory (e.g., results/largeV3.1)"
    )
    parser.add_argument(
        "--fine_tuned_model",
        type=str,
        required=True,
        help="Path to fine-tuned Whisper model directory"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        help="Specific checkpoint to use (e.g., 'checkpoint-6000'). If not specified, uses the final model."
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
    
    content = f"""# Whisper Model Comparison Results

## Execution Details

**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  
**Script:** run_whisper_comparison_pipeline.py  
**Duration:** Started at {datetime.fromtimestamp(start_time).strftime('%Y-%m-%d %H:%M:%S')}  

## Command Line Arguments

```bash
python scripts/run_whisper_comparison_pipeline.py \\
  --input "{args.input}" \\
  --output "{args.output}" \\
  --fine_tuned_model "{args.fine_tuned_model}" \\"""

    if args.checkpoint:
        content += f"""
  --checkpoint "{args.checkpoint}" \\"""
    
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
    
    content = content.rstrip(" \\")  # Remove trailing backslash
    
    content += f"""
```

## Configuration Summary

- **Input Directory:** `{args.input}`
- **Output Directory:** `{args.output}`
- **Fine-tuned Model:** `{args.fine_tuned_model}`
- **Checkpoint:** `{args.checkpoint if args.checkpoint else 'final model'}`
- **Processing:** {args.cpus} CPUs, {args.gpus} GPUs{' (Multi-GPU Support)' if args.gpus > 1 else ''}
- **Batch Size:** {args.batch_size}
- **Transcription Processes:** {args.transcription_batch_processes}
- **Pipeline Steps:** {args.steps}
{f"- **File Limit:** {args.file_limit}" if args.file_limit else ""}
{f"- **Resume From:** `{args.resume_from_transcriptions}`" if args.resume_from_transcriptions else ""}

## Multi-GPU Support

{f'This pipeline includes multi-GPU parallel processing support. With {args.gpus} GPUs configured, the workload is automatically distributed across all available GPUs for faster processing.' if args.gpus > 1 else 'Single-GPU processing mode.'}

## Output Files

- **Main Results:** `whisper_comparison_results.csv`
- **Sample Results:** `whisper_comparison_results_sample.csv` (first 100 rows)
- **Intermediate Files:** `whisper_comparison_results_intermediate/`
  - `segments_with_metadata.json`
  - `segments_with_transcriptions.json`

## Pipeline Steps

1. **Metadata Extraction:** Extract audio segments and speaker information
2. **Transcription:** Transcribe with both Whisper Large V3 and fine-tuned model
3. **Metrics Calculation:** Calculate WER, CER, and BLEU scores
4. **Results Export:** Create comprehensive CSV with all comparisons

## Metrics Included

- **WER (Word Error Rate):** Lower is better
- **CER (Character Error Rate):** Lower is better  
- **BLEU Score:** Higher is better

Comparisons are made between:
- Large V3 vs Original transcripts
- Fine-tuned vs Original transcripts
- Large V3 vs Fine-tuned transcripts
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
    """Run transcription step."""
    print("\n" + "="*60)
    print("STEP 2: TRANSCRIPTION")
    print("="*60)
    
    # Determine model path (with checkpoint if specified)
    if args.checkpoint:
        fine_tuned_model_path = os.path.join(args.fine_tuned_model, args.checkpoint)
        if not os.path.exists(fine_tuned_model_path):
            print(f"Checkpoint not found: {fine_tuned_model_path}")
            sys.exit(1)
        print(f"Using checkpoint: {args.checkpoint}")
    else:
        fine_tuned_model_path = args.fine_tuned_model
        print(f"Using final model (no checkpoint specified)")
    
    # Transcribe with both models
    large_v3_results, fine_tuned_results = transcribe_with_both_models(
        segments,
        fine_tuned_model_path,
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


def create_final_dataframe(segments):
    """Create final DataFrame with all results including individual marker columns."""
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
    
    print("WHISPER MODEL COMPARISON PIPELINE")
    print("="*60)
    print(f"Input directory: {args.input}")
    print(f"Output directory: {args.output}")
    print(f"Fine-tuned model: {args.fine_tuned_model}")
    if args.checkpoint:
        print(f"Checkpoint: {args.checkpoint}")
    else:
        print(f"Checkpoint: final model")
    print(f"Configuration: {args.cpus} CPUs, {args.gpus} GPUs, batch size {args.batch_size}")
    print(f"Steps to run: {args.steps}")
    
    # Validate inputs
    if not os.path.exists(args.input):
        print(f"Input directory not found: {args.input}")
        sys.exit(1)
    
    if not os.path.exists(args.fine_tuned_model):
        print(f"Fine-tuned model directory not found: {args.fine_tuned_model}")
        sys.exit(1)
    
    # Validate checkpoint if specified
    if args.checkpoint:
        checkpoint_path = os.path.join(args.fine_tuned_model, args.checkpoint)
        if not os.path.exists(checkpoint_path):
            print(f"Checkpoint not found: {checkpoint_path}")
            # List available checkpoints
            try:
                checkpoints = [d for d in os.listdir(args.fine_tuned_model) 
                             if d.startswith('checkpoint-') and os.path.isdir(os.path.join(args.fine_tuned_model, d))]
                if checkpoints:
                    print(f"Available checkpoints: {sorted(checkpoints)}")
                else:
                    print("No checkpoints found in model directory")
            except:
                pass
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
    
    # Create README.md with execution details
    create_readme(output_dir, args, start_time)
    
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
            df = create_final_dataframe(segments)
            
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
