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
        help="Path to output CSV file"
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
        progress_callback=progress_callback
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
    """Create final DataFrame with all results."""
    print("\n" + "="*60)
    print("CREATING FINAL DATAFRAME")
    print("="*60)
    
    # Define columns in desired order
    columns = [
        'filename',
        'segment_id',
        'start_time',
        'end_time',
        'speaker_id', 
        'interview_number',
        'gender',
        'dialect',
        'markers',
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
    
    # Convert to DataFrame
    df_data = []
    for segment in segments:
        row = {}
        for col in columns:
            if col == 'segment_duration':
                row[col] = segment.get('duration', 0.0)
            elif col == 'start_time':
                row[col] = segment.get('start_time', 0.0)
            elif col == 'end_time':
                row[col] = segment.get('end_time', 0.0)
            elif col == 'markers':
                # Convert markers list to string
                markers = segment.get('markers', [])
                row[col] = '; '.join(markers) if markers else ''
            else:
                row[col] = segment.get(col, '')
        df_data.append(row)
    
    df = pd.DataFrame(df_data, columns=columns)
    
    print(f"DataFrame created:")
    print(f"   Rows: {len(df)}")
    print(f"   Columns: {len(df.columns)}")
    
    return df


def main():
    """Main pipeline execution."""
    start_time = time.time()
    
    # Parse arguments
    args = parse_arguments()
    
    print("WHISPER MODEL COMPARISON PIPELINE")
    print("="*60)
    print(f"Input directory: {args.input}")
    print(f"Output file: {args.output}")
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
    output_dir = Path(args.output).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Define intermediate file paths
    base_name = Path(args.output).stem
    intermediate_dir = output_dir / f"{base_name}_intermediate"
    intermediate_dir.mkdir(exist_ok=True)
    
    segments_file = intermediate_dir / "segments_with_metadata.json"
    transcriptions_file = intermediate_dir / "segments_with_transcriptions.json"
    
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
            
            print(f"\nSaving final results to {args.output}")
            df.to_csv(args.output, index=False, encoding='utf-8')
            
            # Save a sample for quick inspection
            sample_file = output_dir / f"{base_name}_sample.csv"
            df.head(100).to_csv(sample_file, index=False, encoding='utf-8')
            print(f"Sample (100 rows) saved to {sample_file}")
        
        # Final statistics
        end_time = time.time()
        duration = end_time - start_time
        
        print(f"\n" + "="*60)
        print("PIPELINE COMPLETED SUCCESSFULLY")
        print("="*60)
        print(f"Total time: {duration/60:.1f} minutes")
        if df is not None:
            print(f"Total segments processed: {len(df)}")
            print(f"Final output: {args.output}")
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
