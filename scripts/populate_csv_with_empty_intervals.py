#!/usr/bin/env python3
#############################################################################
# Script Name: populate_csv_with_empty_intervals.py                        #
# Description: Add empty intervals to whisper comparison CSV results        #
# Author: Hanno Müller                                                      #
# Date: 2025-09-10                                                          #
#############################################################################

import os
import sys
import pandas as pd
import argparse
from pathlib import Path
import re
import chardet
from collections import defaultdict

# Add modules to path for TextGrid parsing
sys.path.append(os.path.join(os.path.dirname(__file__), 'modules'))
from metadata_extraction import TextGrid
from speaker_mapping import create_speaker_mapping, get_speaker_info


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Populate CSV with empty intervals from TextGrid files"
    )
    
    parser.add_argument(
        "--input", "-i",
        type=str,
        required=True,
        help="Path to input CSV file (e.g., whisper_comparison_results.csv)"
    )
    parser.add_argument(
        "--output", "-o", 
        type=str,
        required=True,
        help="Path to output CSV file"
    )
    parser.add_argument(
        "--textgrid_dir",
        type=str,
        default="data_original",
        help="Directory containing TextGrid files (default: data_original)"
    )
    
    return parser.parse_args()


def find_textgrid_files(textgrid_dir):
    """Find all TextGrid files in the directory."""
    textgrid_files = {}
    for file_path in Path(textgrid_dir).rglob("*.TextGrid"):
        filename = file_path.stem  # Remove .TextGrid extension
        textgrid_files[filename] = str(file_path)
    return textgrid_files


def get_all_intervals_from_textgrid(textgrid_path):
    """Extract ALL intervals (including empty ones) from a TextGrid file."""
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


def create_empty_row_template(existing_row):
    """Create a template for empty interval rows based on existing CSV structure."""
    template = existing_row.copy()
    
    # Clear transcript and metric columns
    transcript_columns = ['transcript_original', 'transcript_large_v3', 'transcript_fine_tuned']
    metric_columns = [col for col in existing_row.index if any(metric in col.lower() 
                     for metric in ['wer', 'cer', 'bleu'])]
    linguistic_columns = [col for col in existing_row.index if col in [
        'ah', 'bah', 'beh', 'ben', 'chh', 'eh', 'euh', 'ha', 'hé', 'hein', 
        'hop', 'hum', 'm-hm', 'mmh', 'mm', 'oh', 'ouf', 'pff', 'youh',
        '(bru)', '(bou)', '(exp)', '(ges)', '(ins)', '(ono)', '(pau)', 
        '(rac)', '(rir)', '(tou)', '(tss)', '(sou)', '(buzz)', 'XXX'
    ]]
    
    # Set empty/null values
    for col in transcript_columns:
        if col in template.index:
            template[col] = ""
    
    for col in metric_columns:
        if col in template.index:
            template[col] = ""
    
    for col in linguistic_columns:
        if col in template.index:
            template[col] = 0
    
    return template


def populate_csv_with_empty_intervals(input_csv, output_csv, textgrid_dir):
    """Main function to populate CSV with empty intervals."""
    print(f"Loading CSV file: {input_csv}")
    df = pd.read_csv(input_csv)
    
    print(f"Original CSV has {len(df)} rows")
    
    # Find TextGrid files
    textgrid_files = find_textgrid_files(textgrid_dir)
    print(f"Found {len(textgrid_files)} TextGrid files")
    
    # Create speaker mapping for metadata
    trs_dir = textgrid_dir  # Assuming TRS files are in same directory
    try:
        speaker_mapping = create_speaker_mapping(trs_dir)
        print(f"Created speaker mapping with {len(speaker_mapping)} entries")
    except Exception as e:
        print(f"Warning: Could not create speaker mapping: {e}")
        speaker_mapping = {}
    
    new_rows = []
    
    # Group existing data by filename and speaker
    existing_data = defaultdict(lambda: defaultdict(dict))
    for _, row in df.iterrows():
        filename = row['filename']
        speaker = row['speaker_id'] 
        interval = int(row['interval'])
        existing_data[filename][speaker][interval] = row
    
    # Process each file
    for filename in existing_data.keys():
        if filename not in textgrid_files:
            print(f"Warning: TextGrid file not found for {filename}")
            continue
            
        print(f"Processing {filename}...")
        
        # Get all intervals from TextGrid
        all_intervals = get_all_intervals_from_textgrid(textgrid_files[filename])
        
        if not all_intervals:
            print(f"Warning: Could not parse intervals from {filename}")
            continue
        
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
                if interval_num in existing_data[filename][speaker]:
                    # Use existing row
                    new_rows.append(existing_data[filename][speaker][interval_num])
                else:
                    # Create new row for empty interval
                    if existing_data[filename][speaker]:
                        # Use first existing row as template
                        template_row = list(existing_data[filename][speaker].values())[0]
                        new_row = create_empty_row_template(template_row)
                    else:
                        # Create minimal row structure
                        new_row = pd.Series(dtype=object)
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
                    
                    new_rows.append(new_row)
    
    # Create new DataFrame
    result_df = pd.DataFrame(new_rows)
    
    # Sort by filename, speaker, and interval
    result_df = result_df.sort_values(['filename', 'speaker_id', 'interval']).reset_index(drop=True)
    
    print(f"New CSV has {len(result_df)} rows (added {len(result_df) - len(df)} empty intervals)")
    
    # Save result
    result_df.to_csv(output_csv, index=False)
    print(f"Saved complete CSV to: {output_csv}")
    
    # Print summary statistics
    empty_intervals = len(result_df[result_df['transcript_original'] == ""])
    non_empty_intervals = len(result_df[result_df['transcript_original'] != ""])
    
    print(f"\nSummary:")
    print(f"  Total intervals: {len(result_df)}")
    print(f"  Non-empty intervals: {non_empty_intervals}")
    print(f"  Empty intervals: {empty_intervals}")
    print(f"  Empty interval ratio: {empty_intervals/len(result_df)*100:.1f}%")


def main():
    """Main execution function."""
    args = parse_arguments()
    
    # Validate input file
    if not os.path.exists(args.input):
        print(f"Error: Input file does not exist: {args.input}")
        sys.exit(1)
    
    # Validate TextGrid directory
    if not os.path.exists(args.textgrid_dir):
        print(f"Error: TextGrid directory does not exist: {args.textgrid_dir}")
        sys.exit(1)
    
    # Create output directory if needed
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")
    
    try:
        populate_csv_with_empty_intervals(args.input, args.output, args.textgrid_dir)
        print("Script completed successfully!")
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
