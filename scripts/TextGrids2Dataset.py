#############################################################################
# Script Name: TextGrids2Dataset.py                                         #
# Description: Creates train/test split of TextGrid data as HuggingFace     #
#              DataSetDict with memory-efficient parquet file storage       #
# Author: Hanno Müller                                                      #
# Date: 2025-07-16                                                          #
#############################################################################

### Required Libraries ######################################################
import os
import re
import gc
import chardet
import argparse
import soundfile as sf
import pandas as pd
import numpy as np
import librosa
from datasets import Dataset, DatasetDict, Features, Value
from multiprocessing import Pool, cpu_count
import traceback
import random
import shutil
from pathlib import Path

# Import cleaning functions
from clean_TextGrids import clean_textgrid_entries


### Class Definitions ########################################################

class TextGrid:
    def __init__(self, path, content):
        self.path = path
        self.content = content
        self.xmin = None
        self.xmax = None
        self.items = []  # List of dicts: {'name': ..., 'intervals': [...]}
        if content is not None:
            self._parse()

    @staticmethod
    def load_textgrid(path):
        """Loads a TextGrid file from disk and returns a TextGrid object with automatic encoding detection."""
        try:
            # Detect encoding using more data
            with open(path, "rb") as f:
                rawdata = f.read()  # Read the entire file for better detection
            result = chardet.detect(rawdata)
            encoding = result['encoding'] if result['encoding'] else 'utf-8'
        
            # Try detected encoding first
            try:
                with open(path, "r", encoding=encoding) as f:
                    content = f.read()
            except UnicodeDecodeError:
                # If detected encoding fails, try common encodings
                for fallback_encoding in ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']:
                    try:
                        with open(path, "r", encoding=fallback_encoding) as f:
                            content = f.read()
                        print(f"Used fallback encoding {fallback_encoding} for {os.path.basename(path)}")
                        break
                    except UnicodeDecodeError:
                        continue
                else:
                    # If all encodings fail, use utf-8 with error handling
                    with open(path, "r", encoding='utf-8', errors='replace') as f:
                        content = f.read()
                    print(f"Used utf-8 with error replacement for {os.path.basename(path)}")
        
            return TextGrid(path=path, content=content)
        except Exception as e:
            print(f"{os.path.basename(path)} could not be loaded")
            print(f"Error: {e}")
            traceback.print_exc()
            return None

    def save_textgrid(self, out_path=None):
        """
        Saves the TextGrid content to a file in a memory-efficient way.
        If out_path is not provided, uses self.path.
        """
        if out_path is None:
            out_path = self.path
        if self.content is not None and out_path is not None:
            with open(out_path, "w", encoding="utf-8") as f:
                f.write(self.content)

    def _parse(self):
        lines = self.content.splitlines()

        for line in lines:
            if line.strip().startswith("xmin ="):
                num_str = re.findall(r"[-+]?\d*\.\d+|\d+", line.strip().split("=")[1])[0]
                self.xmin = float(num_str)
                break
        for line in lines:
            if line.strip().startswith("xmax ="):
                num_str = re.findall(r"[-+]?\d*\.\d+|\d+", line.strip().split("=")[1])[0]
                self.xmax = float(num_str)
                break

        # Parse items (speakers)
        item_indices = [i for i, l in enumerate(lines) if l.strip().startswith("item [")]
        for idx, start in enumerate(item_indices):
            end = item_indices[idx + 1] if idx + 1 < len(item_indices) else len(lines)
            item_lines = lines[start:end]
            name = None
            intervals = []
            interval_block = []
            inside_interval = False
            for l in item_lines:
                stripped = l.strip()
                if stripped.startswith('name ='):
                    name = stripped.split('=')[1].strip().strip('"')
                if stripped.startswith('intervals ['):
                    inside_interval = True
                    interval_block = [stripped]
                elif inside_interval and (stripped.startswith('xmin =') or stripped.startswith('xmax =') or stripped.startswith('text =')):
                    interval_block.append(stripped)
                    if stripped.startswith('text ='):
                        # End of interval block
                        interval = {}
                        for entry in interval_block:
                            if entry.startswith('xmin ='):
                                interval['xmin'] = float(entry.split('=')[1])
                            elif entry.startswith('xmax ='):
                                interval['xmax'] = float(entry.split('=')[1])
                            elif entry.startswith('text ='):
                                interval['text'] = entry.split('=',1)[1].strip().strip('"')
                        intervals.append(interval)
                        inside_interval = False
                        interval_block = []
            if name and intervals:
                self.items.append({'name': name, 'intervals': intervals})

    def to_dataset_entries(self):
        """
        Converts the TextGrid to dataset entries without loading audio into memory.
        Audio is processed and saved to parquet files for memory efficiency.
        """
        entries = []
        audio_path = os.path.splitext(self.path)[0] + ".wav"
        
        if not os.path.exists(audio_path):
            print(f"Warning: Audio file {audio_path} not found")
            return entries
            
        # Check sampling rate
        with sf.SoundFile(audio_path) as audio_file:
            orig_sr = audio_file.samplerate
            if orig_sr != 16000:
                raise ValueError(f"Audio file {audio_path} has sampling rate {orig_sr} Hz, expected 16000 Hz")
            
            # Load full audio file
            audio_file.seek(0)
            audio = audio_file.read()
            audio = np.asarray(audio, dtype=np.float32)
            print(f"Loaded audio from {audio_path} with sampling rate {orig_sr} Hz")

        for item in self.items:
            speaker = item['name']
            for idx, interval in enumerate(item['intervals'], 1):
                start_sample = int(interval['xmin'] * orig_sr)
                end_sample = int(interval['xmax'] * orig_sr)
                
                if start_sample < end_sample:
                    array = audio[start_sample:end_sample]
                    
                    entry = {
                        'client_id': speaker if speaker else "",
                        'path': self.path,
                        'audio': {
                            'path': audio_path,
                            'array': array.tolist(),
                            'sampling_rate': orig_sr
                        },
                        'sentence': interval['text'],
                        'up_votes': 0,
                        'down_votes': 0,
                        'age': "",
                        'gender': "",
                        'accent': "",
                        'locale': "",
                        'segment': f"{idx}",
                        'textgrid_path': self.path,
                        'speaker': speaker,
                        'interval_idx': idx
                    }
                    entries.append(entry)
        
        return entries


### Function Definitions #####################################################

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Create train/test split of TextGrid data.")
    parser.add_argument(
        "-f", "--folder",
        type=str,
        required=True,
        help="Path to the folder containing TextGrid data"
    )
    parser.add_argument(
        "-o", "--output_folder",
        type=str,
        required=True,
        help="Path where the final DataSetDict should be stored"
    )
    parser.add_argument(
        "-n", "--number_of_processes",
        type=int,
        default=4,
        help="Number of processes to run in parallel (default: 4)"
    )
    parser.add_argument(
        "-c", "--csv_file",
        type=str,
        help="CSV file with test set intervals (columns: path, speaker, interval)"
    )
    return parser.parse_args()

def load_textgrids_from_folder(folder_path):
    """
    Takes a folder with textgrids and returns a list of TextGrid file paths.
    """
    textgrid_paths = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".TextGrid"):
            file_path = os.path.join(folder_path, filename)
            textgrid_paths.append(file_path)
    return textgrid_paths

def process_textgrid(args_tuple):
    """
    Process a single TextGrid file, apply cleaning filters, and save its dataset entries to parquet.
    Returns the path to the saved parquet file and entry count.
    """
    textgrid_path, tmp_dir, file_index = args_tuple
    
    try:
        textgrid = TextGrid.load_textgrid(textgrid_path)
        if textgrid is None:
            return None, 0
        
        entries = textgrid.to_dataset_entries()
        original_count = len(entries)
        
        # Apply cleaning filters
        cleaned_entries = clean_textgrid_entries(
            entries,
            remove_buzz_anon=True,
            remove_empty=True,
            remove_short_audio=True,
            min_duration=0.1
        )
        
        cleaned_count = len(cleaned_entries)
        removed_count = original_count - cleaned_count
        
        print(f"Processed {textgrid_path}: {original_count} entries -> {cleaned_count} entries "
              f"(removed {removed_count} entries)")
        
        if cleaned_entries:
            # Save cleaned entries to individual parquet file
            output_path = tmp_dir / f"textgrid_{file_index}.parquet"
            save_entries_to_parquet(cleaned_entries, output_path)
            return str(output_path), cleaned_count
        else:
            return None, 0
            
    except Exception as e:
        print(f"Error processing {textgrid_path}: {e}")
        traceback.print_exc()
        return None, 0

def save_entries_to_parquet(entries, output_path):
    """Save entries to a parquet file."""
    df = pd.DataFrame(entries)
    df.to_parquet(output_path, index=False)
    print(f"Saved {len(entries)} entries to {output_path}")

def combine_parquet_files(parquet_files, output_path):
    """Combine multiple parquet files into a single file."""
    if not parquet_files:
        return
    
    dfs = []
    for parquet_file in parquet_files:
        if parquet_file and os.path.exists(parquet_file):
            df = pd.read_parquet(parquet_file)
            dfs.append(df)
    
    if dfs:
        combined_df = pd.concat(dfs, ignore_index=True)
        combined_df.to_parquet(output_path, index=False)
        print(f"Combined {len(dfs)} parquet files into {output_path} with {len(combined_df)} entries")
    
    # Clean up individual files
    for parquet_file in parquet_files:
        if parquet_file and os.path.exists(parquet_file):
            os.remove(parquet_file)

def load_entries_from_parquet(parquet_path):
    """Load entries from parquet file."""
    if not os.path.exists(parquet_path):
        return []
    
    df = pd.read_parquet(parquet_path)
    return df.to_dict('records')

def load_csv_test_intervals(csv_path):
    """Load test intervals from CSV file."""
    if not csv_path or not os.path.exists(csv_path):
        return set()
    
    df = pd.read_csv(csv_path)
    required_columns = ['path', 'speaker', 'interval']
    
    if not all(col in df.columns for col in required_columns):
        print(f"Warning: CSV file must contain columns: {required_columns}")
        return set()
    
    test_intervals = set()
    for _, row in df.iterrows():
        # Create a unique identifier for each interval
        interval_id = f"{row['path']}_{row['speaker']}_{row['interval']}"
        test_intervals.add(interval_id)
    
    print(f"Loaded {len(test_intervals)} test intervals from CSV")
    return test_intervals

def create_interval_id(entry):
    """Create a unique identifier for an interval entry."""
    return f"{entry['textgrid_path']}_{entry['speaker']}_{entry['interval_idx']}"

def split_train_test(all_entries, test_intervals, seed=42):
    """
    Split entries into train and test sets based on CSV intervals and random sampling.
    """
    random.seed(seed)
    np.random.seed(seed)
    
    total_entries = len(all_entries)
    target_test_size = int(0.2 * total_entries)
    
    print(f"Total entries: {total_entries}")
    print(f"Target test size (20%): {target_test_size}")
    
    # Separate entries based on CSV intervals
    csv_test_entries = []
    remaining_entries = []
    
    for entry in all_entries:
        interval_id = create_interval_id(entry)
        if interval_id in test_intervals:
            csv_test_entries.append(entry)
        else:
            remaining_entries.append(entry)
    
    print(f"CSV test entries: {len(csv_test_entries)}")
    print(f"Remaining entries: {len(remaining_entries)}")
    
    if len(csv_test_entries) > target_test_size:
        # Too many CSV test entries, randomly sample
        test_entries = random.sample(csv_test_entries, target_test_size)
        test_ids = {create_interval_id(e) for e in test_entries}
        train_entries = remaining_entries + [e for e in csv_test_entries if create_interval_id(e) not in test_ids]
        print(f"Randomly sampled {target_test_size} entries from CSV test set")
    elif len(csv_test_entries) < target_test_size:
        # Not enough CSV test entries, add random entries
        additional_needed = target_test_size - len(csv_test_entries)
        additional_test = random.sample(remaining_entries, min(additional_needed, len(remaining_entries)))
        test_entries = csv_test_entries + additional_test
        additional_test_ids = {create_interval_id(e) for e in additional_test}
        train_entries = [e for e in remaining_entries if create_interval_id(e) not in additional_test_ids]
        print(f"Added {len(additional_test)} random entries to reach target test size")
    else:
        # Perfect match
        test_entries = csv_test_entries
        train_entries = remaining_entries
        print("CSV test entries exactly match target test size")
    
    print(f"Final train size: {len(train_entries)}")
    print(f"Final test size: {len(test_entries)}")
    
    return train_entries, test_entries

def create_dataset_from_entries(entries, name):
    """Create a HuggingFace Dataset from entries."""
    if not entries:
        return None
    
    features = Features({
        'client_id': Value(dtype='string'),
        'path': Value(dtype='string'),
        'audio': {
            'path': Value(dtype='string'),
            'array': [Value(dtype='float32')],
            'sampling_rate': Value(dtype='int64')
        },
        'sentence': Value(dtype='string'),
        'up_votes': Value(dtype='int64'),
        'down_votes': Value(dtype='int64'),
        'age': Value(dtype='string'),
        'gender': Value(dtype='string'),
        'accent': Value(dtype='string'),
        'locale': Value(dtype='string'),
        'segment': Value(dtype='string'),
    })
    
    # Remove extra fields not needed in final dataset
    clean_entries = []
    for entry in entries:
        clean_entry = {k: v for k, v in entry.items() if k in features}
        clean_entries.append(clean_entry)
    
    dataset = Dataset.from_list(clean_entries, features=features)
    print(f"Created {name} dataset with {len(dataset)} entries")
    return dataset


### main ######################################################################

if __name__ == "__main__":
    # Set random seed
    random.seed(42)
    np.random.seed(42)
    
    # Parse command line arguments
    args = parse_arguments()
    
    # Validate number of processes
    max_processes = cpu_count()
    if args.number_of_processes > max_processes:
        print(f"Warning: Requested {args.number_of_processes} processes, but only {max_processes} available")
        args.number_of_processes = max_processes
    
    # Create temporary directory for parquet files
    tmp_dir = Path("tmp_parquet")
    tmp_dir.mkdir(exist_ok=True)
    
    try:
        # Load TextGrid file paths
        textgrid_paths = load_textgrids_from_folder(args.folder)
        print(f"Found {len(textgrid_paths)} TextGrid files")
        
        # Process TextGrids in parallel - each worker saves to its own parquet file
        print(f"Processing TextGrids with {args.number_of_processes} processes...")
        
        # Prepare arguments for worker processes
        worker_args = [(path, tmp_dir, idx) for idx, path in enumerate(textgrid_paths)]
        
        with Pool(processes=args.number_of_processes) as pool:
            results = pool.map(process_textgrid, worker_args)
        
        # Collect parquet file paths and count total entries
        parquet_files = []
        total_entries = 0
        for parquet_path, entry_count in results:
            if parquet_path:
                parquet_files.append(parquet_path)
                total_entries += entry_count
        
        print(f"Total entries collected: {total_entries}")
        
        # Combine all parquet files into a single file
        all_entries_path = tmp_dir / "all_entries.parquet"
        combine_parquet_files(parquet_files, all_entries_path)
        
        # Load entries from the combined parquet file
        print("Loading entries from combined parquet file...")
        all_entries = load_entries_from_parquet(all_entries_path)
        
        # Load CSV test intervals if provided
        test_intervals = load_csv_test_intervals(args.csv_file)
        
        # Split into train and test
        train_entries, test_entries = split_train_test(all_entries, test_intervals)
        
        # Save train and test entries to separate parquet files
        train_path = tmp_dir / "train_entries.parquet"
        test_path = tmp_dir / "test_entries.parquet"
        
        save_entries_to_parquet(train_entries, train_path)
        save_entries_to_parquet(test_entries, test_path)
        
        # Clear entries from memory
        del all_entries, train_entries, test_entries
        gc.collect()
        
        # Create datasets from parquet files
        print("Creating train dataset...")
        train_entries = load_entries_from_parquet(train_path)
        train_dataset = create_dataset_from_entries(train_entries, "train")
        
        print("Creating test dataset...")
        test_entries = load_entries_from_parquet(test_path)
        test_dataset = create_dataset_from_entries(test_entries, "test")
        
        # Create DataSetDict
        dataset_dict = DatasetDict({
            'train': train_dataset,
            'test': test_dataset
        })
        
        # Save the final dataset
        output_path = Path(args.output_folder) / "LangAgeDataSet"
        output_path.mkdir(parents=True, exist_ok=True)
        
        print(f"Saving DataSetDict to {output_path}...")
        dataset_dict.save_to_disk(str(output_path))
        
        print("Dataset creation completed successfully!")
        print(f"Train dataset: {len(train_dataset) if train_dataset else 0} entries")
        print(f"Test dataset: {len(test_dataset) if test_dataset else 0} entries")
        
    except Exception as e:
        print(f"Error during dataset creation: {e}")
        traceback.print_exc()
    
    finally:
        # Clean up temporary directory
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir)
            print("Cleaned up temporary files")






    






