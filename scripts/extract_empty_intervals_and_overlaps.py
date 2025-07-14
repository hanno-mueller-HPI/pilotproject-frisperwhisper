#############################################################################
# Script Name: sTextGrids2Database.py                                       #
# Description: This script takes a folder with TextGrids and converts them  #
#              into a database format.                                      #
# Author: Hanno Müller                                                      #
# Date: 2025-06-26                                                          #
#############################################################################

### Required Libraries ######################################################
import os
import re
import chardet
import argparse
import soundfile as sf
import sounddevice as sd # necessary for audio playback (e.g., debugging, testing)
import numpy as np
import librosa
from datasets import Dataset, DatasetDict, Features, Value, Audio
import traceback
import csv


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
            # Detect encoding
            with open(path, "rb") as f:
                rawdata = f.read(4096)
            result = chardet.detect(rawdata)
            encoding = result['encoding'] if result['encoding'] else 'utf-8'
        
            # Try detected encoding first
            try:
                with open(path, "r", encoding=encoding) as f:
                    content = f.read()
            except UnicodeDecodeError:
                # Fallback to utf-8 if detected encoding fails
                try:
                    with open(path, "r", encoding='utf-8') as f:
                        content = f.read()
                except UnicodeDecodeError:
                    # Final fallback to latin-1 (can decode any byte sequence)
                    with open(path, "r", encoding='latin-1') as f:
                        content = f.read()
        
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

    def to_dict(self, resample=None):
        """
        Converts TextGrid intervals to dicts.
        If resample is an integer, audio is resampled to that sampling rate.
        If resample is None (default), no resampling is performed.
        """
        dicts = []
        audio_path = os.path.splitext(self.path)[0] + ".wav"
        if not os.path.exists(audio_path):
            sampling_rate = None
        else:
            with sf.SoundFile(audio_path) as audio_file:
                orig_sr = audio_file.samplerate

        for item in self.items:
            speaker = item['name']
            for idx, interval in enumerate(item['intervals'], 1):
                if os.path.exists(audio_path):
                    start_sample = int(interval['xmin'] * orig_sr)
                    end_sample = int(interval['xmax'] * orig_sr)
                    with sf.SoundFile(audio_path) as f:
                        f.seek(start_sample)
                        array = f.read(end_sample - start_sample)
                    # Resample if requested
                    if resample is not None and isinstance(resample, int) and resample != orig_sr:
                        array = librosa.resample(np.asarray(array), orig_sr=orig_sr, target_sr=resample)
                        sampling_rate = resample
                    else:
                        sampling_rate = orig_sr
                else:
                    array = []
                    sampling_rate = None
                d = {
                    'path': self.path,
                    'audio': {
                        'path': audio_path,
                        'array': array,
                        'sampling_rate': sampling_rate
                    },
                    'sentence': interval['text'],
                    'speaker': speaker,
                    'interval': idx,
                    'xmin': interval['xmin'],
                    'xmax': interval['xmax']
                }
                dicts.append(d)
        return dicts
    
    def to_dataset(self, resample=None):
        """
        Converts the TextGrid to a Hugging Face Dataset with the specified structure.
        Fills missing metadata fields with default or empty values.
        """
        dicts = []
        audio_path = os.path.splitext(self.path)[0] + ".wav"
        audio = None
        sampling_rate = None

        if os.path.exists(audio_path):
            with sf.SoundFile(audio_path) as audio_file:
                orig_sr = audio_file.samplerate
                audio_file.seek(0)
                audio = audio_file.read()
                print(f"Loaded audio from {audio_path} with original sampling rate {orig_sr} Hz")
            if resample is not None and isinstance(resample, int) and resample != orig_sr:
                audio = librosa.resample(np.asarray(audio), orig_sr=orig_sr, target_sr=resample)
                sampling_rate = resample
                print(f"Resampled audio to {sampling_rate} Hz")
            else:
                sampling_rate = orig_sr

        for item in self.items:
            speaker = item['name']
            for idx, interval in enumerate(item['intervals'], 1):
                if audio is not None:
                    start_sample = int(interval['xmin'] * sampling_rate)
                    end_sample = int(interval['xmax'] * sampling_rate)
                    if start_sample < end_sample:
                        array = audio[start_sample:end_sample]
                        print(f"Extracted audio segment from {start_sample} to {end_sample} samples")
                    else:
                        array = []
                else:
                    array = []
                d = {
                    'client_id': speaker if speaker else "",
                    'path': self.path,
                    'audio': {
                        'path': audio_path,
                        'array': array,
                        'sampling_rate': sampling_rate
                    },
                    'sentence': interval['text'],
                    'up_votes': 0,
                    'down_votes': 0,
                    'age': "",
                    'gender': "",
                    'accent': "",
                    'locale': "",
                    'segment': f"{idx}"
                }
                dicts.append(d)

        features = Features({
            'client_id': Value(dtype='string'),
            'path': Value(dtype='string'),
            'audio': Audio(sampling_rate=48000, mono=True, decode=True),
            'sentence': Value(dtype='string'),
            'up_votes': Value(dtype='int64'),
            'down_votes': Value(dtype='int64'),
            'age': Value(dtype='string'),
            'gender': Value(dtype='string'),
            'accent': Value(dtype='string'),
            'locale': Value(dtype='string'),
            'segment': Value(dtype='string'),
        })

        dataset = Dataset.from_list(dicts, features=features)
        return dataset

### Function Definitions #####################################################

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Process TextGrid data from a specified folder.")
    parser.add_argument(
        "-f", "--folder",
        type=str,
        required=True,
        help="Path to the folder containing TextGrid data"
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default="./",
        help="Output directory for CSV files (default: current directory)"
    )
    return parser.parse_args()

def load_textgrids_from_folder(folder_path):
    """
    Takes a folder with textgrids and returns a list of TextGrid objects.
    """
    textgrids = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".TextGrid"):
            file_path = os.path.join(folder_path, filename)
            textgrids.append(TextGrid.load_textgrid(file_path))
    return textgrids

def extract_empty_intervals_to_csv(textgrids, output_file="./empty_intervals.csv"):
    """Extract all intervals where xmin == xmax and write to CSV."""
    with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['path', 'interval', 'xmin', 'xmax'])
        
        for tg in textgrids:
            if tg is None:
                continue
            for item in tg.items:
                for idx, interval in enumerate(item['intervals'], 1):
                    if interval['xmin'] == interval['xmax']:
                        writer.writerow([
                            tg.path,
                            idx,
                            interval['xmin'],
                            interval['xmax']
                        ])

def extract_empty_intervals_to_csv(textgrids, output_dir="./"):
    """Extract all intervals where xmin == xmax and write to CSV."""
    output_file = os.path.join(output_dir, "empty_intervals.csv")
    with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['path', 'interval', 'xmin', 'xmax'])
        
        for tg in textgrids:
            if tg is None:
                continue
            for item in tg.items:
                for idx, interval in enumerate(item['intervals'], 1):
                    if interval['xmin'] == interval['xmax']:
                        writer.writerow([
                            tg.path,
                            idx,
                            interval['xmin'],
                            interval['xmax']
                        ])

def extract_overlapping_intervals_to_csv(textgrids, output_dir="./"):
    """Extract overlapping intervals and write to CSV."""
    output_file = os.path.join(output_dir, "overlapping_intervals.csv")
    with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([
            'path', 'spk1_text', 'spk1_interval', 'spk1_xmin', 'spk1_xmax',
            'spk2_text', 'spk2_interval', 'spk2_xmin', 'spk2_xmax'
        ])
        
        for tg in textgrids:
            if tg is None:
                continue
            intervals_with_speakers = []
            for item in tg.items:
                speaker = item['name']
                for idx, interval in enumerate(item['intervals'], 1):
                    if interval['text'].strip():  # Only non-empty intervals
                        intervals_with_speakers.append({
                            'speaker': speaker,
                            'interval': interval,
                            'idx': idx
                        })
            
            # Check for overlaps
            for i, a in enumerate(intervals_with_speakers):
                for j, b in enumerate(intervals_with_speakers):
                    if i >= j or a['speaker'] == b['speaker']:
                        continue
                    
                    # Check overlap
                    a_start, a_end = a['interval']['xmin'], a['interval']['xmax']
                    b_start, b_end = b['interval']['xmin'], b['interval']['xmax']
                    overlap_duration = max(0, min(a_end, b_end) - max(a_start, b_start))
                    
                    if overlap_duration > 0:
                        writer.writerow([
                            tg.path,
                            a['interval']['text'],
                            a['idx'],
                            a_start,
                            a_end,
                            b['interval']['text'],
                            b['idx'],
                            b_start,
                            b_end
                        ])
                        

### main ######################################################################

if __name__ == "__main__":

    vars = parse_arguments()
    textgrids = load_textgrids_from_folder(vars.folder)

    # Create output directory if it doesn't exist
    os.makedirs(vars.output, exist_ok=True)
    
    # Extract intervals where xmin == xmax
    FILLERS = re.compile(r"^\s*(uhm+|mhm+|uh+|ah+|hmm+|hm+|äh+|mm+|um+|er+|em+|eh+)\s*$", re.IGNORECASE)
    extract_empty_intervals_to_csv(textgrids, vars.output)
    print(f"Extracted empty intervals (xmin==xmax) to {os.path.join(vars.output, 'empty_intervals.csv')}")
    
    # Extract overlapping intervals
    extract_overlapping_intervals_to_csv(textgrids, vars.output)
    print(f"Extracted overlapping intervals to {os.path.join(vars.output, 'overlapping_intervals.csv')}")