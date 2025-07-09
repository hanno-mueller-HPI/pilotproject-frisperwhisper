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
import gc
import chardet
import argparse
import soundfile as sf
import sounddevice as sd # necessary for audio playback (e.g., debugging, testing)
import numpy as np
import librosa
from datasets import Dataset, DatasetDict, Features, Value, Audio
import traceback


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
            #print(f"Detected encoding for {os.path.basename(path)}: {encoding}")
            # Read file with detected encoding
            with open(path, "r", encoding=encoding) as f:
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
                        continue
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


### main ######################################################################

if __name__ == "__main__":

    # Retrieve command line arguments
    vars = parse_arguments()
    
    # Load TextGrids from the specified folder
    textgrids = load_textgrids_from_folder(vars.folder)

    # check transformation into dataset
    RUN=True
    if RUN:
        for idx, tg in enumerate(textgrids):
            print('*'*20)
            print(f"Processing TextGrid {tg.path}")
            try:
                ds = tg.to_dataset(resample=16000)
                print(f"Processed {tg.path}")
                print(ds[20:23])
                del ds
                gc.collect()  # Free memory
            except Exception as e:
                print(f"Conversion failed at textgrid index {idx} (path: {tg}): {e}")
    
    # code for debugging
    RUN=False
    if RUN:
        dicts = textgrids[0].to_dict(resample=16000)
        spk2_entries = [d for d in dicts if d['speaker'] == 'spk2']

        for i, entry in enumerate(spk2_entries[1:]):  # Skip the first entry
            print(entry)
            input("Press Enter to play the next audio...")
            audio_array = entry['audio']['array']
            sampling_rate = entry['audio']['sampling_rate']
            sd.play(audio_array, sampling_rate)
            sd.wait()
            print('*' * 20)
            if i >= 10:
                break






    






