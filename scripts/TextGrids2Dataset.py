#############################################################################
# Script Name: sTextGrids2Database.py                                       #
# Description: This script takes a folder with TextGrids and converts them  #
#              into a database format.                                      #
# Author: Hanno Müller                                                      #
# Date: 2025-06-26                                                          #
#############################################################################

### Required Libraries ######################################################
import os
import argparse
import soundfile as sf
import sounddevice as sd # necessary for audio playback (e.g., debugging, testing)
import json


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
        """Loads a TextGrid file from disk and returns a TextGrid object."""
        try:
            with open(path, "r", encoding="utf-8") as f:
                content = f.read()
            return TextGrid(path=path, content=content)
        except Exception as e:
            print(f"{os.path.basename(path)} could not be loaded")
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

        # Get global xmin and xmax
        for line in lines:
            if line.strip().startswith("xmin ="):
                self.xmin = float(line.strip().split("=")[1])
                break
        for line in lines:
            if line.strip().startswith("xmax ="):
                self.xmax = float(line.strip().split("=")[1])
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

    def to_dict(self):
        dicts = []
        audio_path = os.path.splitext(self.path)[0] + ".wav"
        if not os.path.exists(audio_path):
            sampling_rate = None
            audio = None
        else:
            audio = sf.SoundFile(audio_path)
            sampling_rate = audio.samplerate

        for item in self.items:
            speaker = item['name']
            for idx, interval in enumerate(item['intervals'], 1):
                if sampling_rate is not None:
                    start_sample = int(interval['xmin'] * sampling_rate)
                    end_sample = int(interval['xmax'] * sampling_rate)
                    with sf.SoundFile(audio_path) as f:
                        f.seek(start_sample)
                        array = f.read(end_sample - start_sample)
                else:
                    array = []
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
        if audio is not None:
            audio.close()
        return dicts

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

###########################################
################## TO-DO ##################
### Add a function to textgrid class to resample to 16000 Hz (or other value)
### Save the Class into a library file
### Write a new script that loads textgrid by textgrid
### and starts tokenisation
###########################################

    from transformers import WhisperFeatureExtractor

    feature_extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-small")

    from transformers import WhisperTokenizer

    tokenizer = WhisperTokenizer.from_pretrained("openai/whisper-small", language="French", task="transcribe")

    input_str = textgrids[1].to_dict()[100]["sentence"]
    labels = tokenizer(input_str).input_ids
    decoded_with_special = tokenizer.decode(labels, skip_special_tokens=False)
    decoded_str = tokenizer.decode(labels, skip_special_tokens=True)

    print(f"Input:                 {input_str}")
    print(f"Decoded w/ special:    {decoded_with_special}")
    print(f"Decoded w/out special: {decoded_str}")
    print(f"Are equal:             {input_str == decoded_str}")

    
    from transformers import WhisperProcessor

    processor = WhisperProcessor.from_pretrained("openai/whisper-small", language="Hindi", task="transcribe")





    # code for debugging
    RUN=False
    if RUN:
        dicts = textgrids[0].to_dict()
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






    






