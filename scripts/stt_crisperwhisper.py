#############################################################################
# Script Name: stt_crisperwhisper.py                                        #
# Description: This script uses the CrisperWhisper model for speech-to-text #
#              transcription, adjusting pause timings in the output.        #
# Author: Hanno Müller                                                      #
# Date: 2025-06-10                                                          #
#############################################################################

### Required Libraries ######################################################
import os
import sys
import argparse
import torch
import glob

import soundfile as sf

from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline


### Function Definitions #####################################################

def parse_arguments():
    """
    Parse command line arguments for the script.
    Example usage:                                                            
    $ python stt_crisperwhisper.py -m file -i path/to/audio.wav                 
    $ python stt_crisperwhisper.py -m directory -i path/to/directory/ 
    """
        
    parser = argparse.ArgumentParser("Usage: python stt_crsiperwhisper.py -m [file|directory] -i path/to/input")
    parser.add_argument("-m", "--mode", type=str, help="Whether the script runs in single file mode or directory mode.")
    parser.add_argument("-i", "--input", type=str, help="Path to a .wav file or a directory containing .wav files.")
    args = parser.parse_args()
    if not args.input:
        sys.exit("Please specify the mode (-m) and provide a path to a .wav file or a directory containing .wav files using the -i argument.")
    
    vars = {
        "mode": args.mode,
        "input": args.input
    }
    
    return vars


def read_audio(wavs, file_path):
    """
    Reads a single wav file into an array with sampling rate.
    """

    print(f"Loading wav file: {file_path}")
    array, sampling_rate = sf.read(file_path)
    audio = {
        "path": file_path,
        "array": array,
        "sampling_rate": sampling_rate
        }
    wavs.append(audio)

    return wavs


def load_wavs(vars):
    """
    Checks if vars["mode"] is 'file' or 'directory'.
    If 'file', reads the wav into an array with sampling rate.
    If 'directory', reads all wavs in the folder into arrays with sampling rates.
    Returns a list of tuples: (filepath, audio_array, sample_rate)
    """
    file_path = vars["input"]
    mode = vars["mode"]
    wavs = []
    if mode == "file":
        if os.path.isfile(file_path) and input_var.lower().endswith('.wav'):
            wavs = read_audio(wavs, file_path)
        else:
            raise ValueError(f"{file_path} is not a valid .wav file.")
    elif mode == "directory":
        if os.path.isdir(file_path):
            wav_files = glob.glob(os.path.join(file_path, "*.wav"))
            for wav_path in wav_files:
                wavs = read_audio(wavs, wav_path)
        else:
            raise ValueError(f"{input_var} is not a valid directory.")
    else:
        raise ValueError("Mode must be 'file' or 'directory'.")
    
    return wavs


def setup_pipeline():
    """
    Set up the CrisperWhisper pipeline for speech-to-text transcription.
    """
    
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    model_id = "nyrahealth/CrisperWhisper"

    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
    )
    model.to(device)

    processor = AutoProcessor.from_pretrained(model_id)

    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        chunk_length_s=30,
        batch_size=4,
        return_timestamps='word',
        torch_dtype=torch_dtype,
        device=device,
    )

    return pipe


def transcribe(pipe, wavs):
    """
    Transcribe audio using the CrisperWhisper pipeline.
    If mode is 'file', transcribe a single file.
    If mode is 'directory', transcribe all files in the directory.
    Returns the transcription result.
    """
    
    if len(wavs) == 1:
        print(f"Transcribing single file: {wavs[0][0]}")
        audio, sr = wavs[0][1], wavs[0][2]
        transcription = pipe(audio)
    elif len(wavs) > 1:
        transcription = []
        for  i, wav in enumerate(wavs):
            print(f"Transcribing file {i} ({wavs[i][0]}) of {len(wavs)}")
            audio, sr = wav[i][1], wav[i][2]
            transcription_i = pipe(audio)
            transcription.append(transcription_i)
    else:
        raise ValueError("Mode must be 'file' or 'directory'.")
    
    return transcription


def adjust_pauses_for_hf_pipeline_output(pipeline_output, split_threshold=0.12):
    """
    Adjust pause timings by distributing pauses up to the threshold evenly between adjacent words.
    """

    adjusted_chunks = pipeline_output["chunks"].copy()

    for i in range(len(adjusted_chunks) - 1):
        current_chunk = adjusted_chunks[i]
        next_chunk = adjusted_chunks[i + 1]

        current_start, current_end = current_chunk["timestamp"]
        next_start, next_end = next_chunk["timestamp"]
        pause_duration = next_start - current_end

        if pause_duration > 0:
            if pause_duration > split_threshold:
                distribute = split_threshold / 2
            else:
                distribute = pause_duration / 2

            # Adjust current chunk end time
            adjusted_chunks[i]["timestamp"] = (current_start, current_end + distribute)

            # Adjust next chunk start time
            adjusted_chunks[i + 1]["timestamp"] = (next_start - distribute, next_end)
    pipeline_output["chunks"] = adjusted_chunks

    return pipeline_output


### main ######################################################################

if __name__ == "__main__":

    # Check for command line arguments
    vars = parse_arguments()

    # Load dataset
    wavs = load_wavs(vars)

    # Configure CrisperWhisper pipeline
    pipe = setup_pipeline()

    # Transcribe audio using CrisperWhisper
    transcription = transcribe(pipe, wavs)
    crisper_whisper_result = adjust_pauses_for_hf_pipeline_output(hf_pipeline_output)

    # Write transcription to file
    print(crisper_whisper_result)





    






