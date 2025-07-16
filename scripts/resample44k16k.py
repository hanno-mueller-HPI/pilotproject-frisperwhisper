import os
import glob
import argparse
import multiprocessing
from functools import partial
import numpy as np
import soundfile as sf
import librosa

def process_audio_file(wav_path, input_dir, target_sr):
    """Process a single audio file for resampling"""
    filename = os.path.basename(wav_path)
    
    print(f"PROCESSING: {filename}")
    
    try:
        audio, orig_sr = sf.read(wav_path)
        
        # Case 1: File already has target sample rate - skip
        if orig_sr == target_sr:
            print(f"SKIPPED: {filename} (already {target_sr}Hz)")
            return
        
        # Case 2: File needs resampling
        print(f"RESAMPLING: {filename} ({orig_sr}Hz -> {target_sr}Hz)")
        audio_resampled = librosa.resample(audio, orig_sr=orig_sr, target_sr=target_sr)
        
        # Overwrite the original file
        sf.write(wav_path, audio_resampled, target_sr)
        print(f"COMPLETED: {filename} - resampled and saved")
        
    except Exception as e:
        print(f"ERROR: {filename} - {e}")

def main():
    parser = argparse.ArgumentParser(description="Resample audio files to 16kHz")
    parser.add_argument("-i", "--input_folder", default=None,
                        help="Input folder containing .wav files")
    parser.add_argument("-p", "--processes", type=int, default=None,
                        help="Number of parallel processes (default: min(4, available_cores))")
    
    args = parser.parse_args()
    
    INPUT_DIR = args.input_folder
    TARGET_SR = 16000
    
    # Determine number of processes
    available_cores = multiprocessing.cpu_count()
    if args.processes is None:
        num_processes = min(4, available_cores)
    else:
        num_processes = min(args.processes, available_cores)
        if args.processes > available_cores:
            print(f"Warning: Requested {args.processes} processes, but only {available_cores} cores available. Using {num_processes}.")
    
    if not os.path.exists(INPUT_DIR):
        print(f"Error: Input directory '{INPUT_DIR}' does not exist")
        return
    
    wav_files = glob.glob(os.path.join(INPUT_DIR, "*.wav"))
    
    if not wav_files:
        print(f"No .wav files found in '{INPUT_DIR}'")
        return
    
    print(f"Found {len(wav_files)} .wav files in '{INPUT_DIR}'")
    print(f"Files will be overwritten with 16kHz version if needed")
    print(f"Using {num_processes} parallel processes")
    print()
    
    # Create a partial function with fixed arguments
    process_func = partial(
        process_audio_file,
        input_dir=INPUT_DIR,
        target_sr=TARGET_SR
    )
    
    # Process files in parallel
    with multiprocessing.Pool(processes=num_processes) as pool:
        pool.map(process_func, wav_files)
    
    print("\nAll files processed!")

if __name__ == "__main__":
    main()