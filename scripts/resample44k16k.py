import os
import glob
import argparse
import multiprocessing
from functools import partial
import numpy as np
import soundfile as sf
import librosa

def process_audio_file(wav_path, input_dir, target_sr, delete_original):
    """Process a single audio file for resampling"""
    filename = os.path.basename(wav_path)
    name, ext = os.path.splitext(filename)
    
    # Print start message
    print(f"PROCESSING: {filename}")
    
    try:
        audio, orig_sr = sf.read(wav_path)
        
        # Case 1: File already ends in 16k.wav and has 16kHz sample rate - do nothing
        if filename.endswith("16k.wav") and orig_sr == target_sr:
            result = f"SKIP (already 16k.wav with {target_sr}Hz): {wav_path}"
            print(f"COMPLETED: {filename} - skipped (already processed)")
            return result
        
        # Case 2: File doesn't end in 16k.wav but has 16kHz sample rate - rename it
        elif not filename.endswith("16k.wav") and orig_sr == target_sr:
            out_path = os.path.join(input_dir, f"{name}16k.wav")
            
            if os.path.exists(out_path):
                result = f"SKIP (target already exists): {out_path}"
                print(f"COMPLETED: {filename} - skipped (target exists)")
                return result
            
            # Rename the file by adding 16k to the end
            os.rename(wav_path, out_path)
            result = f"RENAME (already {target_sr}Hz): {wav_path} -> {out_path}"
            print(f"COMPLETED: {filename} - renamed to {os.path.basename(out_path)}")
            return result
        
        # Case 3: File doesn't have 16kHz sample rate - resample and save with 16k.wav ending
        else:
            out_path = os.path.join(input_dir, f"{name}16k.wav")
            
            if os.path.exists(out_path):
                result = f"SKIP (already exists): {out_path}"
                print(f"COMPLETED: {filename} - skipped (target exists)")
                return result
            
            # Resample to 16kHz
            audio_16k = librosa.resample(audio, orig_sr=orig_sr, target_sr=target_sr)
            sf.write(out_path, audio_16k, target_sr)
            
            result = f"RESAMPLE ({orig_sr}Hz -> {target_sr}Hz): {wav_path} -> {out_path}"
            print(f"COMPLETED: {filename} - resampled and saved as {os.path.basename(out_path)}")
            
            # Delete original file if option is set to delete
            if delete_original:
                os.remove(wav_path)
                result += f"\nDELETED: {wav_path}"
                
            return result
        
    except Exception as e:
        result = f"FAIL: {wav_path} ({e})"
        print(f"FAILED: {filename} - {e}")
        return result

def main():
    parser = argparse.ArgumentParser(description="Resample audio files to 16kHz")
    parser.add_argument("-i", "--input_folder", default=None,
                        help="Input folder containing .wav files")
    parser.add_argument("-o", "--option", choices=["delete", "keep"], default="delete",
                        help="Option: 'delete' to remove original files, 'keep' to preserve them (default: keep)")
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
    print(f"Option: {'Delete original files' if args.option == 'delete' else 'Keep original files'}")
    print(f"Using {num_processes} parallel processes")
    print()
    
    delete_original = args.option == "delete"
    
    # Create a partial function with fixed arguments
    process_func = partial(
        process_audio_file,
        input_dir=INPUT_DIR,
        target_sr=TARGET_SR,
        delete_original=delete_original
    )
    
    # Process files in parallel
    with multiprocessing.Pool(processes=num_processes) as pool:
        results = pool.map(process_func, wav_files)
    
    # Print all results
    for result in results:
        print(result)

if __name__ == "__main__":
    main()