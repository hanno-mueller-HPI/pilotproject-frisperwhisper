import os
import glob
import numpy as np
import soundfile as sf
import librosa

INPUT_DIR = "data"
OUTPUT_DIR = "data/data_resample"
TARGET_SR = 16000

os.makedirs(OUTPUT_DIR, exist_ok=True)

wav_files = glob.glob(os.path.join(INPUT_DIR, "*.wav"))

for wav_path in wav_files:
    filename = os.path.basename(wav_path)
    name, ext = os.path.splitext(filename)
    out_path = os.path.join(OUTPUT_DIR, f"{name}16k.wav")
    if os.path.exists(out_path):
        print(f"SKIP (already exists): {out_path}")
        continue
    try:
        audio, orig_sr = sf.read(wav_path)
        audio_16k = librosa.resample(audio, orig_sr=orig_sr, target_sr=TARGET_SR)
        sf.write(out_path, audio_16k, TARGET_SR)
        print(f"OK: {wav_path} -> {out_path}")
    except Exception as e:
        print(f"FAIL: {wav_path} ({e})")