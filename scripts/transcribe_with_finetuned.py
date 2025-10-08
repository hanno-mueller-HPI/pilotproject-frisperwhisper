#!/usr/bin/env python3

#############################################################################
# Script Name: transcribe_with_finetuned.py                                #
# Description: Transcribe audio using fine-tuned Whisper model             #
# Author: Hanno Müller                                                      #
# Date: 2025-10-08 (Enhanced)                                               #
#############################################################################

import os
import argparse
import torch
import librosa
import pandas as pd
from pathlib import Path
from typing import List, Dict
from transformers import (
    WhisperProcessor,
    WhisperForConditionalGeneration,
    pipeline
)


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Transcribe audio using Whisper model (local or HuggingFace)."
    )
    
    parser.add_argument(
        "-i", "--input",
        type=str,
        required=True,
        help="Path to audio file or directory containing audio files"
    )
    parser.add_argument(
        "-m", "--model",
        type=str,
        required=True,
        help="Model path (local directory) or HuggingFace model ID (e.g., 'openai/whisper-large-v3')"
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        required=True,
        help="Output CSV file path"
    )
    parser.add_argument(
        "--language",
        type=str,
        default="french",
        help="Language for transcription (default: french)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to use: 'cpu', 'cuda', or 'auto' (default: auto)"
    )
    parser.add_argument(
        "--use_pipeline",
        action="store_true",
        help="Use HuggingFace pipeline (automatic segmentation for long audio)"
    )
    
    return parser.parse_args()


def find_audio_files(input_path: str) -> List[str]:
    """
    Find audio files from input path (file or directory).
    
    Args:
        input_path: Path to audio file or directory
        
    Returns:
        List of audio file paths
    """
    audio_extensions = {'.wav', '.mp3', '.flac', '.m4a', '.ogg', '.opus'}
    
    if os.path.isfile(input_path):
        return [input_path]
    
    elif os.path.isdir(input_path):
        audio_files = []
        for file in sorted(os.listdir(input_path)):
            file_path = os.path.join(input_path, file)
            if os.path.isfile(file_path) and Path(file).suffix.lower() in audio_extensions:
                audio_files.append(file_path)
        return audio_files
    
    else:
        raise FileNotFoundError(f"Input path not found: {input_path}")


def load_model(model_path: str, device: str, use_pipeline: bool = False):
    """
    Load Whisper model from local path or HuggingFace Hub.
    
    Args:
        model_path: Local directory path or HuggingFace model ID
        device: Device to use ('cpu', 'cuda', or specific GPU)
        use_pipeline: Whether to use HuggingFace pipeline
        
    Returns:
        tuple: (model, processor) or pipeline object
    """
    # Check if it's a local path
    is_local = os.path.exists(model_path)
    
    if is_local:
        print(f"Loading model from local path: {model_path}")
    else:
        print(f"Loading model from HuggingFace Hub: {model_path}")
    
    if use_pipeline:
        # Use HuggingFace pipeline for automatic segmentation
        pipe = pipeline(
            "automatic-speech-recognition",
            model=model_path,
            device=device if device != "cpu" else -1,
            return_timestamps=True
        )
        return pipe
    else:
        # Load model and processor separately
        processor = WhisperProcessor.from_pretrained(model_path)
        model = WhisperForConditionalGeneration.from_pretrained(model_path)
        model = model.to(device)
        model.eval()
        return model, processor


def transcribe_with_pipeline(audio_path: str, pipe, language: str = "french") -> List[Dict]:
    """
    Transcribe audio using HuggingFace pipeline (with automatic segmentation).
    
    Args:
        audio_path: Path to audio file
        pipe: HuggingFace pipeline object
        language: Language code
        
    Returns:
        List of segment dictionaries with start, end, and text
    """
    print(f"Transcribing: {audio_path}")
    
    # Transcribe with timestamps
    result = pipe(
        audio_path,
        generate_kwargs={"language": language, "task": "transcribe"},
        return_timestamps=True
    )
    
    # Extract segments
    segments = []
    if "chunks" in result:
        for chunk in result["chunks"]:
            segments.append({
                "start_ms": chunk["timestamp"][0] * 1000 if chunk["timestamp"][0] is not None else 0.0,
                "end_ms": chunk["timestamp"][1] * 1000 if chunk["timestamp"][1] is not None else 0.0,
                "text": chunk["text"].strip()
            })
    else:
        # No chunks, treat as single segment
        segments.append({
            "start_ms": 0.0,
            "end_ms": 0.0,  # Unknown end time
            "text": result["text"].strip()
        })
    
    return segments


def transcribe_with_model(audio_path: str, model, processor, device: str, language: str = "french") -> List[Dict]:
    """
    Transcribe audio using model and processor (whole file at once).
    
    Args:
        audio_path: Path to audio file
        model: Whisper model
        processor: Whisper processor
        device: Device to use
        language: Language code
        
    Returns:
        List with single segment dictionary
    """
    print(f"Transcribing: {audio_path}")
    
    # Load audio
    audio, sr = librosa.load(audio_path, sr=16000)
    duration_ms = (len(audio) / sr) * 1000
    
    # Process audio to features
    inputs = processor(audio, sampling_rate=16000, return_tensors="pt")
    input_features = inputs.input_features.to(device)
    
    # Generate transcription
    with torch.no_grad():
        forced_decoder_ids = processor.get_decoder_prompt_ids(
            language=language, 
            task="transcribe"
        )
        
        predicted_ids = model.generate(
            input_features,
            forced_decoder_ids=forced_decoder_ids,
            max_length=448,  # Increased for longer audio
            num_beams=5,
            early_stopping=True
        )
    
    # Decode the transcription
    transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
    
    # Return as single segment
    return [{
        "start_ms": 0.0,
        "end_ms": duration_ms,
        "text": transcription.strip()
    }]


def format_time_ms(milliseconds: float) -> str:
    """
    Format milliseconds as MM:SS.mmm
    
    Args:
        milliseconds: Time in milliseconds
        
    Returns:
        Formatted time string
    """
    if milliseconds == 0.0:
        return "00:00.000"
    
    total_seconds = milliseconds / 1000
    minutes = int(total_seconds // 60)
    seconds = total_seconds % 60
    
    return f"{minutes:02d}:{seconds:06.3f}"


def create_csv(all_segments: List[Dict], output_path: str, is_directory: bool):
    """
    Create CSV file with transcription results.
    
    Args:
        all_segments: List of segment dictionaries with filename, start_ms, end_ms, text
        output_path: Path to output CSV file
        is_directory: Whether input was a directory (adds filename column)
    """
    # Prepare data
    rows = []
    for idx, segment in enumerate(all_segments, start=1):
        row = {
            "ID": idx,
            "Start": format_time_ms(segment["start_ms"]),
            "Stop": format_time_ms(segment["end_ms"]),
            "Transcription": segment["text"]
        }
        
        # Add filename column if processing directory
        if is_directory:
            row["Filename"] = segment["filename"]
        
        rows.append(row)
    
    # Create DataFrame
    if is_directory:
        df = pd.DataFrame(rows, columns=["ID", "Filename", "Start", "Stop", "Transcription"])
    else:
        df = pd.DataFrame(rows, columns=["ID", "Start", "Stop", "Transcription"])
    
    # Save to CSV
    df.to_csv(output_path, index=False, encoding='utf-8')
    print(f"\n✓ CSV saved to: {output_path}")
    print(f"  Total segments: {len(rows)}")


def main():
    """Main transcription function."""
    args = parse_arguments()
    
    print("=" * 60)
    print("WHISPER TRANSCRIPTION")
    print("=" * 60)
    print(f"Input: {args.input}")
    print(f"Model: {args.model}")
    print(f"Output: {args.output}")
    print(f"Language: {args.language}")
    
    # Determine device
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    
    print(f"Device: {device}")
    print("=" * 60)
    
    # Find audio files
    audio_files = find_audio_files(args.input)
    is_directory = os.path.isdir(args.input)
    
    print(f"\nFound {len(audio_files)} audio file(s)")
    
    if len(audio_files) == 0:
        print("No audio files found. Exiting.")
        return
    
    # Load model
    print("\nLoading model...")
    if args.use_pipeline:
        pipe = load_model(args.model, device, use_pipeline=True)
        print("✓ Pipeline loaded successfully")
        model_objects = (pipe,)
    else:
        model, processor = load_model(args.model, device, use_pipeline=False)
        print("✓ Model loaded successfully")
        model_objects = (model, processor)
    
    # Transcribe all files
    print(f"\nTranscribing {len(audio_files)} file(s)...")
    all_segments = []
    
    for audio_path in audio_files:
        try:
            if args.use_pipeline:
                segments = transcribe_with_pipeline(audio_path, model_objects[0], args.language)
            else:
                segments = transcribe_with_model(audio_path, model_objects[0], model_objects[1], device, args.language)
            
            # Add filename to each segment
            filename = os.path.basename(audio_path)
            for segment in segments:
                segment["filename"] = filename
            
            all_segments.extend(segments)
            print(f"  ✓ {filename}: {len(segments)} segment(s)")
            
        except Exception as e:
            print(f"  ✗ Error transcribing {os.path.basename(audio_path)}: {e}")
            import traceback
            traceback.print_exc()
    
    # Create CSV
    if len(all_segments) > 0:
        print(f"\nCreating CSV with {len(all_segments)} total segments...")
        create_csv(all_segments, args.output, is_directory)
        
        print("\n" + "=" * 60)
        print("TRANSCRIPTION COMPLETED SUCCESSFULLY")
        print("=" * 60)
    else:
        print("\nNo segments to save. Exiting.")
        print("=" * 60)


if __name__ == "__main__":
    main()
