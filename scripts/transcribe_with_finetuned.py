#!/usr/bin/env python3

#############################################################################
# Script Name: transcribe_with_finetuned.py                                #
# Description: Transcribe audio using fine-tuned Whisper model             #
# Author: Hanno Müller                                                      #
# Date: 2025-07-30                                                          #
#############################################################################

import os
import argparse
import torch
import librosa
from transformers import (
    WhisperProcessor,
    WhisperForConditionalGeneration
)


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Transcribe audio using fine-tuned Whisper model.")
    
    parser.add_argument(
        "-i", "--input_audio",
        type=str,
        required=True,
        help="Path to the audio file to transcribe"
    )
    parser.add_argument(
        "-m", "--model_path",
        type=str,
        required=True,
        help="Path to the fine-tuned Whisper model directory"
    )
    parser.add_argument(
        "-o", "--output_file",
        type=str,
        default=None,
        help="Output file to save transcription (optional, prints to stdout if not specified)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="transcripts",
        help="Base output directory for transcriptions (default: transcripts)"
    )
    parser.add_argument(
        "--version",
        type=str,
        default=None,
        help="Model version name for subfolder organization"
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
    
    return parser.parse_args()


def load_audio(audio_path, sample_rate=16000):
    """Load and preprocess audio file."""
    print(f"Loading audio from: {audio_path}")
    
    # Load audio using librosa
    audio, sr = librosa.load(audio_path, sr=sample_rate)
    
    print(f"Audio loaded: {len(audio)} samples, {len(audio)/sample_rate:.2f} seconds")
    return audio


def transcribe_audio(audio, model, processor, device, language="french"):
    """Transcribe audio using the fine-tuned model."""
    print("Processing audio features...")
    
    # Process audio to features
    inputs = processor(audio, sampling_rate=16000, return_tensors="pt")
    input_features = inputs.input_features.to(device)
    
    print("Generating transcription...")
    
    # Generate transcription
    with torch.no_grad():
        # Force the model to use French and transcription task
        forced_decoder_ids = processor.get_decoder_prompt_ids(
            language=language, 
            task="transcribe"
        )
        
        predicted_ids = model.generate(
            input_features,
            forced_decoder_ids=forced_decoder_ids,
            max_length=225,
            num_beams=5,
            early_stopping=True
        )
    
    # Decode the transcription
    transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
    
    return transcription


def main():
    """Main transcription function."""
    args = parse_arguments()
    
    print("=== Fine-tuned Whisper Transcription ===")
    print(f"Audio file: {args.input_audio}")
    print(f"Model path: {args.model_path}")
    print("=" * 42)
    
    # Determine device
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    
    print(f"Using device: {device}")
    
    # Check if audio file exists
    if not os.path.exists(args.input_audio):
        raise FileNotFoundError(f"Audio file not found: {args.input_audio}")
    
    # Check if model directory exists
    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"Model directory not found: {args.model_path}")
    
    # Load the fine-tuned model and processor
    print(f"Loading fine-tuned model from: {args.model_path}")
    processor = WhisperProcessor.from_pretrained(args.model_path)
    model = WhisperForConditionalGeneration.from_pretrained(args.model_path)
    
    # Move model to device
    model = model.to(device)
    model.eval()
    
    print("Model loaded successfully!")
    
    # Load and preprocess audio
    audio = load_audio(args.input_audio)
    
    # Transcribe
    transcription = transcribe_audio(audio, model, processor, device, args.language)
    
    # Output results
    print("\n" + "=" * 42)
    print("TRANSCRIPTION RESULT:")
    print("=" * 42)
    print(transcription)
    print("=" * 42)
    
    # Determine output file path
    output_file_path = None
    
    if args.output_file:
        # Use specific output file if provided
        output_file_path = args.output_file
    elif args.version:
        # Create organized folder structure: transcripts/version/filename.txt
        audio_filename = os.path.splitext(os.path.basename(args.input_audio))[0]
        version_dir = os.path.join(args.output, args.version)
        os.makedirs(version_dir, exist_ok=True)
        output_file_path = os.path.join(version_dir, f"{audio_filename}_transcription.txt")
    
    # Save to file if output path is determined
    if output_file_path:
        with open(output_file_path, 'w', encoding='utf-8') as f:
            f.write(f"Audio file: {args.input_audio}\n")
            f.write(f"Model: {args.model_path}\n")
            f.write(f"Language: {args.language}\n")
            f.write(f"Timestamp: {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("-" * 50 + "\n")
            f.write(transcription)
        print(f"Transcription saved to: {output_file_path}")
    else:
        print("No output file specified. Transcription displayed above only.")
    
    print("Transcription completed successfully!")


if __name__ == "__main__":
    main()
