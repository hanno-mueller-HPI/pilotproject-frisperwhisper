#############################################################################
# Module: transcription.py                                                  #
# Description: GPU-accelerated transcription with batch processing          #
# Author: Hanno Müller                                                      #
# Date: 2025-09-03                                                          #
#############################################################################

import os
import torch
import numpy as np
import soundfile as sf
import librosa
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from multiprocessing import Pool
import tempfile
import subprocess
from pathlib import Path


class WhisperTranscriber:
    """GPU-accelerated Whisper transcriber with batch processing."""
    
    def __init__(self, model_path=None, device_map="auto", batch_size=32):
        """
        Initialize Whisper transcriber.
        
        Args:
            model_path (str, optional): Path to fine-tuned model. If None, uses openai/whisper-large-v3
            device_map (str): Device mapping strategy
            batch_size (int): Batch size for processing
        """
        self.batch_size = batch_size
        self.device_map = device_map
        
        # Determine model to load
        if model_path:
            self.model_name = model_path
            self.is_fine_tuned = True
        else:
            self.model_name = "openai/whisper-large-v3"
            self.is_fine_tuned = False
        
        print(f"Loading Whisper model: {self.model_name}")
        
        # Load model and processor
        try:
            # Always use the base model's processor for consistency
            base_model = "openai/whisper-large-v3" if self.is_fine_tuned else self.model_name
            self.processor = WhisperProcessor.from_pretrained(base_model)
            
            # Determine device and dtype based on availability
            if torch.cuda.is_available() and device_map != "cpu":
                device_map = device_map
                torch_dtype = torch.float16
                print(f"   Using GPU with float16")
            else:
                device_map = "cpu"
                torch_dtype = torch.float32
                print(f"   Using CPU with float32")
            
            self.model = WhisperForConditionalGeneration.from_pretrained(
                self.model_name,
                device_map=device_map,
                torch_dtype=torch_dtype,
                attn_implementation="eager"  # Use eager attention to avoid compatibility issues
            )
            
            # Ensure model and inputs use the same dtype
            self.torch_dtype = torch_dtype
            
            print(f"   Model loaded on: {device_map}")
                
        except Exception as e:
            print(f"Error loading model: {e}")
            raise
    
    def load_audio_segment(self, audio_path, start_time, end_time, target_sr=16000):
        """
        Load a specific audio segment from file.
        
        Args:
            audio_path (str): Path to audio file
            start_time (float): Start time in seconds
            end_time (float): End time in seconds
            target_sr (int): Target sampling rate
            
        Returns:
            np.ndarray: Audio array
        """
        try:
            # Method 1: Try soundfile with seek (for smaller files)
            start_sample = int(start_time * target_sr)
            end_sample = int(end_time * target_sr)
            
            with sf.SoundFile(audio_path) as audio_file:
                # Check if file might be too large for reliable seeking
                file_duration = audio_file.frames / audio_file.samplerate
                if file_duration > 1800:  # 30 minutes threshold
                    raise Exception("File too large for soundfile seeking")
                
                # Resample if needed
                if audio_file.samplerate != target_sr:
                    # Load full segment and resample
                    audio_file.seek(int(start_time * audio_file.samplerate))
                    frames_to_read = int((end_time - start_time) * audio_file.samplerate)
                    audio_array = audio_file.read(frames_to_read)
                    audio_array = librosa.resample(
                        audio_array, 
                        orig_sr=audio_file.samplerate, 
                        target_sr=target_sr
                    )
                else:
                    audio_file.seek(start_sample)
                    frames_to_read = end_sample - start_sample
                    audio_array = audio_file.read(frames_to_read)
                
            return np.asarray(audio_array, dtype=np.float32)
            
        except Exception:
            # Method 2: Fallback to ffmpeg for large files
            try:
                duration = end_time - start_time
                with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
                    temp_path = temp_file.name
                
                # Extract segment with ffmpeg
                cmd = [
                    'ffmpeg', '-i', audio_path,
                    '-ss', str(start_time),
                    '-t', str(duration),
                    '-acodec', 'pcm_s16le',
                    '-ar', str(target_sr),
                    '-ac', '1',
                    '-y', temp_path,
                    '-loglevel', 'quiet'
                ]
                
                result = subprocess.run(cmd, capture_output=True)
                if result.returncode == 0:
                    audio_array, _ = librosa.load(temp_path, sr=target_sr, mono=True)
                    os.unlink(temp_path)
                    return audio_array.astype(np.float32)
                else:
                    os.unlink(temp_path)
                    return np.array([], dtype=np.float32)
                    
            except Exception as e:
                print(f"Audio loading failed for {audio_path} [{start_time:.2f}-{end_time:.2f}s]: {e}")
                return np.array([], dtype=np.float32)
    
    def transcribe_batch(self, audio_segments, return_timestamps=False):
        """
        Transcribe a batch of audio segments.
        
        Args:
            audio_segments (list): List of audio arrays
            return_timestamps (bool): Whether to return timestamp information
            
        Returns:
            list: List of transcription dictionaries
        """
        if not audio_segments:
            return []
        
        try:
            # Filter out empty or too short segments
            valid_segments = []
            for i, seg in enumerate(audio_segments):
                if len(seg) > 0:
                    # Ensure minimum length (0.1 seconds at 16kHz = 1600 samples)
                    if len(seg) < 1600:
                        # Pad short segments
                        padded = np.pad(seg, (0, 1600 - len(seg)), mode='constant', constant_values=0)
                        valid_segments.append((i, padded))
                    else:
                        valid_segments.append((i, seg))
            
            if not valid_segments:
                return [{"text": ""} for _ in audio_segments]
            
            indices, segments = zip(*valid_segments)
            
            # Process segments with proper parameters
            inputs = self.processor(
                list(segments),
                sampling_rate=16000,
                return_tensors="pt",
                padding=True,
                language="de",  # Specify German language
                task="transcribe"  # Specify transcription task
            )
            
            # Ensure proper dtype and device
            if torch.cuda.is_available() and self.model.device.type != 'cpu':
                # Convert to the model's dtype before moving to device
                inputs = {k: v.to(dtype=self.torch_dtype).to(self.model.device) 
                         for k, v in inputs.items()}
            else:
                # For CPU inference, ensure float32
                inputs = {k: v.to(dtype=torch.float32) for k, v in inputs.items()}
            
            # Generate transcriptions
            with torch.no_grad():
                if return_timestamps:
                    generated_ids = self.model.generate(
                        inputs["input_features"],
                        return_timestamps=True,
                        max_length=448,
                        language="de",
                        task="transcribe"
                    )
                else:
                    generated_ids = self.model.generate(
                        inputs["input_features"],
                        max_length=448,
                        language="de",
                        task="transcribe"
                    )
            
            # Decode results
            transcriptions = self.processor.batch_decode(
                generated_ids, 
                skip_special_tokens=True
            )
            
            # Map results back to original order
            results = [{"text": ""} for _ in audio_segments]
            for i, transcription in zip(indices, transcriptions):
                results[i] = {"text": transcription.strip()}
            
            return results
            
        except Exception as e:
            print(f"Batch transcription error: {e}")
            # Return empty results for this batch
            return [{"text": ""} for _ in audio_segments]
    
    def transcribe_segments(self, segments, progress_callback=None):
        """
        Transcribe a list of segments with batch processing.
        
        Args:
            segments (list): List of segment dictionaries with audio paths and timing
            progress_callback (callable, optional): Progress callback function
            
        Returns:
            list: List of transcription results
        """
        print(f"Transcribing {len(segments)} segments with {self.model_name}")
        
        results = []
        total_batches = (len(segments) + self.batch_size - 1) // self.batch_size
        
        for batch_idx in range(0, len(segments), self.batch_size):
            batch_segments = segments[batch_idx:batch_idx + self.batch_size]
            batch_num = batch_idx // self.batch_size + 1
            
            print(f"   Processing batch {batch_num}/{total_batches}")
            
            # Load audio for this batch
            audio_batch = []
            for segment in batch_segments:
                audio = self.load_audio_segment(
                    segment['audio_path'],
                    segment['start_time'],
                    segment['end_time']
                )
                audio_batch.append(audio)
            
            # Transcribe batch
            batch_results = self.transcribe_batch(audio_batch)
            results.extend(batch_results)
            
            # Progress callback
            if progress_callback:
                progress_callback(batch_num, total_batches)
            
            # Clear memory (GPU or CPU)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            else:
                # For CPU, force garbage collection
                import gc
                gc.collect()
        
        return results


def transcribe_with_both_models(segments, fine_tuned_model_path, batch_size=32, 
                               num_workers=1, progress_callback=None):
    """
    Transcribe segments with both Whisper Large V3 and fine-tuned model.
    
    Args:
        segments (list): List of segment dictionaries
        fine_tuned_model_path (str): Path to fine-tuned model
        batch_size (int): Batch size for processing
        num_workers (int): Number of GPU workers (for multi-GPU setups)
        progress_callback (callable, optional): Progress callback
        
    Returns:
        tuple: (large_v3_results, fine_tuned_results)
    """
    print(f"Starting transcription with both models")
    print(f"   {len(segments)} segments, batch size {batch_size}")
    
    # Initialize transcribers
    large_v3_transcriber = WhisperTranscriber(
        model_path=None,  # Uses openai/whisper-large-v3
        batch_size=batch_size
    )
    
    fine_tuned_transcriber = WhisperTranscriber(
        model_path=fine_tuned_model_path,
        batch_size=batch_size
    )
    
    # Transcribe with Large V3
    print("\nTranscribing with Whisper Large V3...")
    large_v3_results = large_v3_transcriber.transcribe_segments(
        segments, 
        progress_callback
    )
    
    # Clear memory before loading second model
    del large_v3_transcriber
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    else:
        import gc
        gc.collect()
    
    # Transcribe with fine-tuned model
    print("\nTranscribing with fine-tuned model...")
    fine_tuned_results = fine_tuned_transcriber.transcribe_segments(
        segments,
        progress_callback
    )
    
    # Cleanup
    del fine_tuned_transcriber
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    else:
        import gc
        gc.collect()
    
    print("Transcription completed for both models")
    return large_v3_results, fine_tuned_results


def process_segments_batch_parallel(segments_batch, fine_tuned_model_path, batch_size):
    """Worker function for parallel processing of segment batches."""
    return transcribe_with_both_models(
        segments_batch, 
        fine_tuned_model_path, 
        batch_size
    )
