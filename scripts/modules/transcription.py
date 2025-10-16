############################################################################
# Module: transcription.py                                                #
# Description: Fixed GPU-accelerated transcription with proper padding    #
# Author: Hanno Müller                                                    #
# Date: 2025-09-05 (Fixed version)                                        #
############################################################################

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
import threading
import queue


class WhisperTranscriber:
    """Enhanced Whisper transcriber with proper short segment handling."""
    
    def __init__(self, model_path, device="auto"):
        """
        Initialize the Whisper transcriber.
        
        Args:
            model_path (str): Path to the Whisper model
            device (str): Device to use ('auto', 'cpu', 'cuda')
        """
        self.model_path = model_path
        self.model = None
        self.processor = None
        self.device = device
        self.torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        self._load_model()
    
    def _load_model(self):
        """Load the Whisper model and processor."""
        try:
            print(f"Loading Whisper model: {self.model_path}")
            
            # Load processor (always use the base Whisper processor)
            if "openai/whisper" in self.model_path:
                self.processor = WhisperProcessor.from_pretrained(self.model_path)
            else:
                # For fine-tuned models, use the base processor
                self.processor = WhisperProcessor.from_pretrained("openai/whisper-large-v3")
            
            # Load model
            self.model = WhisperForConditionalGeneration.from_pretrained(
                self.model_path,
                torch_dtype=self.torch_dtype
            )
            
            # Set device
            if self.device == "auto":
                device_map = "cuda" if torch.cuda.is_available() else "cpu"
            else:
                device_map = self.device
            
            self.model = self.model.to(device_map)
            print(f"   Using {'GPU' if torch.cuda.is_available() else 'CPU'} with {self.torch_dtype}")
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
            with sf.SoundFile(audio_path) as audio_file:
                # Get file info
                file_duration = audio_file.frames / audio_file.samplerate
                
                # Clamp times to file bounds
                start_time = max(0, min(start_time, file_duration))
                end_time = max(start_time, min(end_time, file_duration))
                
                if audio_file.samplerate != target_sr:
                    # Resample needed
                    audio_file.seek(int(start_time * audio_file.samplerate))
                    frames_to_read = int((end_time - start_time) * audio_file.samplerate)
                    audio_array = audio_file.read(frames_to_read)
                    audio_array = librosa.resample(
                        audio_array,
                        orig_sr=audio_file.samplerate,
                        target_sr=target_sr
                    )
                else:
                    # No resampling needed
                    audio_file.seek(int(start_time * target_sr))
                    frames_to_read = int((end_time - start_time) * target_sr)
                    audio_array = audio_file.read(frames_to_read)
                
                # Ensure mono
                if len(audio_array.shape) > 1:
                    audio_array = np.mean(audio_array, axis=1)
                
                return audio_array.astype(np.float32)
                
        except Exception as e:
            print(f"Audio loading failed for {audio_path} [{start_time:.2f}-{end_time:.2f}s]: {e}")
            return np.array([], dtype=np.float32)
    
    def _pad_or_truncate_audio(self, audio_array, target_length_seconds=30.0, sample_rate=16000):
        """
        Pad or truncate audio to target length for consistent processing.
        
        Args:
            audio_array (np.ndarray): Input audio
            target_length_seconds (float): Target length in seconds
            sample_rate (int): Sample rate
            
        Returns:
            np.ndarray: Padded/truncated audio
        """
        target_samples = int(target_length_seconds * sample_rate)
        
        if len(audio_array) == 0:
            # Return silence for empty audio
            return np.zeros(target_samples, dtype=np.float32)
        elif len(audio_array) < target_samples:
            # Pad with silence
            padding = target_samples - len(audio_array)
            return np.pad(audio_array, (0, padding), mode='constant', constant_values=0)
        else:
            # Truncate to target length
            return audio_array[:target_samples]
    
    def transcribe_batch(self, audio_segments, return_timestamps=False, min_segment_length=0.1):
        """
        Transcribe a batch of audio segments with proper padding.
        
        Args:
            audio_segments (list): List of audio arrays
            return_timestamps (bool): Whether to return timestamp information
            min_segment_length (float): Minimum segment length in seconds
            
        Returns:
            list: List of transcription dictionaries
        """
        if not audio_segments:
            return []
        
        try:
            results = []
            
            for i, audio in enumerate(audio_segments):
                # Skip very short or empty segments
                if len(audio) < int(min_segment_length * 16000):
                    results.append({"text": ""})
                    continue
                
                # Pad audio to 30 seconds (Whisper's expected input length)
                padded_audio = self._pad_or_truncate_audio(audio, target_length_seconds=30.0)
                
                # Process single segment
                inputs = self.processor(
                    padded_audio,
                    sampling_rate=16000,
                    return_tensors="pt"
                )
                
                # CRITICAL: Convert inputs to match model dtype
                if torch.cuda.is_available() and self.model.device.type != 'cpu':
                    inputs = {k: v.to(self.model.device, dtype=self.torch_dtype if 'input_features' in k else v.dtype) 
                             for k, v in inputs.items()}
                else:
                    inputs = {k: v.to(self.torch_dtype if 'input_features' in k else v.dtype) 
                             for k, v in inputs.items()}
                
                # Generate transcription
                with torch.no_grad():
                    if return_timestamps:
                        generated_ids = self.model.generate(
                            inputs["input_features"],
                            return_timestamps=True,
                            max_length=448
                        )
                    else:
                        generated_ids = self.model.generate(
                            inputs["input_features"],
                            max_length=448
                        )
                
                # Decode
                transcription = self.processor.batch_decode(generated_ids, skip_special_tokens=True)
                results.append({"text": transcription[0] if transcription else ""})
                
                # Clear memory
                del inputs, generated_ids
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            return results
            
        except Exception as e:
            print(f"Batch transcription error: {e}")
            # Return empty results for all segments in case of error
            return [{"text": ""} for _ in audio_segments]
    
    def transcribe_segments(self, segments, batch_size=8, progress_callback=None):
        """
        Transcribe multiple segments with batch processing.
        
        Args:
            segments (list): List of segment dictionaries
            batch_size (int): Number of segments to process at once
            progress_callback (callable): Progress callback function
            
        Returns:
            list: List of transcription results
        """
        results = []
        total_batches = (len(segments) + batch_size - 1) // batch_size
        
        for batch_num in range(total_batches):
            start_idx = batch_num * batch_size
            end_idx = min(start_idx + batch_size, len(segments))
            batch_segments = segments[start_idx:end_idx]
            
            print(f"   Processing batch {batch_num + 1}/{total_batches}")
            
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
                progress_callback(batch_num + 1, total_batches)
            
            # Clear memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            else:
                import gc
                gc.collect()
        
        return results


class MultiGPUWhisperTranscriber:
    """Multi-GPU Whisper transcriber for parallel processing."""
    
    def __init__(self, model_path, num_gpus=1):
        """
        Initialize multi-GPU transcriber.
        
        Args:
            model_path (str): Path to the Whisper model
            num_gpus (int): Number of GPUs to use
        """
        self.model_path = model_path
        self.available_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
        self.num_gpus = min(num_gpus, self.available_gpus) if self.available_gpus > 0 else 1
        self.transcribers = []
        
        print(f"Initializing {self.num_gpus} GPU transcribers for {model_path}")
        print(f"Available GPUs: {self.available_gpus}")
        
        # Create device-specific transcribers
        for gpu_id in range(self.num_gpus):
            device = f"cuda:{gpu_id}" if torch.cuda.is_available() and gpu_id < self.available_gpus else "cpu"
            try:
                transcriber = self._create_device_specific_transcriber(model_path, device, gpu_id)
                self.transcribers.append((gpu_id, device, transcriber))
                print(f"  Transcriber {gpu_id} initialized on {device}")
            except Exception as e:
                print(f"  Failed to initialize transcriber on {device}: {e}")
                continue
        
        if not self.transcribers:
            raise RuntimeError("Failed to initialize any transcribers")
    
    def _create_device_specific_transcriber(self, model_path, device, gpu_id):
        """Create a transcriber for a specific device."""
        
        class DeviceSpecificTranscriber:
            def __init__(self, model_path, device, gpu_id):
                self.model_path = model_path
                self.device = device
                self.gpu_id = gpu_id
                self.torch_dtype = torch.float16 if "cuda" in device else torch.float32
                self._load_model()
            
            def _load_model(self):
                """Load the Whisper model and processor."""
                # Load processor
                if "openai/whisper" in self.model_path:
                    self.processor = WhisperProcessor.from_pretrained(self.model_path)
                else:
                    # For fine-tuned models, use the base processor
                    self.processor = WhisperProcessor.from_pretrained("openai/whisper-large-v3")
                
                # Load model
                self.model = WhisperForConditionalGeneration.from_pretrained(
                    self.model_path,
                    torch_dtype=self.torch_dtype
                )
                
                # Move to specific device
                self.model = self.model.to(self.device)
                
            def load_audio_segment(self, audio_path, start_time, end_time, target_sr=16000):
                """Load a specific audio segment from file."""
                try:
                    with sf.SoundFile(audio_path) as audio_file:
                        file_duration = audio_file.frames / audio_file.samplerate
                        
                        start_time = max(0, min(start_time, file_duration))
                        end_time = max(start_time, min(end_time, file_duration))
                        
                        if audio_file.samplerate != target_sr:
                            audio_file.seek(int(start_time * audio_file.samplerate))
                            frames_to_read = int((end_time - start_time) * audio_file.samplerate)
                            audio_array = audio_file.read(frames_to_read)
                            audio_array = librosa.resample(
                                audio_array,
                                orig_sr=audio_file.samplerate,
                                target_sr=target_sr
                            )
                        else:
                            audio_file.seek(int(start_time * target_sr))
                            frames_to_read = int((end_time - start_time) * target_sr)
                            audio_array = audio_file.read(frames_to_read)
                        
                        # Ensure mono
                        if len(audio_array.shape) > 1:
                            audio_array = np.mean(audio_array, axis=1)
                        
                        return audio_array.astype(np.float32)
                        
                except Exception as e:
                    print(f"Audio loading failed for {audio_path} [{start_time:.2f}-{end_time:.2f}s]: {e}")
                    return np.array([], dtype=np.float32)
            
            def _pad_or_truncate_audio(self, audio_array, target_length_seconds=30.0, sample_rate=16000):
                """Pad or truncate audio to target length."""
                target_samples = int(target_length_seconds * sample_rate)
                
                if len(audio_array) == 0:
                    return np.zeros(target_samples, dtype=np.float32)
                elif len(audio_array) < target_samples:
                    padding = target_samples - len(audio_array)
                    return np.pad(audio_array, (0, padding), mode='constant', constant_values=0)
                else:
                    return audio_array[:target_samples]
            
            def transcribe_batch(self, audio_segments, min_segment_length=0.1):
                """Transcribe a batch of audio segments."""
                if not audio_segments:
                    return []
                
                try:
                    results = []
                    
                    for audio in audio_segments:
                        # Skip very short or empty segments
                        if len(audio) < int(min_segment_length * 16000):
                            results.append({"text": ""})
                            continue
                        
                        # Pad audio to 30 seconds
                        padded_audio = self._pad_or_truncate_audio(audio, target_length_seconds=30.0)
                        
                        # Process single segment
                        inputs = self.processor(
                            padded_audio,
                            sampling_rate=16000,
                            return_tensors="pt"
                        )
                        
                        # Move inputs to device with correct dtype
                        inputs = {k: v.to(self.device, dtype=self.torch_dtype if 'input_features' in k else v.dtype) 
                                 for k, v in inputs.items()}
                        
                        # Generate transcription
                        with torch.no_grad():
                            generated_ids = self.model.generate(
                                inputs["input_features"],
                                max_length=448
                            )
                        
                        # Decode
                        transcription = self.processor.batch_decode(generated_ids, skip_special_tokens=True)
                        results.append({"text": transcription[0] if transcription else ""})
                        
                        # Clear memory
                        del inputs, generated_ids
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                    
                    return results
                    
                except Exception as e:
                    print(f"Batch transcription error on {self.device}: {e}")
                    return [{"text": ""} for _ in audio_segments]
            
            def transcribe_segments(self, segments, batch_size=8, progress_callback=None):
                """Transcribe multiple segments with batch processing."""
                results = []
                total_batches = (len(segments) + batch_size - 1) // batch_size
                
                for batch_num in range(total_batches):
                    start_idx = batch_num * batch_size
                    end_idx = min(start_idx + batch_size, len(segments))
                    batch_segments = segments[start_idx:end_idx]
                    
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
                        progress_callback(batch_num + 1, total_batches)
                    
                    # Clear memory
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                
                return results
        
        return DeviceSpecificTranscriber(model_path, device, gpu_id)
    
    def transcribe_segments_parallel(self, segments, batch_size=8, progress_callback=None):
        """
        Transcribe segments in parallel across multiple GPUs.
        
        Args:
            segments (list): List of segment dictionaries
            batch_size (int): Batch size per GPU
            progress_callback (callable): Progress callback function
        
        Returns:
            list: Transcription results
        """
        if not segments:
            return []
        
        print(f"Distributing {len(segments)} segments across {len(self.transcribers)} GPU(s)")
        
        # Split segments across GPUs
        segments_per_gpu = []
        segments_per_worker = len(segments) // len(self.transcribers)
        remainder = len(segments) % len(self.transcribers)
        
        start_idx = 0
        for i in range(len(self.transcribers)):
            # Calculate segments for this GPU (distribute remainder evenly)
            segments_for_this_gpu = segments_per_worker + (1 if i < remainder else 0)
            end_idx = start_idx + segments_for_this_gpu
            
            gpu_segments = segments[start_idx:end_idx]
            segments_per_gpu.append(gpu_segments)
            gpu_id, device, _ = self.transcribers[i]
            print(f"  GPU {gpu_id} ({device}): {len(gpu_segments)} segments")
            start_idx = end_idx
        
        # Process segments in parallel using threading
        results_queue = queue.Queue()
        threads = []
        
        def process_gpu_segments(gpu_info, gpu_segments):
            """Process segments on a specific GPU."""
            gpu_id, device, transcriber = gpu_info
            try:
                gpu_results = transcriber.transcribe_segments(
                    gpu_segments, 
                    batch_size=batch_size,
                    progress_callback=lambda current, total: print(f"    GPU {gpu_id}: {current}/{total} batches")
                )
                results_queue.put((gpu_id, gpu_results))
            except Exception as e:
                print(f"Error on GPU {gpu_id}: {e}")
                results_queue.put((gpu_id, [{"text": ""} for _ in gpu_segments]))
        
        # Start processing threads
        for i, (gpu_info, gpu_segments) in enumerate(zip(self.transcribers, segments_per_gpu)):
            if gpu_segments:  # Only start thread if there are segments to process
                thread = threading.Thread(
                    target=process_gpu_segments,
                    args=(gpu_info, gpu_segments)
                )
                thread.start()
                threads.append(thread)
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # Collect results in correct order
        gpu_results = {}
        while not results_queue.empty():
            gpu_id, results = results_queue.get()
            gpu_results[gpu_id] = results
        
        # Combine results in original order
        combined_results = []
        for i, (gpu_id, _, _) in enumerate(self.transcribers):
            if gpu_id in gpu_results:
                combined_results.extend(gpu_results[gpu_id])
        
        print(f"Multi-GPU transcription completed: {len(combined_results)} results")
        return combined_results


def transcribe_with_both_models_multi_gpu(segments, fine_tuned_model_path, 
                                         batch_size=8, num_gpus=1, progress_callback=None):
    """
    Transcribe segments with both standard and fine-tuned Whisper models using multiple GPUs.
    
    Args:
        segments (list): List of segment dictionaries
        fine_tuned_model_path (str): Path to fine-tuned model
        batch_size (int): Batch size for processing
        num_gpus (int): Number of GPUs to use
        progress_callback (callable): Progress callback function
    
    Returns:
        tuple: (standard_results, fine_tuned_results)
    """
    if not segments:
        return [], []
    
    print("Starting multi-GPU transcription with both models")
    print(f"   {len(segments)} segments, batch size {batch_size}, {num_gpus} GPUs")
    
    # Initialize multi-GPU transcribers
    print("\nInitializing transcribers...")
    standard_transcriber = MultiGPUWhisperTranscriber("openai/whisper-large-v3", num_gpus)
    
    # Clear GPU memory between model loads
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    
    fine_tuned_transcriber = MultiGPUWhisperTranscriber(fine_tuned_model_path, num_gpus)
    
    # Transcribe with standard model
    print("\nTranscribing with Whisper Large V3...")
    print(f"Transcribing {len(segments)} segments with openai/whisper-large-v3")
    standard_results = standard_transcriber.transcribe_segments_parallel(
        segments, batch_size=batch_size, progress_callback=progress_callback
    )
    
    # Clear GPU memory between models
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    
    # Transcribe with fine-tuned model
    print(f"\nTranscribing with fine-tuned model...")
    print(f"Transcribing {len(segments)} segments with {fine_tuned_model_path}")
    fine_tuned_results = fine_tuned_transcriber.transcribe_segments_parallel(
        segments, batch_size=batch_size, progress_callback=progress_callback
    )
    
    print("Multi-GPU transcription completed for both models")
    
    return standard_results, fine_tuned_results


def transcribe_with_multiple_models(segments, model_paths, batch_size=8, 
                                   num_workers=1, progress_callback=None, num_gpus=1):
    """
    Transcribe segments with multiple Whisper models.
    
    Args:
        segments (list): List of segment dictionaries
        model_paths (list): List of model paths to use for transcription
        batch_size (int): Batch size for processing
        num_workers (int): Number of worker processes (not used in this implementation)
        progress_callback (callable): Progress callback function
        num_gpus (int): Number of GPUs to use (default: 1 for single-GPU mode)
    
    Returns:
        tuple: Tuple of result lists, one for each model
    """
    if not segments:
        return tuple([] for _ in model_paths)
    
    print(f"Starting transcription with {len(model_paths)} models")
    print(f"   {len(segments)} segments, batch size {batch_size}, {num_gpus} GPUs")
    
    all_results = []
    
    # Use multi-GPU transcription if more than 1 GPU requested
    if num_gpus > 1:
        for i, model_path in enumerate(model_paths):
            print(f"\n[Model {i+1}/{len(model_paths)}] Transcribing with {model_path}...")
            transcriber = MultiGPUWhisperTranscriber(model_path, num_gpus)
            results = transcriber.transcribe_segments_parallel(
                segments, batch_size=batch_size, progress_callback=progress_callback
            )
            all_results.append(results)
            
            # Clear GPU memory between models
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
    else:
        # Single-GPU transcription
        for i, model_path in enumerate(model_paths):
            print(f"\n[Model {i+1}/{len(model_paths)}] Transcribing with {model_path}...")
            transcriber = WhisperTranscriber(model_path)
            results = transcriber.transcribe_segments(
                segments, batch_size=batch_size, progress_callback=progress_callback
            )
            all_results.append(results)
            
            # Clear GPU memory between models
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    print(f"Transcription completed for all {len(model_paths)} models")
    
    return tuple(all_results)


def transcribe_with_both_models(segments, fine_tuned_model_path, batch_size=8, 
                               num_workers=1, progress_callback=None, num_gpus=1):
    """
    Transcribe segments with both standard and fine-tuned Whisper models.
    (Wrapper around transcribe_with_multiple_models for backward compatibility)
    
    Args:
        segments (list): List of segment dictionaries
        fine_tuned_model_path (str): Path to fine-tuned model
        batch_size (int): Batch size for processing
        num_workers (int): Number of worker processes (not used in this implementation)
        progress_callback (callable): Progress callback function
        num_gpus (int): Number of GPUs to use (default: 1 for single-GPU mode)
    
    Returns:
        tuple: (standard_results, fine_tuned_results)
    """
    model_paths = ["openai/whisper-large-v3", fine_tuned_model_path]
    return transcribe_with_multiple_models(
        segments, model_paths, batch_size, num_workers, progress_callback, num_gpus
    )
