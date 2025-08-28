#!/usr/bin/env python3

#############################################################################
# Script Name: finetune_whisper_HF_gpu_stable.py                           #
# Description: Fine-tune Whisper with improved stability measures          #
# Author: Hanno Müller                                                      #
# Date: 2025-08-26                                                          #
# Based on: finetune_whisper_HF_gpu.py with stability improvements         #
#############################################################################

import os
import torch
import evaluate
import argparse
import numpy as np
from dataclasses import dataclass
from typing import Any, Dict, List, Union

from datasets import load_from_disk, DatasetDict
from transformers import (
    WhisperFeatureExtractor,
    WhisperTokenizer,
    WhisperProcessor,
    WhisperForConditionalGeneration,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer
)

# ============================================================================
# Argument Parsing
# ============================================================================

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Fine-tune Whisper model with improved stability measures."
    )
    
    parser.add_argument(
        "--dataset_path",
        type=str,
        required=True,
        help="Path to the dataset folder (e.g., data/LangAgeDataSetSubsample1)"
    )
    
    parser.add_argument(
        "--model_size",
        type=str,
        default="medium",
        choices=["tiny", "base", "small", "medium", "large", "large-v2", "large-v3"],
        help="Whisper model size (default: medium)"
    )
    
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./FrisperWhisper/test-gpu",
        help="Output directory for the fine-tuned model (default: ./FrisperWhisper/test-gpu)"
    )
    
    # Core training hyperparameters (improved defaults)
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=4,  # Reduced for stability
        help="Batch size per device during training (default: 4)"
    )
    
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=4,  # Increased for better effective batch size
        help="Number of updates steps to accumulate before performing a backward/update pass (default: 4)"
    )
    
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=3e-6,  # Even more conservative for stability
        help="Learning rate (default: 3e-6)"
    )
    
    parser.add_argument(
        "--warmup_steps",
        type=int,
        default=1500,  # Extended warmup for stability
        help="Number of warmup steps (default: 1500)"
    )
    
    parser.add_argument(
        "--max_steps",
        type=int,
        default=40000,  # More training for better convergence
        help="Maximum number of training steps (default: 12000)"
    )
    
    parser.add_argument(
        "--save_steps",
        type=int,
        default=1000,  # More frequent saves
        help="Save checkpoint every X steps (default: 500)"
    )
    
    parser.add_argument(
        "--eval_steps",
        type=int,
        default=1000,  # More frequent evaluation
        help="Evaluate every X steps (default: 500)"
    )
    
    # Stability measures
    parser.add_argument(
        "--max_grad_norm",
        type=float,
        default=0.5,  # More aggressive clipping for stability
        help="Maximum gradient norm for clipping (default: 0.5, set to 0 to disable)"
    )
    
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.02,  # Slightly higher regularization
        help="Weight decay (L2 regularization) coefficient (default: 0.02)"
    )
    
    parser.add_argument(
        "--lr_scheduler_type",
        type=str,
        default="cosine",
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
        help="Learning rate scheduler type (default: cosine)"
    )
    

    
    parser.add_argument(
        "--early_stopping_patience",
        type=int,
        default=0,
        help="Early stopping patience (steps). Set to 0 to disable early stopping (default: 0)"
    )
    
    parser.add_argument(
        "--early_stopping_threshold",
        type=float,
        default=0.01,
        help="Early stopping threshold for metric improvement (default: 0.01)"
    )
    
    # Advanced training options
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=8,
        help="Number of data loader workers (default: 8)"
    )
    
    parser.add_argument(
        "--fp16",
        action="store_true",
        default=False,  # Disable by default since BF16 is preferred
        help="Use 16-bit floating point precision (default: False, use BF16 instead)"
    )
    
    parser.add_argument(
        "--bf16",
        action="store_true",
        default=True,  # Enable by default for H100
        help="Use bfloat16 precision (recommended for H100, default: True)"
    )
    
    parser.add_argument(
        "--num_gpus",
        type=int,
        default=0,
        help="Number of GPUs to use (default: 0 for CPU-only training)"
    )
    
    # Logging and monitoring
    parser.add_argument(
        "--logging_steps",
        type=int,
        default=50,
        help="Log training metrics every X steps (default: 50)"
    )
    
    parser.add_argument(
        "--report_to",
        type=str,
        nargs="+",
        default=["tensorboard"],
        choices=["tensorboard", "wandb", "none"],
        help="Logging platforms (default: tensorboard)"
    )
    
    # Checkpoint resumption
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="Path to a checkpoint to resume training from. Can be a specific checkpoint directory or 'True' to auto-resume from the latest checkpoint in output_dir (default: None)"
    )
    
    return parser.parse_args()

# ============================================================================
# Step 1: Load Dataset
# ============================================================================

def load_dataset_from_disk(dataset_path):
    """Load dataset from disk."""
    print(f"Loading dataset from: {dataset_path}")
    
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")
    
    dataset = load_from_disk(dataset_path)
    
    print(f"Dataset loaded successfully!")
    #print(f"Dataset structure: {dataset}")
    
    return dataset

# ============================================================================
# Step 2: Prepare Feature Extractor, Tokenizer and Data
# ============================================================================

def load_feature_extractor(model_name):
    """Load WhisperFeatureExtractor."""
    print(f"Loading feature extractor for {model_name}...")
    feature_extractor = WhisperFeatureExtractor.from_pretrained(model_name)
    return feature_extractor

def load_tokenizer(model_name):
    """Load WhisperTokenizer."""
    print(f"Loading tokenizer for {model_name}...")
    tokenizer = WhisperTokenizer.from_pretrained(
        model_name, 
        language="French", 
        task="transcribe"
    )
    return tokenizer

def create_processor(model_name):
    """Create WhisperProcessor."""
    print(f"Creating processor for {model_name}...")
    processor = WhisperProcessor.from_pretrained(
        model_name, 
        language="French", 
        task="transcribe"
    )
    
    return processor

def prepare_dataset_function(batch, feature_extractor, tokenizer):
    """Prepare dataset function as shown in HF tutorial."""
    # Load and resample audio data from 48 to 16kHz
    audio = batch["audio"]

    # Convert audio array to numpy array if it's a list (fix for LangAge dataset)
    audio_array = audio["array"]
    if isinstance(audio_array, list):
        audio_array = np.array(audio_array, dtype=np.float32)

    # Compute log-Mel input features from input audio array 
    batch["input_features"] = feature_extractor(
        audio_array, 
        sampling_rate=audio["sampling_rate"]
    ).input_features[0]

    # Encode target text to label ids 
    batch["labels"] = tokenizer(batch["sentence"]).input_ids
    
    return batch

def prepare_data(dataset, feature_extractor, tokenizer):
    """Prepare the dataset for training."""
    print("Preparing data...")
    
    # Note: Audio is already at 16kHz sampling rate, so no resampling needed
    
    # Apply preparation function
    dataset = dataset.map(
        lambda batch: prepare_dataset_function(batch, feature_extractor, tokenizer),
        remove_columns=dataset.column_names["train"], 
        num_proc=4
    )
    
    return dataset

# ============================================================================
# Step 3: Load Pre-trained Model
# ============================================================================

def load_pretrained_model(model_name):
    """Load pre-trained Whisper model."""
    print(f"Loading pre-trained model: {model_name}")
    
    # Load model normally - should not have missing keys for standard Whisper models
    model = WhisperForConditionalGeneration.from_pretrained(model_name)
    
    # Disable cache to avoid issues with DataParallel and Whisper large-v3
    model.config.use_cache = False
    
    # Create a clean generation config to fix warning 3
    # Remove generation parameters from model config and set them in generation config
    generation_config_params = {}
    if hasattr(model.config, 'max_length'):
        generation_config_params['max_length'] = model.config.max_length
        delattr(model.config, 'max_length')
    if hasattr(model.config, 'suppress_tokens'):
        generation_config_params['suppress_tokens'] = model.config.suppress_tokens
        delattr(model.config, 'suppress_tokens')
    if hasattr(model.config, 'begin_suppress_tokens'):
        generation_config_params['begin_suppress_tokens'] = model.config.begin_suppress_tokens
        delattr(model.config, 'begin_suppress_tokens')
    if hasattr(model.config, 'forced_decoder_ids'):
        # Remove deprecated forced_decoder_ids
        delattr(model.config, 'forced_decoder_ids')
    
    # Update generation config with the parameters
    for key, value in generation_config_params.items():
        setattr(model.generation_config, key, value)
    
    return model

# ============================================================================
# Step 4: Define Data Collator
# ============================================================================

@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any
    decoder_start_token_id: int

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # Split inputs and labels since they have to be of different lengths and need different padding methods
        # First treat the audio inputs by simply returning torch tensors
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        # Get the tokenized label sequences
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        # Pad the labels to max length
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # Replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        # If bos token is appended in previous tokenization step,
        # cut bos token here as it's append later anyways
        if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels

        return batch

# ============================================================================
# Step 5: Evaluation Metrics
# ============================================================================

def load_metric():
    """Load WER metric."""
    print("Loading WER metric...")
    metric = evaluate.load("wer")
    return metric

def create_compute_metrics_function(tokenizer, metric):
    """Create compute_metrics function."""
    def compute_metrics(pred):
        pred_ids = pred.predictions
        label_ids = pred.label_ids

        # Replace -100 with the pad_token_id
        label_ids[label_ids == -100] = tokenizer.pad_token_id

        # We do not want to group tokens when computing the metrics
        pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

        wer = 100 * metric.compute(predictions=pred_str, references=label_str)

        return {"wer": wer}
    
    return compute_metrics

# ============================================================================
# Step 6: Training Arguments and Trainer
# ============================================================================

def create_training_arguments(args):
    """Create training arguments based on command line arguments."""
    # Set pin_memory based on GPU availability
    use_pin_memory = args.num_gpus > 0
    
    # Handle early stopping
    early_stopping_kwargs = {}
    if args.early_stopping_patience > 0:
        early_stopping_kwargs.update({
            "load_best_model_at_end": True,
            "metric_for_best_model": "wer",
            "greater_is_better": False,
        })
    
    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        max_steps=args.max_steps,
        
        # Stability measures
        max_grad_norm=args.max_grad_norm if args.max_grad_norm > 0 else None,
        weight_decay=args.weight_decay,
        lr_scheduler_type=args.lr_scheduler_type,
        
        # Memory and performance
        fp16=args.fp16 and not args.bf16,  # Don't use both fp16 and bf16
        bf16=args.bf16,
        
        # Evaluation and logging
        eval_strategy="steps",
        per_device_eval_batch_size=2,  # Keep small for memory efficiency
        predict_with_generate=True,
        generation_max_length=225,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        logging_steps=args.logging_steps,
        report_to=args.report_to,
        
        # Data loading
        dataloader_pin_memory=use_pin_memory,
        dataloader_num_workers=args.dataloader_num_workers,
        remove_unused_columns=False,  # Keep columns to avoid issues
        group_by_length=False,  # Disable to avoid DataParallel issues
        
        # Multi-GPU optimizations
        ddp_find_unused_parameters=False,  # For better performance
        
        # Apply early stopping settings
        **early_stopping_kwargs,
    )
    
    return training_args

def create_trainer(training_args, model, train_dataset, eval_dataset, data_collator, compute_metrics, processor, args):
    """Create Seq2SeqTrainer with optional early stopping."""
    from transformers.trainer_callback import EarlyStoppingCallback
    
    # Setup callbacks
    callbacks = []
    if args.early_stopping_patience > 0:
        early_stopping_callback = EarlyStoppingCallback(
            early_stopping_patience=args.early_stopping_patience,
            early_stopping_threshold=args.early_stopping_threshold
        )
        callbacks.append(early_stopping_callback)
        print(f"Early stopping enabled: patience={args.early_stopping_patience}, threshold={args.early_stopping_threshold}")
    
    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        processing_class=processor,  # Use processing_class instead of tokenizer
        callbacks=callbacks,
    )
    return trainer

# ============================================================================
# Checkpoint Management
# ============================================================================

def find_latest_checkpoint(output_dir):
    """Find the latest checkpoint in the output directory."""
    import glob
    import re
    
    if not os.path.exists(output_dir):
        return None
    
    # Look for checkpoint directories
    checkpoint_pattern = os.path.join(output_dir, "checkpoint-*")
    checkpoints = glob.glob(checkpoint_pattern)
    
    if not checkpoints:
        return None
    
    # Extract step numbers and find the latest
    checkpoint_steps = []
    for checkpoint in checkpoints:
        match = re.search(r"checkpoint-(\d+)", checkpoint)
        if match:
            step = int(match.group(1))
            checkpoint_steps.append((step, checkpoint))
    
    if not checkpoint_steps:
        return None
    
    # Return the checkpoint with the highest step number
    latest_checkpoint = max(checkpoint_steps, key=lambda x: x[0])[1]
    return latest_checkpoint

def resolve_checkpoint_path(resume_from_checkpoint, output_dir):
    """Resolve the checkpoint path for resumption."""
    if resume_from_checkpoint is None:
        return None
    
    # Handle different cases
    if resume_from_checkpoint.lower() == "true":
        # Auto-detect latest checkpoint in output_dir
        checkpoint_path = find_latest_checkpoint(output_dir)
        if checkpoint_path:
            print(f"Auto-detected latest checkpoint: {checkpoint_path}")
        else:
            print(f"No checkpoints found in {output_dir}, starting fresh training")
        return checkpoint_path
    
    elif os.path.isabs(resume_from_checkpoint):
        # Absolute path provided
        if os.path.exists(resume_from_checkpoint):
            print(f"Resuming from checkpoint: {resume_from_checkpoint}")
            return resume_from_checkpoint
        else:
            print(f"Warning: Checkpoint path does not exist: {resume_from_checkpoint}")
            print("Starting fresh training")
            return None
    
    else:
        # Relative path - assume it's relative to output_dir
        checkpoint_path = os.path.join(output_dir, resume_from_checkpoint)
        if os.path.exists(checkpoint_path):
            print(f"Resuming from checkpoint: {checkpoint_path}")
            return checkpoint_path
        else:
            print(f"Warning: Checkpoint path does not exist: {checkpoint_path}")
            print("Starting fresh training")
            return None

# ============================================================================
# Step 7: Training Function
# ============================================================================

def train_model(trainer, resume_from_checkpoint=None):
    """Train the model with optional checkpoint resumption."""
    if resume_from_checkpoint:
        print(f"Resuming training from checkpoint: {resume_from_checkpoint}")
    else:
        print("Starting training from scratch...")
    
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    print("Training completed!")

# ============================================================================
# Configuration Summary
# ============================================================================

def print_training_config(args, resume_checkpoint_path=None):
    """Print training configuration summary."""
    print("\n" + "="*60)
    print("TRAINING CONFIGURATION SUMMARY")
    print("="*60)
    print(f"Model size: {args.model_size}")
    print(f"Dataset: {args.dataset_path}")
    print(f"Output: {args.output_dir}")
    if resume_checkpoint_path:
        print(f"Resume from: {resume_checkpoint_path}")
    elif args.resume_from_checkpoint:
        print(f"Resume setting: {args.resume_from_checkpoint}")
    print("-"*60)
    print("Core Training Parameters:")
    print(f"  Learning rate: {args.learning_rate}")
    print(f"  Batch size per device: {args.per_device_train_batch_size}")
    print(f"  Gradient accumulation: {args.gradient_accumulation_steps}")
    print(f"  Effective batch size: {args.per_device_train_batch_size * args.gradient_accumulation_steps * max(1, args.num_gpus)}")
    print(f"  Max steps: {args.max_steps}")
    print(f"  Warmup steps: {args.warmup_steps}")
    print(f"  LR scheduler: {args.lr_scheduler_type}")
    print("-"*60)
    print("Stability Measures:")
    print(f"  Gradient clipping: {args.max_grad_norm if args.max_grad_norm > 0 else 'Disabled'}")
    print(f"  Weight decay: {args.weight_decay}")
    print(f"  Early stopping: {'Enabled' if args.early_stopping_patience > 0 else 'Disabled'}")
    if args.early_stopping_patience > 0:
        print(f"    Patience: {args.early_stopping_patience} steps")
        print(f"    Threshold: {args.early_stopping_threshold}")
    print("-"*60)
    print("Hardware:")
    print(f"  GPUs: {args.num_gpus}")
    print(f"  Precision: {'BF16' if args.bf16 else 'FP16' if args.fp16 else 'FP32'}")
    print(f"  Data workers: {args.dataloader_num_workers}")
    print("="*60)

# ============================================================================
# Main Function
# ============================================================================

def main():
    """Main training pipeline with stability improvements."""
    print("=== Fine-tuning Whisper with Stability Measures ===")
    
    # Parse command line arguments
    args = parse_arguments()
    
    # Resolve checkpoint path for resumption
    resume_checkpoint_path = resolve_checkpoint_path(args.resume_from_checkpoint, args.output_dir)
    
    # Print configuration summary
    print_training_config(args, resume_checkpoint_path)
    
    # Setup GPU environment
    if args.num_gpus > 0:
        if not torch.cuda.is_available():
            print("Warning: CUDA not available, falling back to CPU")
            args.num_gpus = 0
        else:
            available_gpus = torch.cuda.device_count()
            if args.num_gpus > available_gpus:
                print(f"Warning: Requested {args.num_gpus} GPUs, but only {available_gpus} available")
                args.num_gpus = available_gpus
            print(f"Using {args.num_gpus} GPU(s) for training")
            
            # Clear GPU memory before starting
            torch.cuda.empty_cache()
            
            # Set memory fraction to avoid full GPU usage
            for i in range(args.num_gpus):
                torch.cuda.set_per_process_memory_fraction(0.95, device=i)
            
            # Set CUDA_VISIBLE_DEVICES to limit GPU usage
            if args.num_gpus < available_gpus:
                import os
                os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(i) for i in range(args.num_gpus))
    
    # Construct model name
    model_name = f"openai/whisper-{args.model_size}"
    print(f"Using model: {model_name}")
    
    # Step 1: Load dataset
    dataset = load_dataset_from_disk(args.dataset_path)
    
    # Step 2: Load components
    feature_extractor = load_feature_extractor(model_name)
    tokenizer = load_tokenizer(model_name)
    processor = create_processor(model_name)
    
    # Step 3: Prepare data
    dataset = prepare_data(dataset, feature_extractor, tokenizer)
    
    # Step 4: Load model
    model = load_pretrained_model(model_name)
    
    # Step 5: Create data collator
    decoder_start_token_id = model.config.decoder_start_token_id
    if decoder_start_token_id is None:
        decoder_start_token_id = tokenizer.convert_tokens_to_ids("<|startoftranscript|>")
    
    data_collator = DataCollatorSpeechSeq2SeqWithPadding(
        processor=processor,
        decoder_start_token_id=decoder_start_token_id,
    )
    
    # Step 6: Load metric and create compute_metrics
    metric = load_metric()
    compute_metrics = create_compute_metrics_function(tokenizer, metric)
    
    # Step 7: Create training arguments
    training_args = create_training_arguments(args)
    
    # Step 8: Create trainer
    trainer = create_trainer(
        training_args=training_args,
        model=model,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        processor=processor,
        args=args,
    )
    
    # Step 9: Train
    train_model(trainer, resume_checkpoint_path)
    
    print("=== Training completed successfully! ===")
    print(f"Model saved to: {args.output_dir}")
    print(f"TensorBoard logs: {args.output_dir}/runs/")

if __name__ == "__main__":
    main()
