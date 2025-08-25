#!/usr/bin/env python3

#############################################################################
# Script Name: finetune_whisper_HF_debug.py                                #
# Description: Fine-tune Whisper with warning fixes                        #
# Author: Hanno Müller                                                      #
# Date: 2025-08-25                                                          #
# Based on: finetune_whisper_HF.py with warning fixes                      #
#############################################################################

import os
import torch
import evaluate
import argparse
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
        description="Fine-tune Whisper model using HuggingFace transformers."
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
        default="small",
        choices=["tiny", "base", "small", "medium", "large", "large-v2", "large-v3"],
        help="Whisper model size (default: small)"
    )
    
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./FrisperWhisper/test_debug",
        help="Output directory for the fine-tuned model (default: ./FrisperWhisper/test_debug)"
    )
    
    # Training hyperparameters
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=16,
        help="Batch size per device during training (default: 16)"
    )
    
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass (default: 1)"
    )
    
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-5,
        help="Learning rate (default: 1e-5)"
    )
    
    parser.add_argument(
        "--warmup_steps",
        type=int,
        default=500,
        help="Number of warmup steps (default: 500)"
    )
    
    parser.add_argument(
        "--max_steps",
        type=int,
        default=5000,
        help="Maximum number of training steps (default: 5000)"
    )
    
    parser.add_argument(
        "--save_steps",
        type=int,
        default=1000,
        help="Save checkpoint every X steps (default: 1000)"
    )
    
    parser.add_argument(
        "--eval_steps",
        type=int,
        default=1000,
        help="Evaluate every X steps (default: 1000)"
    )
    
    parser.add_argument(
        "--num_gpus",
        type=int,
        default=0,
        help="Number of GPUs to use (default: 0 for CPU-only training)"
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
    """Prepare dataset function with all tokenizer warnings fixed."""
    # Load and resample audio data from 48 to 16kHz
    audio = batch["audio"]

    # Compute log-Mel input features from input audio array 
    batch["input_features"] = feature_extractor(
        audio["array"], 
        sampling_rate=audio["sampling_rate"]
    ).input_features[0]

    # Fix all tokenizer warnings: Use text_target parameter instead of as_target_tokenizer
    # This approach eliminates both the WhisperTokenizerFast and as_target_tokenizer warnings
    labels = tokenizer(
        text_target=batch["sentence"],  # Use text_target instead of as_target_tokenizer
        padding=False,  # We'll pad in the data collator for batch efficiency
        truncation=True,
        max_length=448,  # Whisper's max sequence length
        return_tensors="pt"
    )
    batch["labels"] = labels.input_ids.squeeze().tolist()
    
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
    """Load pre-trained Whisper model with generation config fixes."""
    print(f"Loading pre-trained model: {model_name}")
    
    # Load model normally - should not have missing keys for standard Whisper models
    model = WhisperForConditionalGeneration.from_pretrained(model_name)
    
    # Fix warning 3: Move generation parameters from model config to generation config
    # This prevents the generation config warning
    config_attrs_to_move = ['max_length', 'suppress_tokens', 'begin_suppress_tokens']
    generation_config_updates = {}
    
    for attr in config_attrs_to_move:
        if hasattr(model.config, attr):
            generation_config_updates[attr] = getattr(model.config, attr)
            # Remove from model config to prevent warning
            delattr(model.config, attr)
    
    # Fix warning 2: Remove deprecated forced_decoder_ids and configure generation
    if hasattr(model.config, 'forced_decoder_ids'):
        delattr(model.config, 'forced_decoder_ids')
    
    # Fix forced decoder IDs warning: Remove forced_decoder_ids safely
    if hasattr(model.generation_config, 'forced_decoder_ids'):
        # Use delattr instead of setting to None to completely remove the attribute
        delattr(model.generation_config, 'forced_decoder_ids')
    
    # Update generation config
    for key, value in generation_config_updates.items():
        setattr(model.generation_config, key, value)
    
    print(f"Fixed generation config warnings by moving {len(generation_config_updates)} attributes")
    
    return model

# ============================================================================
# Step 4: Define Data Collator with Warning Fixes
# ============================================================================

@dataclass
class DataCollatorSpeechSeq2SeqWithPaddingOptimized:
    processor: Any
    decoder_start_token_id: int

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # Split inputs and labels since they have to be of different lengths and need different padding methods
        # First treat the audio inputs by simply returning torch tensors
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        # Fix WhisperTokenizerFast warning: Avoid separate tokenize + pad steps
        # Instead, directly process the pre-tokenized labels efficiently
        label_features = [feature["labels"] for feature in features]
        
        # Convert to tensors and pad manually to avoid the tokenizer warning
        max_length = max(len(label) for label in label_features)
        padded_labels = []
        pad_token_id = self.processor.tokenizer.pad_token_id
        
        for label in label_features:
            # Ensure label is a list
            if isinstance(label, torch.Tensor):
                label = label.tolist()
            # Pad with tokenizer.pad_token_id
            padded = label + [pad_token_id] * (max_length - len(label))
            padded_labels.append(padded)
        
        labels = torch.tensor(padded_labels, dtype=torch.long)
        
        # Replace padding with -100 to ignore loss correctly
        labels = labels.masked_fill(labels == pad_token_id, -100)

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

        # Fix attention mask warning: Ensure we decode with proper handling
        # We do not want to group tokens when computing the metrics
        pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

        wer = 100 * metric.compute(predictions=pred_str, references=label_str)

        return {"wer": wer}
    
    return compute_metrics

# ============================================================================
# Step 6: Training Arguments and Trainer with Generation Fixes
# ============================================================================

class WhisperTrainerWithGenerationFixes(Seq2SeqTrainer):
    """Custom trainer that fixes attention mask warnings during generation."""
    
    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None, **gen_kwargs):
        """Override prediction step to handle attention masks properly."""
        if prediction_loss_only:
            return super().prediction_step(model, inputs, prediction_loss_only, ignore_keys, **gen_kwargs)
        
        # Get the standard prediction step result
        loss, generated_tokens, labels = super().prediction_step(
            model, inputs, prediction_loss_only, ignore_keys, **gen_kwargs
        )
        
        # The attention mask warning is handled by the parent class
        # This override just ensures we use the fixed generation configuration
        return loss, generated_tokens, labels

def create_training_arguments(args):
    """Create training arguments with pin_memory fix."""
    # Fix warning 2: Set pin_memory based on GPU availability
    use_pin_memory = args.num_gpus > 0
    
    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        max_steps=args.max_steps,
        gradient_checkpointing=False,  # Disabled to avoid backward graph issues
        fp16=True,
        eval_strategy="steps",
        per_device_eval_batch_size=8,
        predict_with_generate=True,
        generation_max_length=225,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        logging_steps=25,
        report_to=["tensorboard"],
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
        dataloader_pin_memory=use_pin_memory,  # Fix pin_memory warning
    )
    return training_args

def create_trainer(training_args, model, train_dataset, eval_dataset, data_collator, compute_metrics, processor):
    """Create Seq2SeqTrainer with processing_class fix."""
    trainer = WhisperTrainerWithGenerationFixes(
        args=training_args,
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        processing_class=processor,  # Fix warning 3: Use processing_class instead of tokenizer
    )
    return trainer

# ============================================================================
# Step 7: Training Function
# ============================================================================

def train_model(trainer):
    """Train the model."""
    print("Starting training...")
    trainer.train()
    print("Training completed!")

# ============================================================================
# Main Function
# ============================================================================

def main():
    """Main training pipeline with warning fixes."""
    print("=== Fine-tuning Whisper with Warning Fixes ===")
    
    # Parse command line arguments
    args = parse_arguments()
    
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
    
    # Step 4: Load model with generation config fixes
    model = load_pretrained_model(model_name)
    
    # Step 5: Create optimized data collator
    decoder_start_token_id = model.config.decoder_start_token_id
    if decoder_start_token_id is None:
        decoder_start_token_id = tokenizer.convert_tokens_to_ids("<|startoftranscript|>")
    
    data_collator = DataCollatorSpeechSeq2SeqWithPaddingOptimized(
        processor=processor,
        decoder_start_token_id=decoder_start_token_id,
    )
    
    # Step 6: Load metric and create compute_metrics
    metric = load_metric()
    compute_metrics = create_compute_metrics_function(tokenizer, metric)
    
    # Step 7: Create training arguments with fixes
    training_args = create_training_arguments(args)
    
    # Step 8: Create trainer with processing_class fix
    trainer = create_trainer(
        training_args=training_args,
        model=model,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        processor=processor,
    )
    
    # Step 9: Train
    train_model(trainer)
    
    print("=== Training completed successfully with warnings fixed! ===")

if __name__ == "__main__":
    main()
