#############################################################################
# Script Name: finetune_whisper.py                                          #
# Description: Fine-tune Whisper model on LangAge dataset for French ASR   #
# Author: Hanno Müller                                                      #
# Date: 2025-07-24                                                          #
#############################################################################

### Required Libraries ######################################################
import os
import argparse
import torch
import multiprocessing
from dataclasses import dataclass
from typing import Any, Dict, List, Union

from transformers import (
    WhisperFeatureExtractor,
    WhisperTokenizer, 
    WhisperProcessor,
    WhisperForConditionalGeneration,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer
)
from datasets import DatasetDict, load_from_disk
import evaluate


### Class Definitions ########################################################

class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any
    decoder_start_token_id: int

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lengths and need different padding methods
        # first treat the audio inputs by simply returning torch tensors
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        # get the tokenized label sequences
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        # pad the labels to max length
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        # if bos token is appended in previous tokenization step,
        # cut bos token here as it's append later anyways
        if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels

        return batch


### Function Definitions #####################################################

def configure_hardware(args):
    """
    Configure CPU and GPU resources based on user specifications.
    """
    print("=== Hardware Configuration ===")
    
    # Configure CPUs
    available_cpus = multiprocessing.cpu_count()
    print(f"Available CPUs: {available_cpus}")
    
    if args.num_cpus is not None:
        if args.num_cpus > available_cpus:
            print(f"Warning: Requested {args.num_cpus} CPUs, but only {available_cpus} available. Using {available_cpus}.")
            num_cpus = available_cpus
        elif args.num_cpus <= 0:
            print("Error: Number of CPUs must be positive.")
            raise ValueError("Invalid number of CPUs specified")
        else:
            num_cpus = args.num_cpus
    else:
        # Auto-detect mode: use at most half of available CPUs
        num_cpus = max(1, available_cpus // 2)
        print(f"Auto-detect mode: using {num_cpus} CPUs (half of available)")
    
    print(f"Using {num_cpus} CPU cores for data preprocessing")
    
    # Configure GPUs
    available_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    print(f"Available GPUs: {available_gpus}")
    
    if available_gpus > 0:
        for i in range(available_gpus):
            gpu_name = torch.cuda.get_device_name(i)
            gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
            print(f"  GPU {i}: {gpu_name} ({gpu_memory:.1f} GB)")
    
    if args.num_gpus is not None:
        if args.num_gpus > available_gpus:
            print(f"Warning: Requested {args.num_gpus} GPUs, but only {available_gpus} available. Using {available_gpus}.")
            num_gpus = available_gpus
        elif args.num_gpus < 0:
            print("Error: Number of GPUs cannot be negative.")
            raise ValueError("Invalid number of GPUs specified")
        else:
            num_gpus = args.num_gpus
    else:
        # Auto-detect mode: use 0 GPUs (CPU-only training)
        num_gpus = 0
        print("Auto-detect mode: using 0 GPUs (CPU-only training)")
    
    if num_gpus > 0:
        print(f"Using {num_gpus} GPUs for training")
        # Enable GPU optimizations
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        
        # Set visible devices if using subset of GPUs
        if num_gpus < available_gpus:
            gpu_ids = ','.join(str(i) for i in range(num_gpus))
            os.environ['CUDA_VISIBLE_DEVICES'] = gpu_ids
            print(f"Set CUDA_VISIBLE_DEVICES to: {gpu_ids}")
    else:
        print("Using CPU-only training")
    
    # Configure data loading workers
    if args.dataloader_workers is not None:
        dataloader_workers = args.dataloader_workers
        print(f"Using {dataloader_workers} data loader workers (user specified)")
    else:
        # Calculate optimal workers based on resources
        if num_gpus > 0:
            # For GPU training: balance between CPU cores and GPUs
            # Rule of thumb: 2-4 workers per GPU, but don't exceed CPU cores
            optimal_workers = min(num_cpus, num_gpus * 4)
            dataloader_workers = max(1, optimal_workers)
        else:
            # For CPU training: use fewer workers to avoid overhead
            dataloader_workers = max(1, num_cpus // 4)
        print(f"Using {dataloader_workers} data loader workers (auto-calculated)")
    
    print("=" * 35)
    
    return num_cpus, num_gpus, dataloader_workers


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Fine-tune Whisper model on LangAge dataset for French ASR.")
    parser.add_argument(
        "-d", "--dataset_path",
        type=str,
        required=True,
        help="Path to the LangAgeDataSet folder (HuggingFace dataset)"
    )
    parser.add_argument(
        "-o", "--output_dir",
        type=str,
        required=True,
        help="Directory where the fine-tuned model will be saved"
    )
    parser.add_argument(
        "--num_cpus",
        type=int,
        default=None,
        help="Number of CPU cores to use for data loading and preprocessing. If not specified, will use all available cores."
    )
    parser.add_argument(
        "--num_gpus",
        type=int,
        default=None,
        help="Number of GPUs to use for training. If not specified, will use all available GPUs. Set to 0 for CPU-only training."
    )
    parser.add_argument(
        "--dataloader_workers",
        type=int,
        default=None,
        help="Number of worker processes for data loading. If not specified, will be calculated based on CPU/GPU configuration."
    )
    return parser.parse_args()


def prepare_dataset(batch, feature_extractor, tokenizer):
    """
    Prepare dataset batch for Whisper training.
    Converts audio to log-Mel spectrograms and text to token IDs.
    """
    # Ensure audio is at 16kHz (should already be from LangAge dataset)
    audio = batch["audio"]
    
    # Compute log-Mel spectrogram input features from audio array
    batch["input_features"] = feature_extractor(
        audio["array"], 
        sampling_rate=audio["sampling_rate"]
    ).input_features[0]
    
    # Encode target text to label ids
    batch["labels"] = tokenizer(batch["text"]).input_ids
    
    return batch


def compute_metrics(pred, tokenizer, metric):
    """
    Compute Word Error Rate (WER) metric for evaluation.
    """
    pred_ids = pred.predictions
    label_ids = pred.label_ids
    
    # Replace -100 with the pad_token_id
    label_ids[label_ids == -100] = tokenizer.pad_token_id
    
    # Decode predicted and label token IDs to strings
    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)
    
    # Compute WER
    wer_result = metric.compute(predictions=pred_str, references=label_str)
    wer = 100 * wer_result
    
    return {"wer": wer}


### main ######################################################################

if __name__ == "__main__":
    print("Starting Whisper fine-tuning on LangAge dataset...")
    
    # Parse command line arguments
    args = parse_arguments()
    
    # Configure CPU and GPU resources based on user input
    num_cpus, num_gpus, dataloader_workers = configure_hardware(args)
    
    # Load the LangAge dataset from disk
    print(f"Loading dataset from {args.dataset_path}")
    try:
        dataset_dict = load_from_disk(args.dataset_path)
        print(f"Dataset loaded successfully!")
        print(f"Available splits: {list(dataset_dict.keys())}")
        
        # Use only the train split (since test was accidentally removed)
        if "train" not in dataset_dict:
            raise ValueError("No 'train' split found in dataset!")
            
        full_train_dataset = dataset_dict["train"]
        print(f"Full training dataset size: {len(full_train_dataset)} samples")
        
    except Exception as e:
        print(f"Error loading dataset: {e}")
        raise
    
    # Use only 1% of the data for now (hardcoded as requested)
    subset_size = int(0.01 * len(full_train_dataset))
    print(f"Using {subset_size} samples (1% of full dataset)")
    
    # Shuffle and select subset
    subset_dataset = full_train_dataset.shuffle(seed=42).select(range(subset_size))
    
    # Create 90% train / 10% eval split from the subset
    eval_size = int(0.1 * subset_size)
    train_size = subset_size - eval_size
    
    train_dataset = subset_dataset.select(range(train_size))
    eval_dataset = subset_dataset.select(range(train_size, subset_size))
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Eval samples: {len(eval_dataset)}")
    
    # Audio is already at 16kHz, no resampling needed
    print("Audio is already at 16kHz, skipping resampling...")
    
    # Initialize Whisper components
    print("Loading Whisper model and processors...")
    feature_extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-small")
    tokenizer = WhisperTokenizer.from_pretrained("openai/whisper-small", language="French", task="transcribe")
    processor = WhisperProcessor.from_pretrained("openai/whisper-small", language="French", task="transcribe")
    
    # Load model
    model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")
    
    # Configure model for French transcription
    model.generation_config.language = "french"
    model.generation_config.task = "transcribe"
    model.generation_config.forced_decoder_ids = None
    
    print("Model and processors loaded successfully!")
    
    # Prepare datasets - apply preprocessing
    print("Preprocessing datasets...")
    
    def prepare_dataset_with_processors(batch):
        return prepare_dataset(batch, feature_extractor, tokenizer)
    
    train_dataset = train_dataset.map(
        prepare_dataset_with_processors, 
        remove_columns=train_dataset.column_names,
        num_proc=num_cpus
    )
    
    eval_dataset = eval_dataset.map(
        prepare_dataset_with_processors,
        remove_columns=eval_dataset.column_names, 
        num_proc=num_cpus
    )
    
    print("Dataset preprocessing completed!")
    
    # Initialize data collator
    data_collator = DataCollatorSpeechSeq2SeqWithPadding(
        processor=processor,
        decoder_start_token_id=model.config.decoder_start_token_id,
    )
    
    # Initialize evaluation metric
    metric = evaluate.load("wer")
    
    def compute_metrics_with_deps(pred):
        return compute_metrics(pred, tokenizer, metric)
    
    # Define training arguments
    print("Setting up training configuration...")
    
    # Calculate optimal batch sizes based on GPU availability
    if num_gpus > 0:
        # GPU optimized settings
        per_device_train_batch_size = 32 if num_gpus > 1 else 16
        per_device_eval_batch_size = 16 if num_gpus > 1 else 8
        gradient_accumulation_steps = 1
        fp16_enabled = True
        print(f"GPU training mode: batch_size={per_device_train_batch_size}, workers={dataloader_workers}")
    else:
        # CPU optimized settings
        per_device_train_batch_size = 8
        per_device_eval_batch_size = 4
        gradient_accumulation_steps = 2
        fp16_enabled = False
        print(f"CPU training mode: batch_size={per_device_train_batch_size}, workers={dataloader_workers}")
    
    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=1e-5,
        warmup_steps=500,
        max_steps=5000,
        gradient_checkpointing=True,
        fp16=fp16_enabled,
        eval_strategy="steps",
        per_device_eval_batch_size=per_device_eval_batch_size,
        predict_with_generate=True,
        generation_max_length=225,
        save_steps=1000,
        eval_steps=1000,
        logging_steps=25,
        report_to=["tensorboard"],
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
        push_to_hub=False,
        dataloader_num_workers=dataloader_workers,
        ddp_find_unused_parameters=False if num_gpus > 1 else None,
        dataloader_pin_memory=True if num_gpus > 0 else False,
    )
    
    # Initialize trainer
    print("Initializing trainer...")
    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics_with_deps,
        tokenizer=processor.feature_extractor,
    )
    
    print("Starting training...")
    print(f"Model will be saved to: {args.output_dir}")
    print(f"Training on {len(train_dataset)} samples")
    print(f"Evaluating on {len(eval_dataset)} samples")
    print("-" * 50)
    
    # Start training
    trainer.train()
    
    print("Training completed!")
    print(f"Model saved to: {args.output_dir}")
    
    # Save final model
    trainer.save_model()
    processor.save_pretrained(args.output_dir)
    
    print("Fine-tuning completed successfully!")