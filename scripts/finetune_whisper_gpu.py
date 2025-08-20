#############################################################################
# Script Name: finetune_whisper_gpu.py                                      #
# Description: GPU-focused Whisper fine-tuning on preprocessed data        #
# Author: Hanno Müller                                                      #
# Date: 2025-07-25                                                          #
#############################################################################

### Required Libraries ######################################################
import os
import argparse
import torch
import multiprocessing
import gc
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

# Enable proper multi-GPU support
os.environ["TOKENIZERS_PARALLELISM"] = "false"


### Class Definitions ########################################################

@dataclass
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
        
        # Ensure all tensors are on the same device and have proper dtype
        for key, tensor in batch.items():
            if isinstance(tensor, torch.Tensor):
                if key == "labels":
                    batch[key] = tensor.long()
                else:
                    batch[key] = tensor.float()

        return batch


### Function Definitions #####################################################

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Fine-tune Whisper model on preprocessed LangAge dataset.")
    
    # Input/Output paths
    parser.add_argument(
        "-i", "--input_dataset",
        type=str,
        required=True,
        help="Path to the preprocessed dataset folder (HuggingFace dataset)"
    )
    parser.add_argument(
        "-o", "--output_dir",
        type=str,
        required=True,
        help="Directory where the fine-tuned model will be saved"
    )
    parser.add_argument(
        "-v", "--version",
        type=str,
        required=True,
        help="Version subfolder for the model (e.g., 'v1', 'v2')"
    )
    
    # Hardware configuration
    parser.add_argument(
        "--num_gpus",
        type=int,
        default=None,
        help="Number of GPUs to use for training. If not specified, will use all available GPUs."
    )
    parser.add_argument(
        "--num_cpus",
        type=int,
        default=10,
        help="Number of CPU cores to use for data loading (default: 10)"
    )
    parser.add_argument(
        "--dataloader_workers",
        type=int,
        default=None,
        help="Number of worker processes for data loading. If not specified, will be calculated from num_cpus."
    )
    
    # Model configuration
    parser.add_argument(
        "--model_size",
        type=str,
        default="large",
        choices=["tiny", "base", "small", "medium", "large"],
        help="Whisper model size for training (default: large)"
    )
    
    # Training hyperparameters
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-5,
        help="Learning rate for training (default: 1e-5)"
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=None,
        help="Training batch size per device. If not specified, will be auto-calculated."
    )
    parser.add_argument(
        "--per_device_eval_batch_size", 
        type=int,
        default=None,
        help="Evaluation batch size per device. If not specified, will be auto-calculated."
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of gradient accumulation steps (default: 1)"
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=5000,
        help="Maximum number of training steps (default: 5000)"
    )
    parser.add_argument(
        "--warmup_steps",
        type=int,
        default=500,
        help="Number of warmup steps (default: 500)"
    )
    parser.add_argument(
        "--save_steps",
        type=int,
        default=1000,
        help="Save checkpoint every N steps (default: 1000)"
    )
    parser.add_argument(
        "--eval_steps",
        type=int,
        default=1000,
        help="Evaluate every N steps (default: 1000)"
    )
    parser.add_argument(
        "--logging_steps",
        type=int,
        default=25,
        help="Log every N steps (default: 25)"
    )
    
    return parser.parse_args()


def configure_hardware(args):
    """Configure CPU and GPU resources based on user specifications."""
    print("=== Hardware Configuration ===")
    
    # Configure CPUs
    available_cpus = multiprocessing.cpu_count()
    print(f"Available CPUs: {available_cpus}")
    
    if args.num_cpus > available_cpus:
        print(f"Warning: Requested {args.num_cpus} CPUs, but only {available_cpus} available. Using {available_cpus}.")
        num_cpus = available_cpus
    elif args.num_cpus <= 0:
        print("Error: Number of CPUs must be positive.")
        raise ValueError("Invalid number of CPUs specified")
    else:
        num_cpus = args.num_cpus
    
    print(f"Using {num_cpus} CPU cores for data loading")
    
    # Configure GPUs - handle distributed training
    if "WORLD_SIZE" in os.environ and int(os.environ["WORLD_SIZE"]) > 1:
        # Distributed training mode
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        world_size = int(os.environ.get("WORLD_SIZE", 1))
        
        print(f"Distributed training mode: {world_size} total processes")
        print(f"Local GPU: {local_rank}")
        print(f"Current GPU: {torch.cuda.get_device_name(local_rank)} ({torch.cuda.get_device_properties(local_rank).total_memory / 1024**3:.1f} GB)")
        
        num_gpus = 1  # Each process handles 1 GPU in distributed mode
    else:
        # Single node mode
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
            # Use all available GPUs
            num_gpus = available_gpus
            print(f"Auto-detect mode: using {num_gpus} GPUs (all available)")
        
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
            print("Error: No GPUs available for training!")
            raise RuntimeError("GPU training script requires at least 1 GPU")
    
    # Configure data loading workers
    if args.dataloader_workers is not None:
        dataloader_workers = args.dataloader_workers
        print(f"Using {dataloader_workers} data loader workers (user specified)")
    else:
        # Calculate optimal workers from available CPUs
        dataloader_workers = min(num_cpus, 8)  # Cap at 8 for stability
        dataloader_workers = max(1, dataloader_workers)
        print(f"Using {dataloader_workers} data loader workers (auto-calculated)")
    
    print("=" * 35)
    
    return num_cpus, num_gpus, dataloader_workers


def validate_dataset(dataset_dict):
    """Validate that the dataset has the required format for training."""
    print("=== Dataset Validation ===")
    
    required_splits = ["train"]
    optional_splits = ["test", "eval", "validation"]
    
    # Check for required splits
    for split in required_splits:
        if split not in dataset_dict:
            raise ValueError(f"Required split '{split}' not found in dataset!")
    
    # Check available splits
    available_splits = list(dataset_dict.keys())
    print(f"Available splits: {available_splits}")
    
    # Validate each split
    for split_name, split_data in dataset_dict.items():
        print(f"Validating {split_name} split...")
        
        # Check required columns
        required_columns = ["input_features", "labels"]
        for column in required_columns:
            if column not in split_data.column_names:
                raise ValueError(f"Required column '{column}' not found in {split_name} split!")
        
        # Check sample data
        sample = split_data[0]
        print(f"  {split_name} samples: {len(split_data)}")
        print(f"  Columns: {split_data.column_names}")
        
        # Validate input_features
        input_features = sample["input_features"]
        if not isinstance(input_features, (list, tuple)) or len(input_features) == 0:
            raise ValueError(f"Invalid input_features format in {split_name} split!")
        print(f"  Sample input_features length: {len(input_features)}")
        
        # Validate labels
        labels = sample["labels"]
        if not isinstance(labels, (list, tuple)) or len(labels) == 0:
            raise ValueError(f"Invalid labels format in {split_name} split!")
        print(f"  Sample labels length: {len(labels)}")
    
    print("Dataset validation passed!")
    print("=" * 27)


def compute_metrics(pred, tokenizer, metric=None):
    """Compute evaluation metrics. Uses WER if available, otherwise accuracy."""
    pred_ids = pred.predictions
    label_ids = pred.label_ids
    
    # Replace -100 with the pad_token_id
    label_ids[label_ids == -100] = tokenizer.pad_token_id
    
    # Decode predicted and label token IDs to strings
    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)
    
    if metric is not None:
        try:
            # Try to compute WER
            wer_result = metric.compute(predictions=pred_str, references=label_str)
            wer = 100 * wer_result
            return {"wer": wer}
        except Exception as e:
            print(f"Warning: Could not compute WER metric: {e}")
    
    # Fallback to simple accuracy metric
    correct = sum(1 for pred, label in zip(pred_str, label_str) if pred.strip() == label.strip())
    accuracy = correct / len(pred_str) if len(pred_str) > 0 else 0.0
    
    return {"accuracy": accuracy * 100}


### main ######################################################################

if __name__ == "__main__":
    print("Starting Whisper GPU fine-tuning on preprocessed dataset...")
    
    # Parse command line arguments first
    args = parse_arguments()
    
    # Create full output path with version subfolder
    full_output_dir = os.path.join(args.output_dir, args.version)
    print(f"Model will be saved to: {full_output_dir}")
    
    # Create output directory if it doesn't exist
    os.makedirs(full_output_dir, exist_ok=True)
    
    # Initialize distributed training if using torchrun/multiple GPUs
    if "WORLD_SIZE" in os.environ and int(os.environ["WORLD_SIZE"]) > 1:
        # Running with torchrun - distributed training
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        world_size = int(os.environ.get("WORLD_SIZE", 1))
        rank = int(os.environ.get("RANK", 0))
        
        print(f"Distributed training detected: rank {rank}/{world_size}, local_rank {local_rank}")
        
        torch.cuda.set_device(local_rank)
        torch.distributed.init_process_group(
            backend="nccl",
            init_method="env://",
            world_size=world_size,
            rank=rank
        )
        
        # Only print from the main process
        if rank != 0:
            import sys
            import logging
            logging.getLogger().setLevel(logging.WARNING)
    else:
        print("Single GPU or DataParallel training mode")
    
    # Configure CPU and GPU resources
    num_cpus, num_gpus, dataloader_workers = configure_hardware(args)
    
    # Load the preprocessed dataset
    print(f"Loading preprocessed dataset from {args.input_dataset}")
    try:
        dataset_dict = load_from_disk(args.input_dataset)
        print(f"Dataset loaded successfully!")
        
        # Validate dataset format
        validate_dataset(dataset_dict)
        
    except Exception as e:
        print(f"Error loading dataset: {e}")
        raise
    
    # Prepare datasets
    train_dataset = dataset_dict["train"]
    
    # Check for evaluation dataset
    eval_dataset = None
    for eval_split_name in ["test", "eval", "validation"]:
        if eval_split_name in dataset_dict:
            eval_dataset = dataset_dict[eval_split_name]
            print(f"Using {eval_split_name} split for evaluation")
            break
    
    if eval_dataset is None:
        # Create eval split from train (90/10 split)
        print("No evaluation split found, creating 90/10 split from train data")
        eval_size = int(0.1 * len(train_dataset))
        train_size = len(train_dataset) - eval_size
        
        train_dataset = train_dataset.select(range(train_size))
        eval_dataset = train_dataset.select(range(train_size, train_size + eval_size))
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Eval samples: {len(eval_dataset)}")
    
    # Initialize Whisper components
    model_name = f"openai/whisper-{args.model_size}"
    print(f"Loading Whisper model: {model_name}")
    
    feature_extractor = WhisperFeatureExtractor.from_pretrained(model_name)
    tokenizer = WhisperTokenizer.from_pretrained(model_name, language="French", task="transcribe")
    processor = WhisperProcessor.from_pretrained(model_name, language="French", task="transcribe")
    
    # Load model
    model = WhisperForConditionalGeneration.from_pretrained(model_name)
    
    # Configure model for French transcription
    model.generation_config.language = "french"
    model.generation_config.task = "transcribe"
    model.generation_config.forced_decoder_ids = None
    model.generation_config.use_cache = False  # Disable cache for training
    
    # Disable caching in the model config to avoid multi-GPU issues
    model.config.use_cache = False
    
    # Check if we're in distributed mode
    is_distributed = "WORLD_SIZE" in os.environ and int(os.environ["WORLD_SIZE"]) > 1
    
    # Additional memory optimizations for distributed training
    if is_distributed:
        # Force garbage collection
        gc.collect()
        torch.cuda.empty_cache()
        
        # Set memory fraction to avoid fragmentation - be more conservative
        if torch.cuda.is_available():
            torch.cuda.set_per_process_memory_fraction(0.75)  # Use only 75% of GPU memory per process
            
        # Additional memory optimizations
        torch.backends.cudnn.benchmark = False  # Disable for memory savings
        
        # Move model to current device immediately after loading
        model = model.to(f"cuda:{torch.cuda.current_device()}")
        torch.cuda.empty_cache()
        gc.collect()
    
    print("Model and processors loaded successfully!")
    
    # Initialize data collator
    data_collator = DataCollatorSpeechSeq2SeqWithPadding(
        processor=processor,
        decoder_start_token_id=model.config.decoder_start_token_id,
    )
    
    # Initialize evaluation metric (optional)
    metric = None
    try:
        metric = evaluate.load("wer")
        print("WER metric loaded successfully")
    except Exception as e:
        print(f"Warning: Could not load WER metric ({e}). Will use accuracy instead.")
    
    def compute_metrics_with_deps(pred):
        return compute_metrics(pred, tokenizer, metric)
    
    # Configure training parameters
    print("Setting up training configuration...")
    
    # Calculate batch sizes if not specified - reduce for distributed training
    if args.per_device_train_batch_size is None:
        if is_distributed:
            per_device_train_batch_size = 1  # Very small batch for distributed training to avoid OOM
        else:
            per_device_train_batch_size = 32 if num_gpus > 1 else 16
    else:
        per_device_train_batch_size = args.per_device_train_batch_size
    
    if args.per_device_eval_batch_size is None:
        if is_distributed:
            per_device_eval_batch_size = 1  # Very small eval batch for distributed training
        else:
            per_device_eval_batch_size = 16 if num_gpus > 1 else 8
    else:
        per_device_eval_batch_size = args.per_device_eval_batch_size
    
    print(f"Training batch size per device: {per_device_train_batch_size}")
    print(f"Evaluation batch size per device: {per_device_eval_batch_size}")
    print(f"Dataloader workers: {dataloader_workers}")
    
    # Define training arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir=full_output_dir,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=4 if is_distributed else args.gradient_accumulation_steps,  # Moderate accumulation for distributed
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        max_steps=args.max_steps,
        gradient_checkpointing=False,  # Disable gradient checkpointing to avoid graph issues in distributed training
        fp16=True,
        eval_strategy="steps",
        per_device_eval_batch_size=per_device_eval_batch_size,
        predict_with_generate=True,
        generation_max_length=225,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        logging_steps=args.logging_steps,
        report_to=["tensorboard"] if not is_distributed or int(os.environ.get("RANK", 0)) == 0 else [],
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",  # Use loss instead of WER since WER might not be available
        greater_is_better=False,
        push_to_hub=False,
        dataloader_num_workers=2 if is_distributed else min(dataloader_workers, 4),  # Reduce workers for distributed
        ddp_find_unused_parameters=False,
        dataloader_pin_memory=False if is_distributed else True,  # Disable pin memory for distributed to save memory
        remove_unused_columns=False,  # Keep all columns to avoid issues with custom data collator
        # Distributed training settings
        ddp_backend="nccl" if is_distributed else None,
        group_by_length=False,  # Disable to avoid issues with variable-length sequences
        save_on_each_node=False,  # Only save on main node
        logging_first_step=True,
        # Memory optimizations for distributed training
        max_grad_norm=1.0,  # Gradient clipping
        optim="adamw_torch", #if is_distributed else "adamw_hf",  # Use torch optimizer for better memory management
        ddp_bucket_cap_mb=25 if is_distributed else None,  # Reduce DDP bucket size for memory efficiency
        dataloader_drop_last=True,  # Drop last incomplete batch for consistent training
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
        processing_class=processor.feature_extractor,  # Use processing_class instead of tokenizer
    )
    
    print("Starting training...")
    print(f"Model will be saved to: {full_output_dir}")
    print(f"Training on {len(train_dataset)} samples")
    print(f"Evaluating on {len(eval_dataset)} samples")
    print("-" * 50)
    
    # Start training
    trainer.train()
    
    print("Training completed!")
    print(f"Model saved to: {full_output_dir}")
    
    # Save final model
    trainer.save_model()
    processor.save_pretrained(full_output_dir)
    
    print("GPU fine-tuning completed successfully!")
