#############################################################################
# Script Name: finetune_whisper.py                                          #
# Description: This script takes a folder with TextGrids and audios and     #
#              uses both to finetune a Whisper model.                       #
# Author: Hanno Müller                                                      #
# Date: 2025-06-30                                                          #
#############################################################################

### Required Libraries ######################################################
import os
import argparse
import soundfile as sf
import sounddevice as sd # necessary for audio playback (e.g., debugging, testing)
import numpy as np
import librosa
from TextGrids2Dataset import TextGrid
from transformers import WhisperFeatureExtractor
from transformers import WhisperTokenizer
from transformers import WhisperProcessor
from transformers import WhisperForConditionalGeneration
from datasets import Dataset, DatasetDict, Audio, load_dataset, concatenate_datasets
import torch
from dataclasses import dataclass
from typing import Any, Dict, List, Union
import evaluate
from transformers import Seq2SeqTrainingArguments
from transformers import Seq2SeqTrainer
import random


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

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Process TextGrid data from a specified folder.")
    parser.add_argument(
        "-f", "--folder",
        type=str,
        required=True,
        help="Path to the folder containing TextGrid data"
    )
    return parser.parse_args()


def load_textgrids_from_folder(folder_path):
    """
    Takes a folder with textgrids and returns a list of TextGrid objects.
    """
    textgrids = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".TextGrid"):
            file_path = os.path.join(folder_path, filename)
            textgrids.append(TextGrid.load_textgrid(file_path))
    return textgrids


def prepare_dataset(batch):
    # load and resample audio data from 48 to 16kHz
    audio = batch["audio"]

    # compute log-Mel input features from input audio array 
    batch["input_features"] = feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]

    # encode target text to label ids 
    batch["labels"] = tokenizer(batch["sentence"]).input_ids
    return batch


def tokenize(input_str, tokenizer):
    """
    Prepare the Whisper processor with feature extractor and tokenizer.
    """
    labels = tokenizer(input_str).input_ids
    decoded_str = tokenizer.decode(labels, skip_special_tokens=True)
    #return decoded_str
    return labels

def compute_metrics(pred):
    pred_ids = pred.predictions
    label_ids = pred.label_ids

    # replace -100 with the pad_token_id
    label_ids[label_ids == -100] = tokenizer.pad_token_id

    # we do not want to group tokens when computing the metrics
    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    wer = 100 * metric.compute(predictions=pred_str, references=label_str)

    return {"wer": wer}


### main ######################################################################

if __name__ == "__main__":

    # Retrieve command line arguments
    vars = parse_arguments()
    
    # Load TextGrids from the specified folder
    textgrids = load_textgrids_from_folder(vars.folder)

    # Combine all dicts from all textgrids
    all_datasets = []
    created_indices = []
    for i, tg in enumerate(textgrids):
        try:
            ds = tg.to_dataset()
            parquet_path = f"temp_dataset_{i}.parquet"
            ds.to_parquet(parquet_path)
            del ds  # free memory
            created_indices.append(i)
            print(f"Processed {tg.path}")
        except Exception as e:
            print(f"Error processing {tg}: {e}")

    # Only load Parquet files that exist
    all_datasets = [
        Dataset.from_parquet(f"temp_dataset_{i}.parquet")
        for i in created_indices
        if os.path.exists(f"temp_dataset_{i}.parquet")
    ]

    #for tg in textgrids:
    #    try:
    #        all_datasets.append(tg.to_dataset())
    #    except Exception as e:
    #        print(f"Could not append dataset for {tg}: {e}")

    # Concatenate all datasets into one
    from datasets import concatenate_datasets
    full_dataset = concatenate_datasets(all_datasets)

    # Shuffle the dataset
    full_dataset = full_dataset.shuffle(seed=42)

    # Split into 80% train, 20% test
    split_idx = int(0.8 * len(full_dataset))
    train_dataset = full_dataset.select(range(split_idx))
    test_dataset = full_dataset.select(range(split_idx, len(full_dataset)))

    # Create DatasetDict
    LangAge = DatasetDict({
        "train": train_dataset,
        "test": test_dataset
    })

    # Example usage:
    print(LangAge)

    RUN = False
    if RUN:

        # Prepare the Whisper processor
        feature_extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-small")
        tokenizer = WhisperTokenizer.from_pretrained("openai/whisper-small", language="French", task="transcribe")

        processor = WhisperProcessor.from_pretrained("openai/whisper-small", language="French", task="transcribe")
        model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")

        model.generation_config.language = "French"
        model.generation_config.task = "transcribe"
        model.generation_config.forced_decoder_ids = None

        #data_collator = DataCollatorSpeechSeq2SeqWithPadding(
        #    processor=processor,
        #    decoder_start_token_id=model.config.decoder_start_token_id,
        #)

        metric = evaluate.load("wer")

        # define training arguments
        """
        training_args = Seq2SeqTrainingArguments(
            output_dir="../tmpTrain",  # change to a repo name of your choice
            per_device_train_batch_size=16,
            gradient_accumulation_steps=1,  # increase by 2x for every 2x decrease in batch size
            learning_rate=1e-5,
            warmup_steps=500,
            max_steps=5000,
            gradient_checkpointing=True,
            fp16=True,
            eval_strategy="steps",
            per_device_eval_batch_size=8,
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
        )

        trainer = Seq2SeqTrainer(
            args=training_args,
            model=model,
            train_dataset=common_voice["train"],
            eval_dataset=common_voice["test"],
            data_collator=data_collator,
            compute_metrics=compute_metrics,
            tokenizer=processor.feature_extractor,
        )
        """

    # code for debugging
    RUN=False
    if RUN:
        all_unique_tokens = []

        for textgrid in textgrids:
            datasets = textgrid.to_dataset()
            sentences = [entry["sentence"] for entry in datasets]
            for sentence in sentences:
                try:
                    token = tokenize(sentence, tokenizer)
                except Exception as e:
                    print(f"Tokenization failed for sentence: {sentence}\nError: {e}")
                    token = None
                all_unique_tokens.extend(token)
            #dataset = dataset.map(prepare_dataset, num_proc=10)

        all_unique_tokens = sorted(list(set(all_unique_tokens)))
        decoded_tokens = [tokenizer.decode([item], skip_special_tokens=True) for item in all_unique_tokens]
        print(decoded_tokens)

        #trainer.train()



    # code for debugging
    RUN=False
    if RUN:
        dicts = textgrids[0].to_dict(resample=16000)
        spk2_entries = [d for d in dicts if d['speaker'] == 'spk2']

        for i, entry in enumerate(spk2_entries[1:]):  # Skip the first entry
            print(entry)
            input("Press Enter to play the next audio...")
            audio_array = entry['audio']['array']
            sampling_rate = entry['audio']['sampling_rate']
            sd.play(audio_array, sampling_rate)
            sd.wait()
            print('*' * 20)
            if i >= 10:
                break