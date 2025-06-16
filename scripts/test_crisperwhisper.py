import os
import sys
import torch
import soundfile as sf

from datasets import load_dataset
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

def adjust_pauses_for_hf_pipeline_output(pipeline_output, split_threshold=0.12):
    """
    Adjust pause timings by distributing pauses up to the threshold evenly between adjacent words.
    """

    adjusted_chunks = pipeline_output["chunks"].copy()

    for i in range(len(adjusted_chunks) - 1):
        current_chunk = adjusted_chunks[i]
        next_chunk = adjusted_chunks[i + 1]

        current_start, current_end = current_chunk["timestamp"]
        next_start, next_end = next_chunk["timestamp"]
        pause_duration = next_start - current_end

        if pause_duration > 0:
            if pause_duration > split_threshold:
                distribute = split_threshold / 2
            else:
                distribute = pause_duration / 2

            # Adjust current chunk end time
            adjusted_chunks[i]["timestamp"] = (current_start, current_end + distribute)

            # Adjust next chunk start time
            adjusted_chunks[i + 1]["timestamp"] = (next_start - distribute, next_end)
    pipeline_output["chunks"] = adjusted_chunks

    return pipeline_output


device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

model_id = "nyrahealth/CrisperWhisper"

model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
)
model.to(device)

processor = AutoProcessor.from_pretrained(model_id)

pipe = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    chunk_length_s=30,
    batch_size=16,
    return_timestamps='word',
    torch_dtype=torch_dtype,
    device=device,
)

# use a test dataset
#dataset = load_dataset("distil-whisper/librispeech_long", "clean", split="validation")
#sample = dataset[0]["audio"]
#hf_pipeline_output = pipe(sample)

# use a custom dataset
file_path = "./data_sample/a040a.wav"
start_sec = 8 * 60 + 40  # 8 minutes 40 seconds = 520 seconds
end_sec = 9 * 60         # 9 minutes = 540 seconds

with sf.SoundFile(file_path) as f:
    sampling_rate = f.samplerate
    start_frame = int(start_sec * sampling_rate)
    num_frames = int((end_sec - start_sec) * sampling_rate)
    f.seek(start_frame)
    array = f.read(frames=num_frames)

sample = {
    "path": file_path,
    "array": array,
    "sampling_rate": sampling_rate
}
hf_pipeline_output = pipe(sample)

# transcribe
crisper_whisper_result = adjust_pauses_for_hf_pipeline_output(hf_pipeline_output)
print(crisper_whisper_result)
