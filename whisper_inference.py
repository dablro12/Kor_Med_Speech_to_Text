import os
import sys 
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import torch
import torchaudio
import pandas as pd
from transformers import WhisperProcessor, WhisperForConditionalGeneration

def transcribe_audio(wav_path, model, processor, sampling_rate=16000):
    # Load audio
    waveform, sr = torchaudio.load(wav_path)

    # Resample if needed
    if sr != sampling_rate:
        waveform = torchaudio.functional.resample(waveform, sr, sampling_rate)

    # Convert to log-mel
    input_features = processor.feature_extractor(
        waveform.squeeze().numpy(),
        sampling_rate=sampling_rate,
        return_tensors="pt"
    ).input_features.to("cuda")

    # Generate predicted ids
    predicted_ids = model.generate(input_features)

    # Decode to text
    transcription = processor.tokenizer.batch_decode(predicted_ids, skip_special_tokens=True)[0]

    return transcription


import os
def run_inference(model_name, csv_path, save_path=None):
    # Load Whisper Small
    processor = WhisperProcessor.from_pretrained(
        model_name,
        language="ko",
        task="transcribe"
    )
    model = WhisperForConditionalGeneration.from_pretrained(model_name).to("cuda")

    df = pd.read_csv(csv_path)

    results = {
        "abs_path": [],
        "gt_text": [],
        "pred_text": [],
    }

    save_dir = os.path.dirname(save_path)
    if save_dir and not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Try to load previous DF, if file does not exist, just start fresh
    if save_path and os.path.exists(save_path):
        prev_df = pd.read_csv(save_path)
        done = set(prev_df["abs_path"].tolist())
        results["abs_path"].extend(prev_df["abs_path"].tolist())
        results["gt_text"].extend(prev_df["gt_text"].tolist())
        results["pred_text"].extend(prev_df["pred_text"].tolist())
    else:
        done = set()

    for _, row in df.iterrows():
        audio_path = row["abs_path"]
        gt_text = row["transcription"]

        # Skip already processed files
        if audio_path in done:
            continue

        try:
            pred = transcribe_audio(audio_path, model, processor)
        except Exception as e:
            print(f"[Error] {audio_path} : {e}")
            pred = ""

        results["abs_path"].append(audio_path)
        results["gt_text"].append(gt_text)
        results["pred_text"].append(pred)

        # Save after every row
        if save_path:
            pd.DataFrame(results).to_csv(save_path, index=False)
            print(f"Saved progress → {save_path}")
            
        # break  # NOTE: Remove or comment out this break if you want to process all rows!
    out_df = pd.DataFrame(results)

    return out_df

if __name__ == "__main__":
    model_series = ['tiny', 'base', 'small', 'medium', 'large', 'turbo']
    
    for model_name in model_series:
        run_inference(
            model_name=f"openai/whisper-{model_name}",
            csv_path="/workspace/kru_data/train.csv",
            save_path=f"/workspace/results/whisper_{model_name}_inference/train_pred.csv"
        )
        run_inference(
            model_name=f"openai/whisper-{model_name}",
            csv_path="/workspace/kru_data/test.csv",
            save_path=f"/workspace/results/whisper_{model_name}_inference/test_pred.csv"
        )