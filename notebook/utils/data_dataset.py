import os
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import librosa      # ← 추가됨

from dataclasses import dataclass
from typing import Any, Dict, List, Union

from transformers import WhisperProcessor


class KruWhisperDataset(Dataset):
    def __init__(self, csv_path, feature_extractor, tokenizer, sampling_rate=16000):
        self.df = pd.read_csv(csv_path)
        self.feature_extractor = feature_extractor
        self.tokenizer = tokenizer
        self.target_rate = sampling_rate

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        audio_path = row["abs_path"]
        text = row["transcription"]

        # 1) Load audio with librosa
        try:
            waveform, sr = librosa.load(audio_path, sr=self.target_rate)  # auto-resample
            # librosa returns np.ndarray of shape (samples,)
            waveform = torch.tensor(waveform, dtype=torch.float32)
        except Exception as e:
            print(f"[ERROR] Audio load failed at {audio_path}: {e}")
            waveform = torch.zeros(self.target_rate, dtype=torch.float32)
            sr = self.target_rate

        # 2) Feature Extract
        input_features = self.feature_extractor(
            waveform.numpy(),
            sampling_rate=self.target_rate
        ).input_features[0]

        # 3) Tokenization
        labels = self.tokenizer(text).input_ids

        return {
            "input_features": torch.tensor(input_features, dtype=torch.float32),
            "labels": labels,
            "audio_path": audio_path,
            "text": text,
        }


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any
    decoder_start_token_id: int

    def __call__(self, features):
        input_features = [f["input_features"] for f in features]
        batch = self.processor.feature_extractor.pad(
            {"input_features": input_features}, return_tensors="pt"
        )

        labels = [f["labels"] for f in features]
        labels_batch = self.processor.tokenizer.pad(
            {"input_ids": labels}, return_tensors="pt"
        )

        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )

        # remove BOS token
        if (labels[:, 0] == self.decoder_start_token_id).all():
            labels = labels[:, 1:]

        batch["labels"] = labels
        return batch


if __name__ == "__main__":
    processor = WhisperProcessor.from_pretrained(
        "openai/whisper-small",
        language="ko",
        task="transcribe"
    )

    train_dataset = KruWhisperDataset(
        csv_path="/workspace/kru_data/sample.csv",
        feature_extractor=processor.feature_extractor,
        tokenizer=processor.tokenizer,
        sampling_rate=16000,
    )

    data_collator = DataCollatorSpeechSeq2SeqWithPadding(
        processor=processor,
        decoder_start_token_id=processor.tokenizer.bos_token_id,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=4,
        shuffle=False,
        collate_fn=data_collator,
    )

    batch = next(iter(train_loader))
