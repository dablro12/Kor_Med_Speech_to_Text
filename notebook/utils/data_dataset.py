import os
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import torchaudio

from dataclasses import dataclass
from typing import Any, Dict, List, Union

from transformers import WhisperProcessor


# ============================================================
# Dataset
# ============================================================
class KruWhisperDataset(Dataset):
    """
    KRU Dataset → Whisper Fine-tuning Custom Dataset

    CSV 구조:
    ├─ abs_path
    ├─ transcription
    """

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

        # ----------------------------------------
        # Load audio
        # ----------------------------------------
        try:
            waveform, sr = torchaudio.load(audio_path)
        except Exception as e:
            print(f"[WARN] Audio load failed at {audio_path} : {e}")
            waveform = torch.zeros(1, self.target_rate)
            sr = self.target_rate

        # Resample → 16kHz
        if sr != self.target_rate:
            waveform = torchaudio.functional.resample(
                waveform, orig_freq=sr, new_freq=self.target_rate
            )

        # ----------------------------------------
        # Whisper Feature Extract
        # ----------------------------------------
        input_features = self.feature_extractor(
            waveform.squeeze().numpy(),
            sampling_rate=self.target_rate
        ).input_features[0]

        # ----------------------------------------
        # Whisper Tokenization (labels)
        # ----------------------------------------
        labels = self.tokenizer(text).input_ids

        return {
            "input_features": torch.tensor(input_features, dtype=torch.float32),
            "labels": labels,   # tensor로 바꾸지 않음! collator가 처리함
            "audio_path": audio_path,
            "text": text,
        }


# ============================================================
# Data Collator (공식 HuggingFace Whisper 구조)
# ============================================================
@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any
    decoder_start_token_id: int

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]):

        # --------------------------------------------
        # 1) input_features: feature_extractor가 처리
        # --------------------------------------------
        input_features = [{"input_features": f["input_features"]} for f in features]
        batch = self.processor.feature_extractor.pad(
            input_features, return_tensors="pt"
        )

        # --------------------------------------------
        # 2) labels: tokenizer로 처리
        # --------------------------------------------
        label_features = [{"input_ids": f["labels"]} for f in features]
        labels_batch = self.processor.tokenizer.pad(
            label_features, return_tensors="pt"
        )

        # padding token → -100
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )

        # bos 제거 (Whisper 규칙)
        if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels

        return batch


# ============================================================
# Usage Example
# ============================================================
if __name__ == "__main__":

    # 1) Load processor
    processor = WhisperProcessor.from_pretrained(
        "openai/whisper-small",
        language="ko",
        task="transcribe"
    )
    feature_extractor = processor.feature_extractor
    tokenizer = processor.tokenizer

    # 2) Load Dataset
    train_dataset = KruWhisperDataset(
        csv_path="/workspace/kru_data/train.csv",
        feature_extractor=feature_extractor,
        tokenizer=tokenizer,
        sampling_rate=16000,
    )

    # 3) Data Collator (공식)
    data_collator = DataCollatorSpeechSeq2SeqWithPadding(
        processor=processor,
        decoder_start_token_id=tokenizer.bos_token_id,
    )

    # 4) DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=4,
        shuffle=True,
        collate_fn=data_collator,
    )

    # 5) 샘플 배치 확인
    batch = next(iter(train_loader))
    print("Batch Keys:", batch.keys())
    print("input_features:", batch["input_features"].shape)
    print("labels:", batch["labels"].shape)
