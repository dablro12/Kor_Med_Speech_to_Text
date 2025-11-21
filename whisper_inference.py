import os
import sys 
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import time
import torch
import librosa
import numpy as np
import pandas as pd
from tqdm import tqdm
from transformers import WhisperProcessor, WhisperForConditionalGeneration

import pyarrow as pa
import pyarrow.parquet as pq


# --------------------------------------------------------
# 1) Auto Batch 기반 Whisper 배치 inference + time tracking
# --------------------------------------------------------
def transcribe_batch_auto(batch_paths, model, processor, sampling_rate=16000, init_batch_size=64):
    """
    Whisper 배치 inference 자동 스케일링 + 배치 단위 소요시간 측정
    """
    batch_size = min(init_batch_size, len(batch_paths))
    audios = []

    # Load audio
    for path in batch_paths:
        try:
            audio, _ = librosa.load(path, sr=sampling_rate)
        except:
            audio = np.zeros(sampling_rate, dtype=np.float32)
        audios.append(audio)

    while batch_size > 0:
        try:
            t0 = time.time()

            input_features = processor.feature_extractor(
                audios[:batch_size],
                sampling_rate=sampling_rate,
                return_tensors="pt"
            ).input_features.to("cuda")

            predicted_ids = model.generate(input_features)
            transcriptions = processor.tokenizer.batch_decode(
                predicted_ids, skip_special_tokens=True
            )

            t1 = time.time()
            elapsed = t1 - t0  # total time for this batch

            # per-file time
            per_sample_time = elapsed / batch_size
            per_times = [per_sample_time] * batch_size

            return transcriptions, per_times, batch_size

        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"[WARN] OOM → batch_size {batch_size} → {batch_size // 2}")
                torch.cuda.empty_cache()
                batch_size //= 2
            else:
                print("[ERROR] Unexpected error:", e)
                return [""] * batch_size, [0] * batch_size, batch_size

    return [""] * len(batch_paths), [0] * len(batch_paths), 1



# --------------------------------------------------------
# 2) Inference + Parquet 저장 + Auto Batch + time tracking
# --------------------------------------------------------
def run_inference(model_name, csv_path, save_path=None, init_batch_size=64):
    processor = WhisperProcessor.from_pretrained(
        model_name, language="ko", task="transcribe"
    )
    model = WhisperForConditionalGeneration.from_pretrained(model_name).to("cuda")

    df = pd.read_csv(csv_path)

    results = {
        "abs_path": [],
        "gt_text": [],
        "pred_text": [],
        "inference_time_sec": [],   # 🔥 추가됨
    }

    parquet_path = save_path.replace(".csv", ".parquet")
    save_dir = os.path.dirname(parquet_path)
    os.makedirs(save_dir, exist_ok=True)

    # Resume
    if os.path.exists(parquet_path):
        prev_df = pq.read_table(parquet_path).to_pandas()

        done = set(prev_df["abs_path"].tolist())
        results["abs_path"].extend(prev_df["abs_path"].tolist())
        results["gt_text"].extend(prev_df["gt_text"].tolist())
        results["pred_text"].extend(prev_df["pred_text"].tolist())
        results["inference_time_sec"].extend(prev_df["inference_time_sec"].tolist())

        print(f"[INFO] Resumed from {parquet_path}")
    else:
        done = set()

    batch_paths = []
    batch_gt = []
    batch_times = []

    last_saved_count = len(results["abs_path"])

    print(f"[INFO] Total rows = {len(df)}, Already processed = {len(done)}")

    # -----------------------------
    # Batch inference loop
    # -----------------------------
    for _, row in tqdm(df.iterrows(), total=len(df)):
        audio_path = row["abs_path"]
        gt_text = row["transcription"]

        if audio_path in done:
            continue

        batch_paths.append(audio_path)
        batch_gt.append(gt_text)

        if len(batch_paths) >= init_batch_size:

            preds, per_times, used_bs = transcribe_batch_auto(
                batch_paths, model, processor, init_batch_size=init_batch_size
            )

            results["abs_path"].extend(batch_paths[:used_bs])
            results["gt_text"].extend(batch_gt[:used_bs])
            results["pred_text"].extend(preds)
            results["inference_time_sec"].extend(per_times)

            # Remove processed
            batch_paths = batch_paths[used_bs:]
            batch_gt = batch_gt[used_bs:]

            # Save every 10k
            if (len(results["abs_path"]) - last_saved_count) >= 10000:
                print(f"[INFO] Saving chunk → {parquet_path}")
                table = pa.Table.from_pandas(pd.DataFrame(results))
                pq.write_table(table, parquet_path)
                last_saved_count = len(results["abs_path"])

    # -----------------------------
    # 마지막 남은 batch
    # -----------------------------
    if len(batch_paths) > 0:
        preds, per_times, used_bs = transcribe_batch_auto(
            batch_paths, model, processor, init_batch_size=init_batch_size
        )
        results["abs_path"].extend(batch_paths[:used_bs])
        results["gt_text"].extend(batch_gt[:used_bs])
        results["pred_text"].extend(preds)
        results["inference_time_sec"].extend(per_times)

    # -----------------------------
    # 최종 저장
    # -----------------------------
    print(f"[INFO] Final save → {parquet_path}")
    table = pa.Table.from_pandas(pd.DataFrame(results))
    pq.write_table(table, parquet_path)

    return pd.DataFrame(results)



# --------------------------------------------------------
# 3) 실행
# --------------------------------------------------------
if __name__ == "__main__":
    model_series = ['base', 'small', 'medium', 'tiny']

    for model_name in model_series:
        run_inference(
            model_name=f"openai/whisper-{model_name}",
            csv_path="/workspace/kru_data/test.csv",
            save_path=f"/workspace/results/whisper_inference/whisper_{model_name}_inference/test_pred.csv",
            init_batch_size=1024
        )
