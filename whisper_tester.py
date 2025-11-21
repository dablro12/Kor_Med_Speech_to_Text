import os
import torch
import pandas as pd
import librosa
import numpy as np
from transformers import AutoProcessor, WhisperForConditionalGeneration
import pyarrow.parquet as pq
import pyarrow as pa
from tqdm import tqdm
import transformers
import time

transformers.logging.set_verbosity_error()

class WhisperInference:
    def __init__(
        self,
        model_dir: str,
        sampling_rate: int = 16000,
        device: str = "cuda",
        save_every: int = 10000,
        init_batch_size: int = 64
    ):
        self.model_dir = model_dir
        self.sampling_rate = sampling_rate
        self.device = device
        self.save_every = save_every
        self.init_batch_size = init_batch_size

        # base model name
        model_name = os.path.basename(os.path.dirname(self.model_dir))
        hf_model_id = f"openai/{model_name}"

        # Processor
        self.processor = AutoProcessor.from_pretrained(
            hf_model_id, language="ko", task="transcribe"
        )

        # Model
        self.model = WhisperForConditionalGeneration.from_pretrained(model_dir).to(device)


    # ---------------------------------------------------------
    # 🔥 1) 배치 inference + VRAM 자동 조절 + 타이머 추가
    # ---------------------------------------------------------
    def transcribe_batch(self, batch_paths):
        batch_size = len(batch_paths)
        audios = []

        for p in batch_paths:
            try:
                audio, _ = librosa.load(p, sr=self.sampling_rate)
            except:
                audio = np.zeros(self.sampling_rate, dtype=np.float32)
            audios.append(audio)

        bs = min(batch_size, self.init_batch_size)

        while bs > 0:
            try:
                feats = self.processor.feature_extractor(
                    audios[:bs],
                    sampling_rate=self.sampling_rate,
                    return_tensors="pt"
                ).input_features.to(self.device)

                torch.cuda.synchronize()
                t0 = time.time()

                with torch.no_grad():
                    pred_ids = self.model.generate(feats)

                torch.cuda.synchronize()
                t1 = time.time()

                batch_time = t1 - t0
                per_sample_time = batch_time / bs

                texts = self.processor.batch_decode(pred_ids, skip_special_tokens=True)

                return texts, [per_sample_time] * bs, bs

            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    print(f"[WARN] OOM 발생 → batch_size {bs} → {bs//2}")
                    torch.cuda.empty_cache()
                    bs //= 2
                else:
                    print("[ERROR]", e)
                    return [""] * bs, [0.0] * bs, bs

        return [""] * batch_size, [0.0] * batch_size, 1


    # ---------------------------------------------------------
    # 🔥 2) CSV + tqdm + resume + auto-save (inference_time 저장)
    # ---------------------------------------------------------
    def transcribe_csv_auto(
        self,
        input_csv: str,
        audio_column: str = "abs_path",
        gt_column: str = "gt_text",
        output_parquet: str = "test_pred.parquet",
    ):
        df = pd.read_csv(input_csv)

        results = {
            "abs_path": [],
            "gt_text": [],
            "pred_text": [],
            "inference_time_sec": []
        }
        done = set()

        # Resume logic
        if os.path.exists(output_parquet):
            prev = pq.read_table(output_parquet).to_pandas()
            results["abs_path"] = prev["abs_path"].tolist()
            results["gt_text"] = prev["gt_text"].tolist()
            results["pred_text"] = prev["pred_text"].tolist()

            # 🔥 기존 inference_time도 resume
            if "inference_time_sec" in prev.columns:
                results["inference_time_sec"] = prev["inference_time_sec"].tolist()
            else:
                results["inference_time_sec"] = [0.0] * len(prev)

            done = set(results["abs_path"])
            print(f"[INFO] Resume from {output_parquet} ({len(done)} rows)")

        last_saved = len(results["abs_path"])
        batch_paths = []
        batch_gt = []

        total_remaining = len(df) - len(done)
        pbar = tqdm(total=total_remaining, desc="Transcribing", unit="audio")

        for _, row in df.iterrows():
            audio_path = row[audio_column]
            gt_text = row[gt_column]

            if audio_path in done:
                pbar.update(1)
                continue

            batch_paths.append(audio_path)
            batch_gt.append(gt_text)

            if len(batch_paths) >= self.init_batch_size:
                preds, times, used_bs = self.transcribe_batch(batch_paths)

                results["abs_path"].extend(batch_paths[:used_bs])
                results["gt_text"].extend(batch_gt[:used_bs])
                results["pred_text"].extend(preds)
                results["inference_time_sec"].extend(times)

                batch_paths = batch_paths[used_bs:]
                batch_gt = batch_gt[used_bs:]
                pbar.update(used_bs)

                processed = len(results["abs_path"]) - last_saved
                if processed >= self.save_every:
                    print(f"[INFO] Auto-save → {output_parquet}")

                    os.makedirs(os.path.dirname(output_parquet), exist_ok=True)

                    table = pa.Table.from_pandas(pd.DataFrame(results))
                    pq.write_table(table, output_parquet)
                    last_saved = len(results["abs_path"])

        # 마지막 남은 batch
        if len(batch_paths) > 0:
            preds, times, used_bs = self.transcribe_batch(batch_paths)

            results["abs_path"].extend(batch_paths[:used_bs])
            results["gt_text"].extend(batch_gt[:used_bs])
            results["pred_text"].extend(preds)
            results["inference_time_sec"].extend(times)
            pbar.update(used_bs)

        pbar.close()

        print(f"[INFO] Final save → {output_parquet}")

        os.makedirs(os.path.dirname(output_parquet), exist_ok=True)
        table = pa.Table.from_pandas(pd.DataFrame(results))
        pq.write_table(table, output_parquet)

        return pd.DataFrame(results)
