import os
import sys
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import time
import torch
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

from omnilingual_asr.models.inference.pipeline import ASRInferencePipeline


# --------------------------------------------------------
# 1) Auto Batch Size 기반 Batch Inference + Timer 추가
# --------------------------------------------------------
def transcribe_audio_batch_auto(batch_paths, pipeline, lang="kor_Hang", init_batch_size=64):
    """
    OmniASR에 대해 GPU 메모리를 보면서 자동으로 batch size 조절하여 inference.
    OOM 시 batch size 줄여 재시도.
    + inference_time_sec 추가
    """
    batch_langs = [lang] * len(batch_paths)
    batch_size = min(init_batch_size, len(batch_paths))

    while batch_size > 0:
        try:
            # ------------------------------
            # 🔥 Time measurement 시작
            # ------------------------------
            torch.cuda.synchronize()
            t0 = time.time()

            preds = pipeline.transcribe(
                batch_paths[:batch_size],
                lang=batch_langs[:batch_size],
                batch_size=batch_size
            )

            torch.cuda.synchronize()
            t1 = time.time()
            batch_time = t1 - t0
            per_sample_time = batch_time / batch_size

            return preds, [per_sample_time] * batch_size, batch_size

        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"[WARN] OOM → batch_size {batch_size} -> {batch_size // 2}")
                torch.cuda.empty_cache()
                batch_size = batch_size // 2
            else:
                print("[ERROR] Unexpected runtime error:", e)
                return [""] * batch_size, [0.0] * batch_size, batch_size

    print("[ERROR] Batch failed; returning empty predictions")
    return [""] * len(batch_paths), [0.0] * len(batch_paths), 1



# --------------------------------------------------------
# 2) Auto Batch Inference + Parquet 저장 + Resume
# --------------------------------------------------------
def run_inference_omni(model_card, csv_path, save_path=None, lang="kor_Hang", init_batch_size=64):

    pipeline = ASRInferencePipeline(model_card=model_card)
    df = pd.read_csv(csv_path)
    parquet_path = save_path.replace(".csv", ".parquet")

    # --------------------------------------------------------
    # Result dict (🔥 inference_time_sec 추가)
    # --------------------------------------------------------
    results = {
        "abs_path": [],
        "gt_text": [],
        "pred_text": [],
        "inference_time_sec": []
    }

    # --------------------------------------------------------
    # 디렉토리 생성
    # --------------------------------------------------------
    save_dir = os.path.dirname(parquet_path)
    if save_dir and not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # --------------------------------------------------------
    # Resume
    # --------------------------------------------------------
    if os.path.exists(parquet_path):
        prev_df = pq.read_table(parquet_path).to_pandas()

        results["abs_path"].extend(prev_df["abs_path"].tolist())
        results["gt_text"].extend(prev_df["gt_text"].tolist())
        results["pred_text"].extend(prev_df["pred_text"].tolist())

        # 🔥 Resume 시 inference_time_sec 컬럼 유지
        if "inference_time_sec" in prev_df.columns:
            results["inference_time_sec"].extend(prev_df["inference_time_sec"].tolist())
        else:
            results["inference_time_sec"].extend([0.0] * len(prev_df))

        done = set(prev_df["abs_path"].tolist())
        print(f"[INFO] Resumed from: {parquet_path} (already {len(done)} rows)")
    else:
        done = set()

    batch_paths = []
    batch_gt = []
    processed_count = 0

    print(f"[INFO] Total rows={len(df)}, Already done={len(done)}")

    # --------------------------------------------------------
    # Main inference loop
    # --------------------------------------------------------
    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Inference: {model_card}"):

        audio_path = row["abs_path"]
        gt_text = row["transcription"]

        if audio_path in done:
            continue

        batch_paths.append(audio_path)
        batch_gt.append(gt_text)

        if len(batch_paths) >= init_batch_size:

            preds, times, used_bs = transcribe_audio_batch_auto(
                batch_paths, pipeline, lang=lang, init_batch_size=init_batch_size
            )

            # 결과 저장
            results["abs_path"].extend(batch_paths[:used_bs])
            results["gt_text"].extend(batch_gt[:used_bs])
            results["pred_text"].extend(preds)
            results["inference_time_sec"].extend(times)

            batch_paths = batch_paths[used_bs:]
            batch_gt = batch_gt[used_bs:]
            processed_count += used_bs

            # 주기적 저장
            if processed_count % 10000 == 0:
                print(f"[INFO] Saving chunk → {parquet_path}")
                table = pa.Table.from_pandas(pd.DataFrame(results))
                pq.write_table(table, parquet_path)

    # --------------------------------------------------------
    # Final leftover 처리
    # --------------------------------------------------------
    if len(batch_paths) > 0:
        preds, times, used_bs = transcribe_audio_batch_auto(
            batch_paths, pipeline, lang=lang, init_batch_size=init_batch_size
        )

        results["abs_path"].extend(batch_paths[:used_bs])
        results["gt_text"].extend(batch_gt[:used_bs])
        results["pred_text"].extend(preds)
        results["inference_time_sec"].extend(times)

    # --------------------------------------------------------
    # Final save
    # --------------------------------------------------------
    print(f"[INFO] Final save → {parquet_path}")
    table = pa.Table.from_pandas(pd.DataFrame(results))
    pq.write_table(table, parquet_path)

    return pd.DataFrame(results)



# --------------------------------------------------------
# 3) 실행 코드
# --------------------------------------------------------
if __name__ == "__main__":
    model_series = [
        'omniASR_CTC_300M',
        'omniASR_CTC_1B',
        'omniASR_CTC_3B',
        'omniASR_CTC_7B'
    ]

    for model_card in model_series:
        run_inference_omni(
            model_card=model_card,
            csv_path="/workspace/kru_data/test.csv",
            save_path=f"/workspace/results/omniasr_inference/omniasr_ctc/{model_card}/test_pred.csv",
            lang="kor_Hang",
            init_batch_size=64
        )
