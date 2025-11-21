import numpy as np
import pandas as pd
import evaluate
from jiwer import wer as jiwer_wer, cer as jiwer_cer
import json 
wer_metric = evaluate.load("wer")
cer_metric = evaluate.load("cer")

def fast_asr_metrics(df, gt_col="gt_text", pred_col="pred_text"):
    gts = df[gt_col].tolist()
    preds = df[pred_col].tolist()

    # evaluate는 전체 1번만 수행
    wer_all = 100 * wer_metric.compute(predictions=preds, references=gts)
    cer_all = 100 * cer_metric.compute(predictions=preds, references=gts)

    # sample-level은 jiwer로 초고속 처리
    sample_wers = [100 * jiwer_wer([g], [p]) for g, p in zip(gts, preds)]
    sample_cers = [100 * jiwer_cer([g], [p]) for g, p in zip(gts, preds)]

    return {
        "wer_all": wer_all,
        "cer_all": cer_all,
        "wer_mean": float(np.mean(sample_wers)),
        "wer_std": float(np.std(sample_wers)),
        "wer_median": float(np.median(sample_wers)),
        "wer_min": float(np.min(sample_wers)),
        "wer_max": float(np.max(sample_wers)),
        "cer_mean": float(np.mean(sample_cers)),
        "cer_std": float(np.std(sample_cers)),
        "cer_median": float(np.median(sample_cers)),
        "cer_min": float(np.min(sample_cers)),
        "cer_max": float(np.max(sample_cers)),
        "num_samples": len(sample_wers)
    }

if __name__ == "__main__":
    file_path = "/workspace/results/omniasr_inference/omniasr_ctc/whisper_tiny_inference/test_pred.parquet"
    save_path = file_path.replace(file_path.split("/")[-1], "metrics.json") # file_path 의 디렉토리
    df = pd.read_parquet(file_path)

    metrics = fast_asr_metrics(df, gt_col = "gt_text", pred_col = "pred_text")
    print(metrics)

    with open(save_path, "w") as f:
        json.dump(metrics, f, indent=4)
        print(f"Metrics saved to {save_path}")
