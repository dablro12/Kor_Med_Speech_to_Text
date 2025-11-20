
import os
import time
from datetime import timedelta

import pandas as pd
import torch
from transformers import (
    AutoModelForSpeechSeq2Seq,
    AutoProcessor,
    pipeline as asr_pipeline,
)

# Environment configuration
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("NCCL_P2P_DISABLE", "1")
os.environ.setdefault("NCCL_IB_DISABLE", "1")


MODEL_SERIES = ["tiny", "base", "small", "medium", "large", "turbo"]
CSV_PATH = "/workspace/kru_data/test.csv"
AUDIO_COLUMN = "abs_path"
TRANSCRIPT_COLUMN = "transcription"
RESULTS_DIR = "/workspace/results/whisper_tester"
os.makedirs(RESULTS_DIR, exist_ok=True)

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if torch.cuda.is_available() else torch.float32


def load_dataset(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    missing_cols = [col for col in (AUDIO_COLUMN, TRANSCRIPT_COLUMN) if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns in {csv_path}: {missing_cols}")
    return df


def benchmark_model(model_id: str, df: pd.DataFrame) -> tuple[float, int]:
    print(f"\nEvaluating model: {model_id}")
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id,
        torch_dtype=DTYPE,
        low_cpu_mem_usage=True,
        use_safetensors=True,
    ).to(DEVICE)

    processor = AutoProcessor.from_pretrained(model_id)
    asr = asr_pipeline(
        task="automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        torch_dtype=DTYPE,
        device=DEVICE,
    )

    total_time = 0.0
    processed = 0
    for _, row in df.iterrows():
        audio_path = row[AUDIO_COLUMN]
        if not os.path.exists(audio_path):
            print(f"[WARN] Missing audio file: {audio_path}")
            continue

        start = time.time()
        _ = asr(audio_path)
        total_time += time.time() - start
        processed += 1

    average = total_time / processed if processed else float("nan")
    print(f"Average inference time ({model_id}): {average:.4f}s over {processed} samples")
    return average, processed


def log_results(model_id: str, avg_time: float, processed: int, elapsed: float) -> None:
    process_dir = os.path.join(RESULTS_DIR, model_id.replace("/", "_"))
    os.makedirs(process_dir, exist_ok=True)
    log_path = os.path.join(process_dir, "benchmark.log")
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(f"Average inference time: {avg_time:.4f}s\n")
        f.write(f"Processed samples: {processed}\n")
        f.write(f"Total elapsed time: {elapsed:.2f}s ({timedelta(seconds=elapsed)})\n")


def main() -> None:
    df = load_dataset(CSV_PATH)
    for model_name in MODEL_SERIES:
        model_id = f"openai/whisper-{model_name}"
        start_time = time.time()
        avg_time, processed = benchmark_model(model_id, df)
        elapsed = time.time() - start_time
        log_results(model_id, avg_time, processed, elapsed)


if __name__ == "__main__":
    main()
