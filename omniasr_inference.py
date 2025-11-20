import os
import sys
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import pandas as pd
from omnilingual_asr.models.inference.pipeline import ASRInferencePipeline


# --------------------------------------------------------
# 1) 단일 오디오 파일 inference 함수
# --------------------------------------------------------
def transcribe_audio_omni(wav_path, pipeline, lang="kor_Hang"):
    """
    단일 wav 파일을 OmniASR로 inference
    """
    transcription = pipeline.transcribe(
        [wav_path],
        lang=[lang],
        batch_size=1
    )[0]

    return transcription


# --------------------------------------------------------
# 2) 전체 CSV inference 함수
# --------------------------------------------------------
def run_inference_omni(model_card, csv_path, save_path=None, lang="kor_Hang"):
    """
    Whisper 버전과 동일한 형태로 작동하는 OmniASR inference 함수
    """

    # Load OmniASR Pipeline
    pipeline = ASRInferencePipeline(model_card=model_card)

    # Load CSV
    df = pd.read_csv(csv_path)

    # Prepare results
    results = {
        "abs_path": [],
        "gt_text": [],
        "pred_text": [],
    }

    # Create save directory
    save_dir = os.path.dirname(save_path)
    if save_dir and not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Resume capability (이미 처리한 파일 스킵)
    if save_path and os.path.exists(save_path):
        prev_df = pd.read_csv(save_path)
        done = set(prev_df["abs_path"].tolist())

        results["abs_path"].extend(prev_df["abs_path"].tolist())
        results["gt_text"].extend(prev_df["gt_text"].tolist())
        results["pred_text"].extend(prev_df["pred_text"].tolist())

        print(f"Resuming from previous checkpoint: {save_path}")
    else:
        done = set()

    # --------------------------------------------------------
    # Inference Loop
    # --------------------------------------------------------
    for _, row in df.iterrows():
        audio_path = row["abs_path"]
        gt_text = row["transcription"]

        # Skip if already inferred
        if audio_path in done:
            continue

        try:
            pred = transcribe_audio_omni(audio_path, pipeline, lang=lang)
        except Exception as e:
            print(f"[Error] {audio_path} : {e}")
            pred = ""

        results["abs_path"].append(audio_path)
        results["gt_text"].append(gt_text)
        results["pred_text"].append(pred)

        # Save after each row
        if save_path:
            pd.DataFrame(results).to_csv(save_path, index=False)
            print(f"Saved progress → {save_path}")

    return pd.DataFrame(results)

# --------------------------------------------------------
# 3) Main
# --------------------------------------------------------
if __name__ == "__main__":
    model_series = ['omniASR_CTC_300M', 'omniASR_CTC_1B', 'omniASR_CTC_3B', 'omniASR_CTC_7B']
    for model_card in model_series:
        # # Train CSV inference
        # run_inference_omni(
        #     model_card=model_card,
        #     csv_path="/workspace/kru_data/train.csv",
        #     save_path=f"/workspace/results/omniasr_ctc/{model_card}/train_pred.csv",
        #     lang="kor_Hang"
        # )

        # Test CSV inference
        run_inference_omni(
            model_card=model_card,
            csv_path="/workspace/kru_data/test.csv",
            save_path=f"/workspace/results/omniasr_inference/omniasr_ctc/{model_card}/test_pred.csv",
            lang="kor_Hang"
        )
