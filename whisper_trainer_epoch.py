import os
import sys
import time
from datetime import timedelta
import torch

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["NCCL_IB_DISABLE"] = "1"

from transformers import (
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    WhisperForConditionalGeneration,
    WhisperProcessor,
)

sys.path.append("/workspace/notebook")
from utils.data_dataset import KruWhisperDataset, DataCollatorSpeechSeq2SeqWithPadding
from utils.metrics import compute_metrics

import librosa

def find_max_batch_size(model_id, processor, dataset, start_bs=16, min_bs=1, sampling_rate=16000):
    """
    GPU VRAM 한도 내에서 실제 학습 가능한 batch size 자동 탐색
    ★ 학습 모델과 분리된 별도 모델로 테스트(backward)
    """

    # -----------------------------
    # 1) 테스트용 임시 모델 생성 (학습모델과 완전히 분리)
    # -----------------------------
    test_model = WhisperForConditionalGeneration.from_pretrained(model_id).to("cuda")
    test_model.train()

    # -----------------------------
    # 2) 가장 긴 오디오 기준으로 mel 생성
    # -----------------------------
    first = dataset[0]
    audio_path = first.get("abs_path") or first.get("audio_path")
    if not audio_path:
        raise KeyError("Dataset missing 'abs_path' or 'audio_path'")

    audio, _ = librosa.load(audio_path, sr=sampling_rate)
    feat = processor.feature_extractor(
        audio, sampling_rate=sampling_rate, return_tensors="pt"
    ).input_features.to("cuda")

    dummy_label = torch.tensor(
        [[processor.tokenizer.bos_token_id]],
        dtype=torch.long,
        device="cuda"
    )

    # -----------------------------
    # 3) backward 테스트 (optimizer 없음)
    # -----------------------------
    print(f"[INFO] Auto batch scaling start (requested={start_bs})")
    batch_size = start_bs

    while batch_size >= min_bs:
        try:
            torch.cuda.empty_cache()

            dummy_in = feat.repeat(batch_size, 1, 1)
            dummy_lab = dummy_label.repeat(batch_size, 1)

            test_model.zero_grad(set_to_none=True)

            out = test_model(input_features=dummy_in, labels=dummy_lab)
            loss = out.loss

            loss.backward()  # only once

            print(f"[OK] TRAIN batch_size {batch_size} fits GPU memory")

            # cleanup
            del test_model
            torch.cuda.empty_cache()
            return batch_size

        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"[WARN] OOM → {batch_size} → {batch_size//2}")
                batch_size //= 2
                torch.cuda.empty_cache()
            else:
                raise e

    print("[ERROR] Default batch size = 1")
    del test_model
    torch.cuda.empty_cache()
    return 1



class WhisperTrainer:
    def __init__(
        self,
        model_id="openai/whisper-small",
        language="ko",
        task="transcribe",
        train_csv_path="/workspace/kru_data/train.csv",
        eval_csv_path="/workspace/kru_data/test.csv",
        sampling_rate=16000,
        output_dir="/workspace/kru_data/results/whisper-small-ko",
        logging_dir="/workspace/logs",
        do_eval=False,
        init_batch_size=64,      # ⭐ 추가됨
    ):
        self.model_id = model_id
        self.language = language
        self.task = task
        self.output_dir = output_dir
        self.logging_dir = logging_dir
        self.train_csv_path = train_csv_path
        self.eval_csv_path = eval_csv_path
        self.do_eval = do_eval
        self.sampling_rate = sampling_rate

        self.processor = WhisperProcessor.from_pretrained(model_id, language=language, task=task)
        self.feature_extractor = self.processor.feature_extractor
        self.tokenizer = self.processor.tokenizer

        self.train_dataset = self._build_dataset(train_csv_path)
        self.eval_dataset = self._build_dataset(eval_csv_path) if do_eval else None
        self.data_collator = DataCollatorSpeechSeq2SeqWithPadding(
            processor=self.processor, decoder_start_token_id=self.tokenizer.bos_token_id
        )

        self.model = WhisperForConditionalGeneration.from_pretrained(model_id).to("cuda")

        # 🔥 GPU 메모리 기반 배치 사이즈 최종 결정
        auto_bs = find_max_batch_size(
            model_id=self.model_id,
            processor=self.processor,
            dataset=self.train_dataset,
            start_bs=init_batch_size
        )

        print(f"[INFO] Selected Batch Size: {auto_bs}")

        self.training_args = self._build_training_args(auto_bs)
        self.trainer = self._build_trainer()

    def _build_dataset(self, csv_path):
        return KruWhisperDataset(
            csv_path=csv_path,
            feature_extractor=self.feature_extractor,
            tokenizer=self.tokenizer,
            sampling_rate=self.sampling_rate
        )

    def _build_training_args(self, batch_size):
        return Seq2SeqTrainingArguments(
            output_dir=self.output_dir,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            gradient_accumulation_steps=1,
            learning_rate=1e-5,
            warmup_steps=100,
            num_train_epochs=5,
            save_strategy="epoch",
            eval_strategy="no",
            logging_strategy="steps",
            bf16=True,
            gradient_checkpointing=False,
            predict_with_generate=True,
            logging_steps=25,
            report_to=["tensorboard"],
            load_best_model_at_end=False,
        )

    def _build_trainer(self):
        return Seq2SeqTrainer(
            args=self.training_args,
            model=self.model,
            train_dataset=self.train_dataset,
            data_collator=self.data_collator,
            compute_metrics=None if not self.do_eval else compute_metrics,
            tokenizer=self.processor.tokenizer,
        )

    def train(self):
        self.trainer.train()


if __name__ == "__main__":
    model_series = ['tiny', 'base', 'small', 'medium']

    for model_name in model_series:
        whisper_trainer = WhisperTrainer(
            model_id=f"openai/whisper-{model_name}",
            init_batch_size=1024,       # ⭐ 여기서 조정 가능
            do_eval=False,
            output_dir=f"/workspace/results/whisper_train/whisper-{model_name}",
            logging_dir=f"/workspace/logs/whisper-{model_name}"
        )

        start = time.time()
        whisper_trainer.train()
        elapsed = timedelta(seconds=(time.time() - start))

        log_path = f"{whisper_trainer.output_dir}/process"
        os.makedirs(log_path, exist_ok=True)

        with open(f"{log_path}/time.log", "a") as f:
            f.write(f"Training Time: {elapsed}\n")

        print(f"[DONE] {model_name} completed. Time: {elapsed}")
