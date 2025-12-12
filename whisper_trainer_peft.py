# whisper_trainer_peft.py
import os
import sys
import time
from datetime import timedelta
import torch

# LoRA Fine-tuning
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from peft.utils import TaskType
from transformers import BitsAndBytesConfig

os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["NCCL_IB_DISABLE"] = "1"

from transformers import (
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    WhisperForConditionalGeneration,
    WhisperProcessor,
)

from transformers import Seq2SeqTrainer, TrainerCallback, TrainingArguments, TrainerState, TrainerControl
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR

sys.path.append("/workspace/notebook")
from utils.data_dataset import KruWhisperDataset, DataCollatorSpeechSeq2SeqWithPadding
from utils.metrics import compute_metrics

import librosa
from transformers import WhisperForConditionalGeneration


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
        output_dir="/workspace/kor_med_stt_data/results/whisper_train_lora/whisper-small-ko",
        logging_dir="/workspace/kor_med_stt_data/results/whisper_train_lora/whisper-small-ko/logs",
        do_eval=False,
        train_epochs=5,
        init_batch_size=64,      # ⭐ 추가됨
        load_in_8bit=True,
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
        self.train_epochs = train_epochs
        self.processor = WhisperProcessor.from_pretrained(model_id, language=language, task=task)
        self.feature_extractor = self.processor.feature_extractor
        self.tokenizer = self.processor.tokenizer

        self.train_dataset = self._build_dataset(train_csv_path)
        self.eval_dataset = self._build_dataset(eval_csv_path) if do_eval else None

        # --- Patch: Fix KeyError 'labels' (ensure labels are present) ---
        # If labels are missing in dataset items, add them as BOS token.
        # This handles the "features" each batch is produced from.
        self.data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=self.processor)
        self.load_in_8bit = load_in_8bit
        self.model = self._build_lora_model(model_id, load_in_8bit=self.load_in_8bit) # LoRA Finetuning with 8Bit Quantization

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

    def _build_lora_model(self, model_id, load_in_8bit = False):

        model = WhisperForConditionalGeneration.from_pretrained(
            model_id,
            load_in_8bit=load_in_8bit,
            device_map="auto",
        )

        # 2) kbit 학습 준비 (중요)
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=False)
        
        def make_inputs_require_grad(module, input, output):
            output.requires_grad_(True)

        model.model.encoder.conv1.register_forward_hook(make_inputs_require_grad) # 기능 설명 : 입력 텐서를 그래디언트 계산에 포함시키는 함수
        # 3) LoRA 설정 (Whisper에서 가장 보편: q_proj, v_proj)
        peft_config = LoraConfig(
            r=32, # rank 32 is recommended for Whisper
            lora_alpha=64,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=0.05,
            bias="none",
        )

        # 4) LoRA 장착
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()

        # (권장) 학습 시 cache 끄기
        model.config.use_cache = False
        return model
        
    def _build_dataset(self, csv_path):
        return KruWhisperDataset(
            csv_path=csv_path,
            feature_extractor=self.feature_extractor,
            tokenizer=self.tokenizer,
            sampling_rate=self.sampling_rate
        )

    def _build_training_args(self, batch_size):
        use_bf16 = torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8

        return Seq2SeqTrainingArguments(
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            gradient_accumulation_steps=1,
            learning_rate=1e-5,
            warmup_steps=100,
            num_train_epochs=self.train_epochs,
            save_strategy="epoch",
            eval_strategy="no", 
            logging_strategy="epoch",
            bf16=use_bf16,
            fp16=not use_bf16,
            gradient_checkpointing=False,
            predict_with_generate=True,
            remove_unused_columns=False, #
            report_to=["tensorboard"],
            load_best_model_at_end=False,
            label_names=["labels"],
            output_dir=self.output_dir,
            logging_dir=self.logging_dir,
        )

    def _build_trainer(self):
        return Seq2SeqTrainer(
            args=self.training_args,
            model=self.model,
            train_dataset=self.train_dataset,
            data_collator=self.data_collator,
            compute_metrics=None if not self.do_eval else compute_metrics,
            tokenizer=self.processor.tokenizer,
            callbacks=[self.SavePeftModelCallback]
        )
    # This callback helps to save only the adapter weights and remove the base model weights.
    class SavePeftModelCallback(TrainerCallback):
        def on_save(
            self,
            args: TrainingArguments,
            state: TrainerState,
            control: TrainerControl,
            **kwargs,
        ):
            checkpoint_folder = os.path.join(args.output_dir, f"{PREFIX_CHECKPOINT_DIR}-{state.global_step}")

            peft_model_path = os.path.join(checkpoint_folder, "adapter_model")
            kwargs["model"].save_pretrained(peft_model_path)

            pytorch_model_path = os.path.join(checkpoint_folder, "pytorch_model.bin")
            if os.path.exists(pytorch_model_path):
                os.remove(pytorch_model_path)
            return control


    def train(self):
        self.trainer.train()
        self.trainer.save_model(self.output_dir)
        self.processor.save_pretrained(self.output_dir)

import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Whisper model.")
    parser.add_argument("--model_name", type=str, default="tiny", help="Name of the Whisper model (e.g., 'tiny', 'base', etc.)")
    parser.add_argument("--init_batch_size", type=int, default=16, help="Initial batch size for training")
    parser.add_argument("--do_eval", action='store_true', help="Whether to run evaluation during training")
    parser.add_argument("--output_dir", type=str, default=None, help="Directory for training outputs")
    parser.add_argument("--logging_dir", type=str, default=None, help="Directory for training logs")
    parser.add_argument("--train_csv_path", type=str, default=None, help="Path to the training CSV file")
    parser.add_argument("--eval_csv_path", type=str, default=None, help="Path to the evaluation CSV file")
    parser.add_argument("--load_in_8bit", action='store_true', help="Whether to load the model in 8bit")
    parser.add_argument("--train_epochs", type=int, default=5, help="Number of training epochs")
    args = parser.parse_args()

    model_name = args.model_name

    output_dir = args.output_dir
    logging_dir = args.logging_dir
    train_epochs = args.train_epochs
    train_csv_path = args.train_csv_path
    eval_csv_path = args.eval_csv_path
    whisper_trainer = WhisperTrainer(
        model_id=f"openai/whisper-{model_name}",
        train_csv_path=train_csv_path,
        eval_csv_path=eval_csv_path,
        train_epochs=train_epochs,
        init_batch_size=args.init_batch_size,
        do_eval=args.do_eval,
        load_in_8bit=args.load_in_8bit,
        output_dir=output_dir,
        logging_dir=logging_dir,
    )

    start = time.time()
    whisper_trainer.train()
    elapsed = timedelta(seconds=(time.time() - start))

    log_path = f"{whisper_trainer.output_dir}/process"
    os.makedirs(log_path, exist_ok=True)

    with open(f"{log_path}/time.log", "a") as f:
        f.write(f"Training Time: {elapsed}\n")

    print(f"[DONE] {model_name} completed. Time: {elapsed}")
