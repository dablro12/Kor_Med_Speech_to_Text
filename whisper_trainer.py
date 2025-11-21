
import os
import sys
import time
from datetime import timedelta

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

class WhisperTrainer:
    def __init__(
        self,
        model_id: str = "openai/whisper-small",
        language: str = "ko",
        task: str = "transcribe",
        train_csv_path: str = "/workspace/kru_data/train.csv",
        eval_csv_path: str = "/workspace/kru_data/test.csv",
        sampling_rate: int = 16000,
        output_dir: str = "/workspace/kru_data/results/whisper-small-ko",
        logging_dir : str = "/workspace/logs",
        do_eval: bool = False,  # 평가 비활성 플래그 추가
    ) -> None:
        self.model_id = model_id
        self.language = language
        self.task = task
        self.train_csv_path = train_csv_path
        self.eval_csv_path = eval_csv_path
        self.sampling_rate = sampling_rate
        self.output_dir = output_dir
        self.logging_dir = logging_dir
        self.do_eval = do_eval  # 평가 플래그 저장

        self.processor = WhisperProcessor.from_pretrained(
            self.model_id,
            language=self.language,
            task=self.task,
        )
        self.feature_extractor = self.processor.feature_extractor
        self.tokenizer = self.processor.tokenizer

        self.train_dataset = self._build_dataset(self.train_csv_path)
        self.eval_dataset = self._build_dataset(self.eval_csv_path) if self.do_eval else None
        self.data_collator = DataCollatorSpeechSeq2SeqWithPadding(
            processor=self.processor,
            decoder_start_token_id=self.tokenizer.bos_token_id,
        )
        self.model = WhisperForConditionalGeneration.from_pretrained(self.model_id)
        self.training_args = self._build_training_args()
        self.trainer = self._build_trainer()

    def _build_dataset(self, csv_path: str) -> KruWhisperDataset:
        return KruWhisperDataset(
            csv_path=csv_path,
            feature_extractor=self.feature_extractor,
            tokenizer=self.tokenizer,
            sampling_rate=self.sampling_rate,
        )

    def _build_training_args(self) -> Seq2SeqTrainingArguments:
        # 평가를 하지 않게 하기 위해 evaluation_strategy="no"로 설정
        return Seq2SeqTrainingArguments(
            output_dir=self.output_dir,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=16,
            gradient_accumulation_steps=1,
            learning_rate=1e-5,
            warmup_steps=100,
            max_steps=10000,
            fp16=False,
            bf16=True,
            gradient_checkpointing=False,
            # eval_strategy="steps",
            eval_strategy="no",  # 평가 전략 비활성화
            predict_with_generate=True,
            generation_max_length=225,
            save_steps=2000,
            eval_steps=2000,
            logging_steps=25,
            report_to=["tensorboard"],
            # report_to=["wandb"],
            load_best_model_at_end=False,
            metric_for_best_model="cer",
            greater_is_better=False,
            push_to_hub=False,
            logging_dir = self.logging_dir, # tensor board logging dir 
        )

    def _build_trainer(self) -> Seq2SeqTrainer:
        # 평가/metrics 관련 인자도 조건에 따라 비활성화
        if self.do_eval:
            return Seq2SeqTrainer(
                args=self.training_args,
                model=self.model,
                train_dataset=self.train_dataset,
                eval_dataset=self.eval_dataset,
                data_collator=self.data_collator,
                compute_metrics=compute_metrics,
                tokenizer=self.processor.tokenizer,
            )
        else:
            return Seq2SeqTrainer(
                args=self.training_args,
                model=self.model,
                train_dataset=self.train_dataset,
                # eval_dataset=self.eval_dataset,  # 평가 데이터셋 비활성화
                data_collator=self.data_collator,
                compute_metrics=compute_metrics,  # metrics 비활성화
                tokenizer=self.processor.tokenizer,
            )

    def train(self) -> Seq2SeqTrainer:
        self.trainer.train()
        return self.trainer

def get_last_dir_name(path):
    """output_dir에서 마지막 디렉토리 이름 추출"""
    return os.path.basename(os.path.normpath(path))

if __name__ == "__main__":
    # model_series = ['tiny', 'base', 'small', 'medium', 'large', 'turbo']
    model_series = ['small', 'medium', 'large', 'turbo']
    
    for model_name in model_series:
        whisper_trainer = WhisperTrainer(
                model_id=f"openai/whisper-{model_name}",
                language="ko",
                task="transcribe",
                train_csv_path="/workspace/kru_data/train.csv",
                eval_csv_path="/workspace/kru_data/test.csv",
                sampling_rate=16000,
                output_dir=f"/workspace/results/whisper_train/whisper-{model_name}",
                logging_dir=f"/workspace/logs/whisper-{model_name}", # tensorboard logging dir 
                do_eval=False,  # 평가 끄기
            )
        
        # 훈련 시작시간 측정
        start_time = time.time()
        whisper_trainer.train() # Training
        end_time = time.time()
        elapsed = end_time - start_time

        elapsed_td = timedelta(seconds=elapsed)
        elapsed_str = str(elapsed_td)
        # 로그 폴더 만들고 파일 경로 지정
        process_dir = whisper_trainer.output_dir + "/process"
        os.makedirs(process_dir, exist_ok=True)
        log_file_path = process_dir + "/time.log"

        with open(log_file_path, "a", encoding="utf-8") as f:
            f.write(f"Training time: {elapsed:.2f} seconds\n")
            f.write(f"Training time (hh:mm:ss): {elapsed_str}\n")

    print(f"Training time: {elapsed:.2f} seconds ({elapsed_str}) - saved to {log_file_path}")
