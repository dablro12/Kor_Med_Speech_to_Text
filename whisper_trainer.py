import os
import sys 
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
    ) -> None:
        self.model_id = model_id
        self.language = language
        self.task = task
        self.train_csv_path = train_csv_path
        self.eval_csv_path = eval_csv_path
        self.sampling_rate = sampling_rate
        self.output_dir = output_dir

        self.processor = WhisperProcessor.from_pretrained(
            self.model_id,
            language=self.language,
            task=self.task,
        )
        self.feature_extractor = self.processor.feature_extractor
        self.tokenizer = self.processor.tokenizer

        self.train_dataset = self._build_dataset(self.train_csv_path)
        self.eval_dataset = self._build_dataset(self.eval_csv_path)
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
        return Seq2SeqTrainingArguments(
            output_dir=self.output_dir,
            per_device_train_batch_size=24,
            gradient_accumulation_steps=1,
            learning_rate=1e-5,
            warmup_steps=500,
            max_steps=5000,
            fp16=False,
            bf16=True,
            gradient_checkpointing=False,
            eval_strategy="steps",
            per_device_eval_batch_size=8,
            predict_with_generate=True,
            generation_max_length=225,
            save_steps=1000,
            eval_steps=1000,
            logging_steps=25,
            report_to=["wandb"],
            load_best_model_at_end=True,
            metric_for_best_model="cer",
            greater_is_better=False,
            push_to_hub=False,
        )

    def _build_trainer(self) -> Seq2SeqTrainer:
        return Seq2SeqTrainer(
            args=self.training_args,
            model=self.model,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            data_collator=self.data_collator,
            compute_metrics=compute_metrics,
            tokenizer=self.processor.tokenizer,
        )

    def train(self) -> Seq2SeqTrainer:
        self.trainer.train()
        return self.trainer

if __name__ == "__main__":
    whisper_trainer = WhisperTrainer(
        model_id="openai/whisper-small",
        language="ko",
        task="transcribe",
        train_csv_path="/workspace/kor_med_stt_data/kru_data/train.csv",
        eval_csv_path="/workspace/kor_med_stt_data/kru_data/test.csv",
        sampling_rate=16000,
        output_dir="/workspace/kor_med_stt_data/kru_data/results/whisper-small-ko",
    )
    
    whisper_trainer.train() # Training 
