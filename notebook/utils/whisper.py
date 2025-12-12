import os
import time
from typing import Optional, Tuple

import numpy as np
import librosa
import torch
from transformers import WhisperProcessor, WhisperConfig, WhisperForConditionalGeneration


class WhisperInference:
    def __init__(
        self,
        model_dir: str,
        sampling_rate: int = 16000,
        device: str = "cuda",
        language: str = "ko",
        task: str = "transcribe",
    ):
        self.model_dir = model_dir
        self.sampling_rate = int(sampling_rate)
        self.device = self._resolve_device(device)

        is_local = os.path.exists(self.model_dir)

        if is_local:
            parts = self.model_dir.rstrip("/").split("/")
            base_name = parts[-2] if len(parts) >= 2 else "whisper-small"
            self.base_model = f"openai/{base_name}"

            print(f"[Processor] from base_model: {self.base_model}")
            self.processor = WhisperProcessor.from_pretrained(
                self.base_model, language=language, task=task
            )

            print(f"[Model] local ckpt: {self.model_dir}")
            config = WhisperConfig.from_pretrained(self.base_model)
            self.model = WhisperForConditionalGeneration.from_pretrained(
                self.model_dir,
                config=config,
                local_files_only=True,
            ).to(self.device)
        else:
            self.base_model = self.model_dir

            print(f"[Processor] from hub: {self.base_model}")
            self.processor = WhisperProcessor.from_pretrained(
                self.base_model, language=language, task=task
            )

            print(f"[Model] hub: {self.model_dir}")
            self.model = WhisperForConditionalGeneration.from_pretrained(
                self.model_dir
            ).to(self.device)

        self.model.eval()
        self.forced_decoder_ids = self.processor.get_decoder_prompt_ids(
            language=language, task=task
        )

        # 동시 호출 방지 (Gradio/streaming에서 중요)
        import threading
        self._lock = threading.Lock()

    @staticmethod
    def _resolve_device(device_str: str) -> torch.device:
        if device_str.startswith("cuda") and torch.cuda.is_available():
            return torch.device(device_str)
        if device_str.startswith("cuda"):
            print("[WARN] CUDA unavailable. Falling back to CPU.")
        return torch.device("cpu")

    def _normalize_audio(self, audio: np.ndarray, sr: Optional[int]) -> np.ndarray:
        """mono float32 + (필요시) 16k resample"""
        if audio is None:
            return np.zeros(self.sampling_rate, dtype=np.float32)

        audio = np.asarray(audio)

        # int PCM -> float32 (-1~1)
        if audio.dtype.kind == "i":
            audio = audio.astype(np.float32) / 32768.0
        else:
            audio = audio.astype(np.float32)

        # mono
        if audio.ndim == 2:
            # (n, ch) or (ch, n) 모두 커버
            if audio.shape[0] <= 8 and audio.shape[0] < audio.shape[1]:
                audio = audio.mean(axis=0)   # (ch, n) -> (n,)
            else:
                audio = audio.mean(axis=1)   # (n, ch) -> (n,)
        elif audio.ndim != 1:
            raise ValueError(f"Unsupported audio shape: {audio.shape}")

        # resample
        if sr is not None and int(sr) != self.sampling_rate:
            audio = librosa.resample(audio, orig_sr=int(sr), target_sr=self.sampling_rate).astype(np.float32)

        return audio

    def transcribe_array(self, audio: np.ndarray, sr: Optional[int] = None) -> Tuple[str, float]:
        """
        audio: numpy array (mono or stereo)
        sr: audio의 원래 sampling rate (None이면 이미 16k라고 가정)
        returns: (text, latency_sec)
        """
        audio = self._normalize_audio(audio, sr)

        # 너무 짧으면 skip
        if len(audio) < int(self.sampling_rate * 0.5):
            return "", 0.0

        with self._lock:
            inputs = self.processor(
                audio,
                sampling_rate=self.sampling_rate,
                return_tensors="pt",
            )
            feats = inputs.input_features.to(self.device)

            if self.device.type == "cuda":
                torch.cuda.synchronize()
            start = time.time()

            with torch.no_grad():
                pred_ids = self.model.generate(
                    feats,
                    forced_decoder_ids=self.forced_decoder_ids,
                    num_beams=1,
                    do_sample=False,
                )

            if self.device.type == "cuda":
                torch.cuda.synchronize()
            lat = time.time() - start

            text = self.processor.batch_decode(pred_ids, skip_special_tokens=True)[0].strip()
            return text, lat

    def transcribe(
        self,
        audio_path: Optional[str] = None,
        audio_array: Optional[np.ndarray] = None,
        sr: Optional[int] = None,
        return_latency: bool = False,
    ):
        """
        - audio_array 우선, 없으면 audio_path 로드
        - return_latency=True면 (text, lat) 반환
        """
        if audio_array is None:
            if audio_path is None:
                audio_array = np.zeros(self.sampling_rate, dtype=np.float32)
                sr = self.sampling_rate
            else:
                # mono=False로 로드하면 (ch,n) 가능 -> _normalize_audio에서 처리
                audio_array, sr_loaded = librosa.load(audio_path, sr=None, mono=False)
                sr = sr_loaded

        text, lat = self.transcribe_array(audio_array, sr=sr)
        return (text, lat) if return_latency else text
