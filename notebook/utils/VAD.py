# notebook/utils/VAD.py
import os
import torch
import numpy as np
import librosa
import soundfile as sf
from typing import Any, List, Dict, Tuple
import torch
import librosa
import numpy as np

class SieroVAD:
    def __init__(
        self,
        sampling_rate: int = 16000,
        device: str = "cpu",
        threshold: float = 0.5,
        min_speech_ms: int = 250,
        min_silence_ms: int = 100,
        pad_ms: int = 30,
        max_speech_s: float = float("inf"),
    ):
        """
        Args:
            threshold: 0~1, 높을수록 'speech' 판정이 엄격 (잡음 많은 환경은 0.6~0.8 추천)
            min_speech_ms: 이보다 짧은 발화 구간은 제거 (짧은 잡음 제거용)
            min_silence_ms: 이보다 짧은 무음은 발화로 붙여서 하나의 segment로 합침
            pad_ms: segment 앞뒤로 padding(초 단위) 추가 (단어 앞/뒤 잘림 방지)
            max_speech_s: segment 길이 상한 (inf면 제한 없음)
        """
        self.sampling_rate = sampling_rate
        self.device = torch.device(device)

        self.threshold = float(threshold)
        self.min_speech_ms = int(min_speech_ms)
        self.min_silence_ms = int(min_silence_ms)
        self.pad_ms = int(pad_ms)
        self.max_speech_s = float(max_speech_s)

        self.model, self.utils = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            trust_repo=True,
        )
        self.model.to(self.device).eval()

        (self.get_speech_timestamps,
         self.save_audio,
         self.read_audio,
         self.VADIterator,
         self.collect_chunks) = self.utils

    def get_segments_from_vad(self, wav_np, vad_results, sr):
        segments = []
        for seg in vad_results:
            # seg['start'], seg['end']는 seconds
            start_sample = int(seg["start"] * sr)
            end_sample = int(seg["end"] * sr)
            segments.append(wav_np[start_sample:end_sample])
        return segments

    def run_array(self, wav_np: np.ndarray, sr: int):
        if sr != self.sampling_rate:
            wav_np = librosa.resample(wav_np, orig_sr=sr, target_sr=self.sampling_rate)
            sr = self.sampling_rate
        if wav_np.ndim > 1:
            wav_np = wav_np.mean(axis=0)
        wav_np = wav_np.astype(np.float32)

        # ✅ 핵심: 텐서로 만들고 device로 올리기
        wav_t = torch.from_numpy(wav_np).to(self.device)

        speech_timestamps = self.get_speech_timestamps(
            wav_t,                 # ✅ numpy 대신 tensor
            self.model,
            sampling_rate=sr,
            threshold=self.threshold,
            min_speech_duration_ms=self.min_speech_ms,
            min_silence_duration_ms=self.min_silence_ms,
            speech_pad_ms=self.pad_ms,
            max_speech_duration_s=self.max_speech_s,
            return_seconds=True,
        )

        vad_results = [{"start": seg["start"], "end": seg["end"]} for seg in speech_timestamps]

        # segment 추출은 numpy가 편하니 cpu로 가져와서 처리
        wav_cpu = wav_t.detach().cpu().numpy()
        segment_list = self.get_segments_from_vad(wav_cpu, vad_results, sr)

        return {"timestamp": vad_results, "segment_array": segment_list, "sr": sr}

    def run(self, audio_path: str):
        wav_np, sr = librosa.load(audio_path, sr=None)  # 원본 sr 유지 로드
        return self.run_array(wav_np, sr)

import numpy as np
import torch
class SileroVADGate:
    def __init__(
        self,
        sampling_rate=16000,
        device="cpu",
        threshold=0.25,
        min_speech_ms=96,
        agg="max",
        hangover_ms=150,
    ):
        self.sr = sampling_rate
        self.device = torch.device(device)
        self.threshold = float(threshold)
        self.min_speech_ms = int(min_speech_ms)
        self.agg = agg
        self.hangover_ms = int(hangover_ms)

        self.model, self.utils = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            trust_repo=True,
        )
        self.model.to(self.device).eval()

        self.frame_samples = 512 if self.sr == 16000 else 256
        self.hop_samples = self.frame_samples

        frame_ms = 1000.0 * (self.frame_samples / self.sr)
        self.min_speech_frames = max(1, int(np.ceil(self.min_speech_ms / frame_ms)))
        self.hangover_frames = max(0, int(np.ceil(self.hangover_ms / frame_ms)))

        self._speech_run_frames = 0
        self._hangover_left = 0

    @torch.no_grad()
    def is_speech(self, audio_np: np.ndarray, debug: bool = False) -> Tuple[bool, float]:
        if audio_np is None or len(audio_np) < self.frame_samples:
            self._speech_run_frames = 0
            self._hangover_left = 0
            return False, 0.0

        x = audio_np.astype(np.float32)

        probs = []
        for start in range(0, len(x) - self.frame_samples + 1, self.hop_samples):
            frame = x[start:start + self.frame_samples]
            t = torch.from_numpy(frame).to(self.device).unsqueeze(0)
            p = float(self.model(t, self.sr).item())
            probs.append(p)

        if not probs:
            self._speech_run_frames = 0
            self._hangover_left = 0
            return False, 0.0

        score = float(np.mean(probs)) if self.agg == "mean" else float(np.max(probs))

        consec_frames = 0
        for p in reversed(probs):
            if p >= self.threshold:
                consec_frames += 1
            else:
                break

        if consec_frames > 0:
            self._speech_run_frames += consec_frames
        else:
            self._speech_run_frames = 0

        ok = self._speech_run_frames >= self.min_speech_frames

        if ok:
            self._hangover_left = self.hangover_frames
        else:
            if self._hangover_left > 0:
                self._hangover_left -= 1
                ok = True

        if debug:
            frame_ms = 1000.0 * (self.frame_samples / self.sr)
            print(
                f"[VAD] score={score:.3f} thr={self.threshold:.3f} "
                f"consec={consec_frames} run={self._speech_run_frames}/{self.min_speech_frames} "
                f"hang={self._hangover_left} (frame_ms={frame_ms:.1f}) -> {ok}"
            )

        return ok, score

