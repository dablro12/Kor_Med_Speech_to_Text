import argparse
import os, time
from typing import Optional, Tuple

import gradio as gr
import librosa
import numpy as np
import torch
import transformers
import noisereduce as nr

from transformers import (
    WhisperForConditionalGeneration,
    WhisperProcessor,
    WhisperConfig,
)

transformers.logging.set_verbosity_error()

WINDOW_SECONDS = 4
DEFAULT_SR = 16000


def rms(x: np.ndarray) -> float:
    if x is None or len(x) == 0:
        return 0.0
    x = x.astype(np.float32)
    return float(np.sqrt(np.mean(x * x)) + 1e-12)


# =========================
# Silero VAD Gate (Streaming)
# =========================
import sys 
sys.path.append("/workspace/notebook")
from utils.VAD import SileroVADGate

# =========================
# Whisper Inference
# =========================
from utils.whisper import WhisperInference
# =========================
# Realtime App
# =========================
class WhisperRealtimeApp:
    def __init__(
        self,
        model_dir: str,
        sampling_rate: int = 16000,
        device: str = "cuda",
        language: str = "ko",
        task: str = "transcribe",
        # VAD
        use_vad: bool = True,
        vad_threshold: float = 0.3,
        vad_window_seconds: float = 0.8,
        vad_min_speech_ms: int = 96,
        hangover_ms: int = 150,
        # NR
        use_nr: bool = False,
        nr_prop_decrease: float = 1.0,
        nr_stationary: bool = False,
        nr_every_n_chunks: int = 3,
        noise_profile_seconds: float = 1.0,
        noise_rms_thresh: float = 0.01,
        # Endpoint
        end_silence_frames: int = 4,
        # safety
        max_utt_seconds: float = 20.0,
    ):
        self.sampling_rate = sampling_rate
        self.window_samples = int(self.sampling_rate * WINDOW_SECONDS)
        self.vad_window_samples = int(self.sampling_rate * vad_window_seconds)

        self.asr = WhisperInference(
            model_dir=model_dir,
            sampling_rate=sampling_rate,
            device=device,
            language=language,
            task=task,
        )

        self.use_vad = use_vad
        self.vad_gate = None
        if self.use_vad:
            vad_device = "cuda" if str(self.asr.device).startswith("cuda") else "cpu"
            self.vad_gate = SileroVADGate(
                sampling_rate=sampling_rate,
                device=vad_device,
                threshold=vad_threshold,
                min_speech_ms=vad_min_speech_ms,
                agg="max",
                hangover_ms=hangover_ms,
            )
            print(
                f"[INFO] VAD enabled thr={vad_threshold} window={vad_window_seconds}s "
                f"min_speech_ms={vad_min_speech_ms} hangover_ms={hangover_ms} device={vad_device}"
            )

        # NR settings
        self.use_nr = use_nr
        self.nr_prop_decrease = float(nr_prop_decrease)
        self.nr_stationary = bool(nr_stationary)
        self.nr_every_n_chunks = max(1, int(nr_every_n_chunks))
        self.noise_profile_seconds = float(noise_profile_seconds)
        self.noise_rms_thresh = float(noise_rms_thresh)

        if self.use_nr:
            print(
                f"[INFO] NR enabled (VAD-only) prop_decrease={self.nr_prop_decrease} "
                f"stationary={self.nr_stationary} every_n_chunks={self.nr_every_n_chunks} "
                f"profile_sec={self.noise_profile_seconds} noise_rms_thresh={self.noise_rms_thresh}"
            )

        # endpoint
        self.end_silence_frames = max(1, int(end_silence_frames))
        self.max_utt_samples = int(self.sampling_rate * float(max_utt_seconds))

    def transcribe(self, audio_16k: np.ndarray) -> Tuple[str, float]:
        return self.asr.transcribe_array(audio_16k)


def build_interface(app: WhisperRealtimeApp):
    with gr.Blocks(title="Whisper ASR (Endpointed)") as demo:
        gr.Markdown(
            f"""
# 🎙️ Whisper ASR (Endpointed)
- Streaming input
- **NR → VAD (VAD-only)**  
- **ASR is called only when endpoint is confirmed** (silence_frames)
- Keeps history (append)
"""
        )

        # ✅ state: per-session
        state = gr.State({
            "stream": None,
            "utt": None,
            "in_speech": False,
            "silence_frames": 0,
            "history": "",
            "chunk_idx": 0,
            "noise_buf": None,
            "noise_profile": None,
        })

        with gr.Row():
            audio_input = gr.Audio(
                sources=["microphone"],
                streaming=True,
                type="numpy",
                label="Microphone (streaming)",
            )
            text_output = gr.Textbox(
                label="Transcript (history)",
                placeholder="말 끝날 때마다 누적됩니다.",
                lines=12,
            )

        def process_stream(st, new_chunk: Optional[Tuple[int, np.ndarray]]):
            if st is None:
                st = {
                    "stream": None, "utt": None, "in_speech": False,
                    "silence_frames": 0, "history": "",
                    "chunk_idx": 0, "noise_buf": None, "noise_profile": None
                }

            if new_chunk is None:
                return st, st["history"]

            chunk_sr, y = new_chunk
            st["chunk_idx"] += 1

            # normalize chunk -> float32 mono
            if y.dtype.kind == "i":
                y = y.astype(np.float32) / 32768.0
            else:
                y = y.astype(np.float32)
            if y.ndim > 1:
                y = y.mean(axis=1)

            # resample to 16k
            if chunk_sr != app.sampling_rate:
                y_16k = librosa.resample(y, orig_sr=chunk_sr, target_sr=app.sampling_rate).astype(np.float32)
            else:
                y_16k = y.astype(np.float32)

            # unpack state
            stream = st["stream"]
            utt = st["utt"]
            in_speech = st["in_speech"]
            silence_frames = int(st["silence_frames"])
            history = st["history"]
            noise_buf = st["noise_buf"]
            noise_profile = st["noise_profile"]

            # accumulate stream (raw 16k)
            stream = np.concatenate([stream, y_16k]) if stream is not None else y_16k
            if len(stream) > app.window_samples * 5:
                stream = stream[-app.window_samples * 5:]

            # VAD input window
            vad_audio_raw = stream[-app.vad_window_samples:]
            vad_audio = vad_audio_raw

            # -------------------------
            # (A) noise profile build (무음 구간에서만)
            # -------------------------
            # 아주 단순/안전하게: "현재 chunk rms가 낮고, 아직 in_speech가 아니면" noise_buf에 모음
            cur_rms = rms(y_16k)
            if app.use_nr and (not in_speech) and cur_rms <= app.noise_rms_thresh:
                noise_buf = np.concatenate([noise_buf, y_16k]) if noise_buf is not None else y_16k
                target_len = int(app.sampling_rate * app.noise_profile_seconds)
                # 너무 길어지지 않게 유지
                if len(noise_buf) > target_len * 3:
                    noise_buf = noise_buf[-target_len * 3:]

                # profile 아직 없으면 일정 길이 모였을 때 고정
                if noise_profile is None and len(noise_buf) >= target_len:
                    noise_profile = noise_buf[-target_len:].copy()
                    print(f"[NR] noise_profile fixed: {len(noise_profile)} samples ({app.noise_profile_seconds}s)")

            # -------------------------
            # (B) NR throttle (VAD-only)
            # -------------------------
            if app.use_nr and (st["chunk_idx"] % app.nr_every_n_chunks == 0):
                t0 = time.time()
                try:
                    # noise_profile가 있으면 그걸 고정 noise로 사용 (가장 안정적)
                    if noise_profile is not None:
                        vad_audio = nr.reduce_noise(
                            y=vad_audio_raw,
                            y_noise=noise_profile,
                            sr=app.sampling_rate,
                            prop_decrease=app.nr_prop_decrease,
                            stationary=app.nr_stationary,
                        ).astype(np.float32)
                    else:
                        # profile 없으면 그냥 기본 NR (비싸면 skip해도 됨)
                        vad_audio = nr.reduce_noise(
                            y=vad_audio_raw,
                            sr=app.sampling_rate,
                            prop_decrease=app.nr_prop_decrease,
                            stationary=app.nr_stationary,
                        ).astype(np.float32)

                    print(f"[NR] (VAD-only) applied took={time.time()-t0:.3f}s")
                except Exception as e:
                    vad_audio = vad_audio_raw
                    print(f"[NR] failed -> bypass ({e})")

            # -------------------------
            # (C) VAD decision
            # -------------------------
            if app.use_vad and app.vad_gate is not None:
                ok, _ = app.vad_gate.is_speech(vad_audio, debug=False)
            else:
                ok = True

            # -------------------------
            # (D) Endpoint logic with silence_frames
            # -------------------------
            if ok:
                # speech ongoing
                in_speech = True
                silence_frames = 0

                utt = np.concatenate([utt, y_16k]) if utt is not None else y_16k
                if len(utt) > app.max_utt_samples:
                    utt = utt[-app.max_utt_samples:]

                st.update({
                    "stream": stream, "utt": utt, "in_speech": in_speech,
                    "silence_frames": silence_frames, "history": history,
                    "noise_buf": noise_buf, "noise_profile": noise_profile,
                })
                return st, history

            # ok == False : silence
            if in_speech:
                silence_frames += 1

                # 아직 endpoint 확정 전이면(짧은 끊김) 그냥 기다림
                if silence_frames < app.end_silence_frames:
                    st.update({
                        "stream": stream, "utt": utt, "in_speech": in_speech,
                        "silence_frames": silence_frames, "history": history,
                        "noise_buf": noise_buf, "noise_profile": noise_profile,
                    })
                    return st, history

                # ✅ endpoint confirmed: 여기서 딱 1번 ASR
                if utt is not None and len(utt) >= int(app.sampling_rate * 0.5):
                    text, lat = app.transcribe(utt)
                    if text:
                        history = (history + "\n" + text) if history else text
                    print(f"[ASR] endpoint latency={lat:.3f}s text='{text}'")

                # reset utterance
                utt = None
                in_speech = False
                silence_frames = 0

                st.update({
                    "stream": stream, "utt": utt, "in_speech": in_speech,
                    "silence_frames": silence_frames, "history": history,
                    "noise_buf": noise_buf, "noise_profile": noise_profile,
                })
                return st, history

            # not in_speech & silence: keep idle
            utt = None
            in_speech = False
            silence_frames = 0

            st.update({
                "stream": stream, "utt": utt, "in_speech": in_speech,
                "silence_frames": silence_frames, "history": history,
                "noise_buf": noise_buf, "noise_profile": noise_profile,
            })
            return st, history

        def clear_state():
            return {
                "stream": None,
                "utt": None,
                "in_speech": False,
                "silence_frames": 0,
                "history": "",
                "chunk_idx": 0,
                "noise_buf": None,
                "noise_profile": None,
            }, ""

        audio_input.stream(
            process_stream,
            inputs=[state, audio_input],
            outputs=[state, text_output],
            show_progress=False,
        )
        audio_input.clear(clear_state, outputs=[state, text_output])

    return demo


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()

    p.add_argument("--model_dir", type=str, required=True, help="모델 디렉토리 또는 허브 id (예: openai/whisper-small)")
    p.add_argument("--device", type=str, default="cuda", help="cuda / cpu")
    p.add_argument("--sampling_rate", type=int, default=16000, help="샘플링 레이트 (Hz)")
    p.add_argument("--language", type=str, default="ko", help="언어 코드")
    p.add_argument("--task", type=str, default="transcribe", help="transcribe / translate")
    p.add_argument("--port", type=int, default=7860, help="서버 포트")

    # VAD
    p.add_argument("--use_vad", action="store_true", default=True, help="VAD 사용")
    p.add_argument("--vad_threshold", type=float, default=0.3, help="VAD 임계값")
    p.add_argument("--vad_window_seconds", type=float, default=0.8, help="VAD window (sec)")
    p.add_argument("--vad_min_speech_ms", type=int, default=96, help="최소 음성 길이(ms)")
    p.add_argument("--hangover_ms", type=int, default=150, help="hangover(ms)")

    # NR
    p.add_argument("--use_nr", action="store_true", default=False, help="NR 사용 (VAD-only)")
    p.add_argument("--nr_prop_decrease", type=float, default=1.0, help="NR 강도 (0~1)")
    p.add_argument("--nr_stationary", action="store_true", default=False, help="stationary NR 모드")
    p.add_argument("--nr_every_n_chunks", type=int, default=3, help="NR 적용 주기 (N chunk마다 1회)")
    p.add_argument("--noise_profile_seconds", type=float, default=1.0, help="noise profile 길이(sec)")
    p.add_argument("--noise_rms_thresh", type=float, default=0.01, help="이 RMS 이하를 무음으로 보고 noise profile 축적")

    # Endpoint
    p.add_argument("--end_silence_frames", type=int, default=4, help="연속 무음 프레임 N번이면 endpoint 확정")

    # safety
    p.add_argument("--max_utt_seconds", type=float, default=20.0, help="발화 최대 길이(sec)")

    return p


def main():
    args = build_parser().parse_args()

    app = WhisperRealtimeApp(
        model_dir=args.model_dir,
        sampling_rate=args.sampling_rate,
        device=args.device,
        language=args.language,
        task=args.task,
        use_vad=args.use_vad,
        vad_threshold=args.vad_threshold,
        vad_window_seconds=args.vad_window_seconds,
        vad_min_speech_ms=args.vad_min_speech_ms,
        hangover_ms=args.hangover_ms,
        use_nr=args.use_nr,
        nr_prop_decrease=args.nr_prop_decrease,
        nr_stationary=args.nr_stationary,
        nr_every_n_chunks=args.nr_every_n_chunks,
        noise_profile_seconds=args.noise_profile_seconds,
        noise_rms_thresh=args.noise_rms_thresh,
        end_silence_frames=args.end_silence_frames,
        max_utt_seconds=args.max_utt_seconds,
    )

    demo = build_interface(app)
    demo.launch(server_name="0.0.0.0", server_port=args.port, share=False)

if __name__ == "__main__":
    main()
