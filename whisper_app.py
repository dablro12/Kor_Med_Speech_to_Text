import argparse
import os
import time
from typing import Optional, Tuple

import gradio as gr
import librosa
import numpy as np
import torch
import transformers
from transformers import WhisperForConditionalGeneration, WhisperProcessor, pipeline

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
transformers.logging.set_verbosity_error()

# 실시간 성능 유지를 위한 고정 처리 윈도우 크기 (10초 @ 16000 Hz)
# 오디오 버퍼의 무한 증가를 막고, 낮은 지연 시간을 유지하여 응답성을 개선합니다.
WINDOW_SECONDS = 5
WINDOW_SAMPLES = 16000 * WINDOW_SECONDS

class WhisperRealtimeApp:
    def __init__(
        self,
        model_dir: str,
        sampling_rate: int = 16000,
        device: str = "cuda",
        language: str = "ko",
        task: str = "transcribe",
    ):
        self.sampling_rate = sampling_rate
        self.device = self._resolve_device(device)
        self.language = language
        self.task = task
        
        # 모델 경로 설정
        model_name = os.path.basename(os.path.dirname(model_dir)) or "whisper-small"
        hf_model_id = f"openai/{model_name}"
        model_source = model_dir if os.path.exists(model_dir) else hf_model_id
        
        print(f"[INFO] Loading model from: {model_source}")

        # 프로세서 및 모델 로드
        self.processor = WhisperProcessor.from_pretrained(
            hf_model_id, language=language, task=task
        )
        self.model = WhisperForConditionalGeneration.from_pretrained(
            model_source
            # hf_model_id
        ).to(self.device)
        self.model.eval()
        
        # 강제 디코딩 설정
        forced_ids = self.processor.tokenizer.get_decoder_prompt_ids(
            language=language, task=task
        )
        if forced_ids is not None:
            self.model.generation_config.forced_decoder_ids = forced_ids

    @staticmethod
    def _resolve_device(device_str: str) -> torch.device:
        if device_str.startswith("cuda"):
            if torch.cuda.is_available():
                return torch.device(device_str)
            print("[WARN] CUDA unavailable. Falling back to CPU.")
        return torch.device("cpu")

    def transcribe(self, audio: np.ndarray) -> Tuple[str, float]:
        """
        16000Hz mono audio array -> text, latency
        """
        if len(audio) < self.sampling_rate * 0.5: # 0.5초 미만은 스킵
            return "", 0.0

        input_features = self.processor(
            audio, sampling_rate=self.sampling_rate, return_tensors="pt"
        ).input_features.to(self.device)

        if self.device.type == "cuda":
            torch.cuda.synchronize()
        start = time.time()
        
        with torch.no_grad():
            predicted_ids = self.model.generate(input_features)

        if self.device.type == "cuda":
            torch.cuda.synchronize()
        latency = time.time() - start
        
        transcription = self.processor.batch_decode(
            predicted_ids, skip_special_tokens=True
        )[0].strip()
        
        # Streaming 모드에서는 latency를 반환할 필요가 없으므로 튜플의 첫 번째 요소만 사용
        return transcription, latency

def build_interface(app: WhisperRealtimeApp):
    # Gradio Blocks를 사용하여 UI 구성
    with gr.Blocks(title="Real-time Whisper ASR (Streaming)") as demo:
        gr.Markdown(
            f"""
            # 🎙️ Whisper ASR (실시간 스트리밍)
            **1. 마이크로 녹음** 버튼을 누르면 인식이 **실시간**으로 시작됩니다.
            **2. 녹음 중지** 버튼을 누르면 스트림이 종료됩니다.
            
            **주의**: 실시간 성능 유지를 위해 모델은 **최근 {WINDOW_SECONDS}초**의 오디오만 사용하여 인식을 수행합니다.
            """
        )
        
        # 상태 저장을 위한 State (누적된 오디오)
        state = gr.State(None) 
        
        with gr.Row():
            # 입력: 마이크 (Streaming)
            audio_input = gr.Audio(
                sources=["microphone"], 
                streaming=True, # 스트리밍 활성화
                type="numpy",
                label="1. 마이크로 녹음 (실시간 처리)",
            )
            
            # 출력: 인식된 텍스트
            text_output = gr.Textbox(
                label="2. 실시간 인식 결과", 
                placeholder="말씀하시면 실시간으로 텍스트가 표시됩니다...",
                lines=10
            )
        
        # Streaming 처리 함수
        def process_stream(stream: Optional[np.ndarray], new_chunk: Optional[Tuple[int, np.ndarray]]):
            if new_chunk is None:
                # 새로운 청크가 없으면 현재 상태 그대로 반환
                return stream, ""
            
            sr, y = new_chunk
            
            # --- Preprocessing (Gradio input -> 16kHz np.ndarray) ---
            
            # 1. 포맷 변환 (float32)
            if y.dtype.kind == 'i':
                y = y.astype(np.float32) / 32768.0
            else:
                y = y.astype(np.float32)
                
            # 2. 모노 변환
            if y.ndim > 1:
                y = y.mean(axis=1)
                
            # 3. 리샘플링 (16kHz 필수)
            if sr != app.sampling_rate:
                y = librosa.resample(y, orig_sr=sr, target_sr=app.sampling_rate)

            # 4. 스트림 누적 (Tutorial logic: concatenate)
            if stream is not None:
                stream = np.concatenate([stream, y])
            else:
                stream = y
            
            # 5. 실시간 성능 유지를 위해, 누적된 오디오 중 최근 WINDOW_SECONDS만 사용합니다.
            audio_segment_for_transcription = stream[-WINDOW_SAMPLES:]

            # 6. 추론 실행
            text, _ = app.transcribe(audio_segment_for_transcription) # latency는 무시
            
            # 7. 새로운 누적 스트림(전체)과 텍스트 반환
            return stream, text

        def clear_state():
            # 마이크가 멈추거나 리셋되었을 때 상태 초기화
            return None, ""

        # 이벤트 연결: 오디오 입력이 '스트림'되면 함수 실행
        audio_input.stream(
            process_stream,
            inputs=[state, audio_input],
            outputs=[state, text_output],
            show_progress=False # 실시간 스트리밍에서는 프로그레스 바 숨김
        )
        
        # 오디오 컴포넌트의 X 버튼 등을 눌렀을 때 상태 초기화
        audio_input.clear(clear_state, outputs=[state, text_output])

    return demo

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, default="/workspace/results/whisper_train/whisper-tiny/checkpoint-15909")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--sampling_rate", type=int, default=16000)
    parser.add_argument("--language", type=str, default="ko")
    parser.add_argument("--task", type=str, default="transcribe")
    parser.add_argument("--port", type=int, default=7860)
    return parser

def main():
    args = build_parser().parse_args()
    
    app = WhisperRealtimeApp(
        model_dir=args.model_dir,
        sampling_rate=args.sampling_rate,
        device=args.device,
        language=args.language,
        task=args.task
    )
    
    demo = build_interface(app)
    demo.launch(server_name="0.0.0.0", server_port=args.port, share=True)

if __name__ == "__main__":
    main()