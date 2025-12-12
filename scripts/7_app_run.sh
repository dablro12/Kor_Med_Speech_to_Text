/opt/conda/bin/python whisper_app.py \
    --model_dir /workspace/kor_med_stt_data/results/whisper_train/whisper-small/checkpoint-318165 \
    --device cuda \
    --sampling_rate 16000 \
    --language ko \
    --task transcribe \
    --port 7860 \
    --use_vad \
    --vad_threshold 0.3 \
    --vad_window_seconds 0.8 \
    --vad_min_speech_ms 96 \
    --end_silence_frames 6 \
    --use_nr \
    --nr_stationary \
    --nr_every_n_chunks 5 \
    --noise_profile_seconds 1.2 \
    --noise_rms_thresh 0.01 \
    --nr_prop_decrease 1.0 \
    --hangover_ms 100



    # --model_dir openai/whisper-small \
    # --model_dir /workspace/results/whisper_train/whisper-base/checkpoint-5090605 \
