# !/bin/bash

export CUDA_VISIBLE_DEVICES=1

MODEL_SERIES="tiny base small medium large"

for model_name in $MODEL_SERIES
do
    /opt/conda/bin/python /workspace/whisper_trainer.py \
        --model_name $model_name \
        --init_batch_size 16 \
        --do_eval \
        --output_dir /workspace/kor_med_stt_data/results/whisper_train/whisper-$model_name \
        --logging_dir /workspace/kor_med_stt_data/logs/whisper-$model_name
done