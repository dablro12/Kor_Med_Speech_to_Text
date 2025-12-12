# !/bin/bash

export CUDA_VISIBLE_DEVICES=0

MODEL_SERIES="tiny base small medium large large-v3-turbo"

for model_name in $MODEL_SERIES
do
    /opt/conda/bin/python /workspace/whisper_trainer_peft.py \
        --model_name $model_name \
        --init_batch_size 160 \
        --do_eval \
        --train_epochs 5 \
        --train_csv_path /workspace/kru_data/train.csv \
        --eval_csv_path /workspace/kru_data/test.csv \
        --output_dir /workspace/kor_med_stt_data/results/whisper_train_lora/whisper-$model_name \
        --logging_dir /workspace/kor_med_stt_data/results/whisper_train_lora/whisper-$model_name/logs \
        --load_in_8bit
done