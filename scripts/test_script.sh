# /bin/bash
export CUDA_VISIBLE_DEVICES=1

input_csv="/workspace/kru_data/test.csv"
save_every=10000
batch_size=512

# base
model_root_dir="/workspace/results/whisper_train/whisper-base"
model_epochs=($(ls $model_root_dir | grep '^checkpoint-'))

for model_epoch in "${model_epochs[@]}"
do
    model_dir="${model_root_dir}/${model_epoch}"
    output_parquet="/workspace/results/whisper_test/whisper-base/${model_epoch}/test_pred.parquet"
    /opt/conda/bin/python /workspace/whisper_tester.py \
        --model_dir "$model_dir" \
        --input_csv "$input_csv" \
        --output_parquet "$output_parquet" \
        --save_every "$save_every" \
        --init_batch_size "$batch_size" \
        --gt_column "transcription" \
        --audio_column "abs_path"
done


# # small 
# model_root_dir="/workspace/results/whisper_train/whisper-small"
# model_epochs=($(ls $model_root_dir | grep '^checkpoint-'))

# for model_epoch in "${model_epochs[@]}"
# do
#     model_dir="${model_root_dir}/${model_epoch}"
#     output_parquet="/workspace/results/whisper_test/whisper-small/${model_epoch}/test_pred.parquet"
#     /opt/conda/bin/python /workspace/whisper_tester.py \
#         --model_dir "$model_dir" \
#         --input_csv "$input_csv" \
#         --output_parquet "$output_parquet" \
#         --save_every "$save_every" \
#         --init_batch_size "$batch_size" \
#         --gt_column "transcription" \
#         --audio_column "abs_path"
# done

# # medium
# model_root_dir="/workspace/results/whisper_train/whisper-medium"
# model_epochs=($(ls $model_root_dir | grep '^checkpoint-'))

# for model_epoch in "${model_epochs[@]}"
# do
#     model_dir="${model_root_dir}/${model_epoch}"
#     output_parquet="/workspace/results/whisper_test/whisper-medium/${model_epoch}/test_pred.parquet"
#     /opt/conda/bin/python /workspace/whisper_tester.py \
#         --model_dir "$model_dir" \
#         --input_csv "$input_csv" \
#         --output_parquet "$output_parquet" \
#         --save_every "$save_every" \
#         --init_batch_size "$batch_size" \
#         --gt_column "transcription" \
#         --audio_column "abs_path"
# done