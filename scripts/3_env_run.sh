# prod version
docker run -dit --gpus all --shm-size=64g --name kor_med_stt -v $(pwd)/:/workspace/ torch2.5.1-cuda12.1-env:latest 

# dev version
docker run -dit --gpus all --shm-size=64g --name kor_med_stt \
  -v $(pwd)/:/workspace/ \
  -v /mnt/hdd/kor_med_stt_data:/workspace/kor_med_stt_data \
  torch2.5.1-cuda12.1-env:latest