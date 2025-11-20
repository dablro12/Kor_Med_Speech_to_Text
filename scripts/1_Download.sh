#!/bin/bash

# 1.Training 및 2.Validation 데이터 다운로드
# datasetkey 208: 비대면 진료를 위한 의료진 및 환자 음성

cd /home/eiden/eiden/medical_stt/data

# train dataset
aihubshell -mode d \
  -datasetkey 208 \
  -filekey 48745,48746,48747,48748,48749,48750,48751,48752,48753,48754,48755,48756,48757,48758 \
  -aihubapikey "$aihubshell_api_key"

# validation dataset
aihubshell -mode d \
  -datasetkey 208 \
  -filekey 48759,48760,48761,48762 \
  -aihubapikey "$aihubshell_api_key"