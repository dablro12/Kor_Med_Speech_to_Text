# ====================================================
# Base: PyTorch 2.5.1 + CUDA 12.1 + cuDNN9
# ====================================================
FROM pytorch/pytorch:2.5.1-cuda12.1-cudnn9-devel

# ====================================================
# 기본 환경 설정
# ====================================================
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Seoul
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8

# ----------------------------------------------------
# System dependencies
# ----------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    wget \
    curl \
    vim \
    unzip \
    sudo \
    ca-certificates \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    fonts-nanum \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# ====================================================
# Python 패키지 설치
# ====================================================
# Conda 초기화 및 환경 설정
RUN conda init bash && \
    echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
    echo "conda activate base" >> ~/.bashrc

# pip 업그레이드 및 패키지 설치 (conda base 환경에서 실행)
RUN /bin/bash -c "source /opt/conda/etc/profile.d/conda.sh && conda activate base && \
    pip install --upgrade pip setuptools wheel && \
    pip install \
    numpy \
    pandas \
    openpyxl \
    matplotlib \
    seaborn \
    scikit-learn \
    opencv-python \
    tqdm \
    jupyterlab \
    tensorboard \
    notebook \
    Pillow \
    ipywidgets \
    rich \
    pyarrow \
    fastapi \
    uvicorn \
    gradio \
    accelerate \
    transformers \
    datasets \
    einops \
    peft \
    evaluate \
    torchmetrics \
    opencv-python-headless"

# ====================================================
# 작업 디렉토리 및 권한
# ====================================================
WORKDIR /workspace
RUN chmod -R 777 /workspace
