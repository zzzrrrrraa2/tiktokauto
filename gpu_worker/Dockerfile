# GPU Worker for RunPod Serverless
# RTX 4090 / A100 compatible — all weights baked into image, no Network Volume needed
# Uses CUDA 11.8 for PaddlePaddle-GPU compatibility

FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3.10-dev \
    python3-pip \
    ffmpeg \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    wget \
    git \
    xz-utils \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1 \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1

WORKDIR /app

COPY requirements.txt .
RUN pip uninstall -y numpy || true \
    && pip install --no-cache-dir \
    torch==2.5.1 torchvision==0.20.1 \
    --index-url https://download.pytorch.org/whl/cu118
RUN pip install --no-cache-dir opencv-python==4.11.0.86
RUN pip install --no-cache-dir numpy==1.26.4
RUN pip install --no-cache-dir scipy==1.15.1
RUN pip install --no-cache-dir scikit-image==0.25.1
RUN pip install --no-cache-dir tqdm==4.67.1 requests==2.32.3 pyyaml==6.0.2
RUN pip install --no-cache-dir einops av==12.0.0
RUN pip install --no-cache-dir imageio==2.36.1 imageio-ffmpeg==0.5.1
RUN pip install --no-cache-dir "scenedetect[opencv]>=0.6" ffmpeg-python==0.2.0
RUN pip install --no-cache-dir paddlepaddle-gpu==2.6.2
RUN pip install --no-cache-dir paddleocr==2.9.1
RUN pip install --no-cache-dir runpod

RUN git clone --depth 1 https://github.com/sczhou/ProPainter.git /app/propainter_repo

COPY handler.py .

CMD ["python", "-u", "handler.py"]
