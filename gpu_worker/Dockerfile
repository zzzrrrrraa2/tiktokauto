# GPU Worker for RunPod Serverless
# RTX 4090 / A100 compatible — all weights baked into image, no Network Volume needed
# CUDA 12.4 base + cuDNN 8 — PyTorch cu124 + PaddlePaddle cu118 (self-contained)

FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH
ENV MPLBACKEND=Agg

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
    torch==2.5.1+cu124 torchvision==0.20.1+cu124 \
    --index-url https://download.pytorch.org/whl/cu124
RUN pip install --no-cache-dir opencv-python==4.11.0.86
RUN pip install --no-cache-dir numpy==1.26.4
RUN pip install --no-cache-dir scipy==1.15.1
RUN pip install --no-cache-dir scikit-image==0.25.1
RUN pip install --no-cache-dir matplotlib tqdm==4.67.1 requests==2.32.3 pyyaml==6.0.2
RUN pip install --no-cache-dir einops av==12.0.0
RUN pip install --no-cache-dir imageio==2.36.1 imageio-ffmpeg==0.5.1
RUN pip install --no-cache-dir "scenedetect[opencv]>=0.6" ffmpeg-python==0.2.0
RUN pip install --no-cache-dir paddlepaddle-gpu==3.3.1 -i https://www.paddlepaddle.org.cn/packages/stable/cu118/
RUN pip install --no-cache-dir paddleocr==3.7.0
RUN pip install --no-cache-dir runpod

RUN git clone --depth 1 https://github.com/sczhou/ProPainter.git /app/propainter_repo

# Bake ProPainter weights into the image so cold starts don't depend on GitHub
RUN mkdir -p /app/weights/propainter \
    && wget -q -O /app/weights/propainter/ProPainter.pth \
       https://github.com/sczhou/ProPainter/releases/download/v0.1.0/ProPainter.pth \
    && wget -q -O /app/weights/propainter/raft-things.pth \
       https://github.com/sczhou/ProPainter/releases/download/v0.1.0/raft-things.pth \
    && wget -q -O /app/weights/propainter/recurrent_flow_completion.pth \
       https://github.com/sczhou/ProPainter/releases/download/v0.1.0/recurrent_flow_completion.pth

# Pre-download PaddleOCR models at build time (CPU device — no GPU during build).
# The same model files are reused by the GPU runtime, so cold starts skip the download.
RUN python -c "from paddleocr import PaddleOCR; common=dict(device='cpu', use_doc_orientation_classify=False, use_doc_unwarping=False, use_textline_orientation=False); PaddleOCR(lang='en', **common); PaddleOCR(lang='pl', **common)" \
    || echo "PaddleOCR model prefetch failed — models will download on first cold start instead"

COPY handler.py headline.py ./

CMD ["python", "-u", "handler.py"]
