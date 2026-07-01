# Deploy GPU Worker to RunPod

This covers deploying the video inpainting worker (PaddleOCR + ProPainter) on RunPod Serverless.

## Overview

The GPU worker receives a video clip + ROI coordinates, runs OCR to find text, generates masks, and runs ProPainter for inpainting. Results are returned to the local backend.

## Prerequisites

- RunPod account (https://runpod.io)
- API key from Settings → API Keys
- Docker installed locally (for building/pushing the image)

## Step 1: Create Network Volume

Network Volumes persist across worker restarts and hold the model weights (~500MB).

1. Go to RunPod → Storage → Network Volumes
2. Create volume:
   - **Name**: `caption-removal-weights`
   - **Region**: Pick one close to you
   - **Size**: 20 GB (minimum)
   - **Data Center**: Any with RTX 4090 availability
3. Note the Volume ID

## Step 2: Populate Weights

Spin up a temporary GPU Pod to download model weights onto the Network Volume.

1. Go to Pods → Deploy
2. Select **RTX 4090** (or any cheap GPU)
3. Template: `runpod/pytorch:2.2.0-py3.10-cuda12.1.1-devel-ubuntu22.04`
4. Attach the Network Volume, mount at `/weights`
5. Start the pod
6. SSH in or use Web Terminal:

```bash
cd /weights
git clone --depth 1 https://github.com/sczhou/ProPainter.git propainter_repo

mkdir -p propainter/model propainter/core

# Symlink model files so Python imports work from /weights/propainter/* 
ln -s /weights/propainter_repo/model/modules /weights/propainter/model/modules
ln -s /weights/propainter_repo/model/misc.py /weights/propainter/model/misc.py
ln -s /weights/propainter_repo/model/propainter.py /weights/propainter/model/propainter.py
ln -s /weights/propainter_repo/model/recurrent_flow_completion.py /weights/propainter/model/recurrent_flow_completion.py
ln -s /weights/propainter_repo/core/utils.py /weights/propainter/core/utils.py

# Download weights
wget -P /weights/propainter https://github.com/sczhou/ProPainter/releases/download/v0.1.0/ProPainter.pth
wget -P /weights/propainter https://github.com/sczhou/ProPainter/releases/download/v0.1.0/raft-things.pth
wget -P /weights/propainter https://github.com/sczhou/ProPainter/releases/download/v0.1.0/recurrent_flow_completion.pth

# Verify
ls -lh /weights/propainter/*.pth
```

7. Stop/terminate the pod (volume persists)

## Step 3: Build and Push Docker Image

```bash
cd gpu_worker

# Build
docker build -t caption-removal-worker:v1 .

# Tag for Docker Hub or any registry
docker tag caption-removal-worker:v1 yourusername/caption-removal-worker:v1

# Push
docker push yourusername/caption-removal-worker:v1
```

## Step 4: Create Serverless Endpoint

1. Go to RunPod → Serverless → New Endpoint
2. Configure:
   - **Endpoint Name**: `caption-removal`
   - **GPU Type**: RTX 4090 (24GB)
   - **Container Image**: `yourusername/caption-removal-worker:v1`
   - **Network Volume**: Attach `caption-removal-weights`, mount at `/weights`
   - **Min Workers**: 0 (scale to zero when idle)
   - **Max Workers**: 5 (concurrent clip processing)
   - **Idle Timeout**: 120 seconds
   - **Execution Timeout**: 300 seconds
   - **Max Request Size**: 500 MB
   - **Container Disk**: 30 GB
   - **Worker Memory**: 48 GB

3. Create endpoint
4. Note the Endpoint ID (e.g., `abc123def456`)

## Step 5: Configure Local Backend

Set your RunPod credentials:

```bash
# .env file (in project root)
RUNPOD_API_KEY=rpa_xxxxxxxxxxxx
RUNPOD_ENDPOINT_ID=abc123def456
```

Or edit `config.yaml`:

```yaml
runpod:
  api_key: "rpa_xxxxxxxxxxxx"
  endpoint_id: "abc123def456"
```

## Testing

```bash
# Test the endpoint directly
curl -X POST "https://api.runpod.ai/v2/YOUR_ENDPOINT_ID/run" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "input": {
      "clip_path": "/path/to/test_clip.mp4",
      "roi": {"x": 0, "y": 600, "w": 1080, "h": 120},
      "fps": 30,
      "clip_id": "test_001",
      "mask_dilation": 8
    }
  }'
```

## Pricing (Estimates)

| GPU | VRAM | Price/hr (Spot) | 60s Video Cost |
|---|---|---|---|
| RTX 4090 | 24GB | ~$0.34 | ~$0.03-0.05 |
| A100 80GB | 80GB | ~$1.15 | ~$0.10-0.15 |

RTX 4090 is sufficient for 720p-1080p. Use A100 only if processing many clips concurrently (higher throughput).
