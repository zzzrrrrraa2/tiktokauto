# Deploy GPU Worker to RunPod

This covers deploying the video inpainting worker (PaddleOCR + ProPainter) on RunPod Serverless.

## Overview

The GPU worker receives a video (as a `video_download_url`) + ROI coordinates, runs scene detection, OCR to
find text, generates masks, and runs ProPainter for inpainting. Results are returned to the local backend.

## Prerequisites

- RunPod account (https://runpod.io)
- API key from Settings → API Keys
- Docker installed locally (for building/pushing the image)

## Weights: no Network Volume needed

The current `gpu_worker/Dockerfile` builds a self-contained image (PyTorch cu124 + PaddlePaddle-GPU cu118 +
PaddleOCR, cloned ProPainter repo). It does **not** bake ProPainter's `.pth` weights into the image at build
time — instead, `handler.py`'s `_download_weights()` fetches the three ProPainter weight files from GitHub
releases into `/app/weights` automatically the first time the container cold-starts. PaddleOCR downloads its
own models on first use the same way.

This means you do **not** need to create a RunPod Network Volume or pre-populate weights before deploying —
just make sure the endpoint has outbound network access on cold start. `gpu_worker/download_weights.py` (a
script for populating a `/weights` Network Volume with ProPainter + TransNetV2 weights) is a leftover from an
earlier architecture and is not part of the current deploy path — ignore it.

## Step 1: Build and Push Docker Image

```bash
cd gpu_worker

# Build
docker build -t caption-removal-worker:v1 .

# Tag for Docker Hub or any registry
docker tag caption-removal-worker:v1 yourusername/caption-removal-worker:v1

# Push
docker push yourusername/caption-removal-worker:v1
```

## Step 2: Create Serverless Endpoint

1. Go to RunPod → Serverless → New Endpoint
2. Configure:
   - **Endpoint Name**: `caption-removal`
   - **GPU Type**: RTX 4090 (24GB)
   - **Container Image**: `yourusername/caption-removal-worker:v1`
   - **Min Workers**: 0 (scale to zero when idle)
   - **Max Workers**: 5 (concurrent clip processing)
   - **Idle Timeout**: 120 seconds
   - **Execution Timeout**: 300 seconds
   - **Max Request Size**: 500 MB
   - **Container Disk**: 30 GB
   - **Worker Memory**: 48 GB

   No Network Volume attachment is needed — see "Weights: no Network Volume needed" above.

3. Create endpoint
4. Note the Endpoint ID (e.g., `abc123def456`)

## Step 3: Configure Local Backend

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
# Test the endpoint directly — handler.py expects a video_download_url (or video_data_b64 /
# video_path for manual testing), not a per-clip path; it runs the full pipeline internally.
curl -X POST "https://api.runpod.ai/v2/YOUR_ENDPOINT_ID/run" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "input": {
      "video_download_url": "https://example.com/test_video.mp4",
      "roi": {"x": 0, "y": 600, "w": 1080, "h": 120},
      "fps": 30,
      "video_id": "test_001",
      "mask_dilation": 8,
      "scene_threshold": 27.0,
      "min_scene_length": 1.5
    }
  }'
```

## Pricing (Estimates)

| GPU | VRAM | Price/hr (Spot) | 60s Video Cost |
|---|---|---|---|
| RTX 4090 | 24GB | ~$0.34 | ~$0.03-0.05 |
| A100 80GB | 80GB | ~$1.15 | ~$0.10-0.15 |

RTX 4090 is sufficient for 720p-1080p. Use A100 only if processing many clips concurrently (higher throughput).
