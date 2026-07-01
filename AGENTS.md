# AGENTS.md — TikTok Caption Removal Pipeline

## Architecture

Two isolated components:
- **`backend/`** — Flask web UI + job orchestration (local). No ML here.
- **`gpu_worker/`** — RunPod Serverless handler. TransNetV2 + PaddleOCR + ProPainter (all GPU). Deployed separately.

The `backend/` sends the **entire video** to the GPU worker as one RunPod job — the GPU worker handles scene detection, splitting, per-clip OCR+inpainting, and concatenation internally. The local backend does none of that anymore (the `pipeline/` ffmpeg/scene_detect modules exist but the current `app.py` flow bypasses them — everything runs on the GPU side via `handler.py`).

## Commands

```bash
# Backend (local)
cd backend
pip install -r requirements.txt
python app.py                  # http://localhost:5000

# GPU Worker (RunPod — see DEPLOY.md for full deploy steps)
cd gpu_worker
docker build -t caption-removal-worker:v1 .
docker tag caption-removal-worker:v1 yourusername/caption-removal-worker:v1
docker push yourusername/caption-removal-worker:v1
```

## Configuration priority

Credentials are loaded from **both `.env` and `config.yaml`**, with `.env` taking precedence for RunPod keys:

```python
# app.py:34 — .env wins for RUNPOD_API_KEY
api_key=os.environ.get("RUNPOD_API_KEY", "")
endpoint_id=os.environ.get("RUNPOD_ENDPOINT_ID", config["runpod"]["endpoint_id"])
```

If the `.env` file is missing or has empty values, the fallback comes from `config.yaml`. Ensure at least one source has valid creds.

## Key quirks

- **`data/queue.db`** is auto-created at runtime (SQLite). It persists between restarts. Delete it to reset all jobs.
- **`data/` subdirectories** (uploads/, clips/, masks/, results/, final/) are auto-created by app.py and the GPU worker at runtime.
- **No test suite exists** — there are no unit tests, no CI, no lint/typecheck config.
- **TransNetV2** runs on the GPU worker side only (imported via pip install of the cloned repo during Docker build). The local `backend/pipeline/` has no `scene_detect.py` module despite README mentioning it — the file doesn't exist in the current checkout.
- **ProPainter weights** (~500MB) must be populated on a RunPod Network Volume before first use. See `DEPLOY.md` and `download_weights.py`.
- The GPU worker Dockerfile expects `/weights` mount point for model weights; the handler assumes ProPainter source at `/weights/propainter/repo/`.
- **Video transfer: base64 in payload** — `app.py` reads the uploaded video file, base64-encodes it, and sends it as `video_data_b64` in the RunPod payload. The GPU worker (`handler.py`) decodes it to `/tmp/` before processing. This was fixed because RunPod Serverless does NOT auto-transfer files — the old code sent a local path string that meant nothing to the worker, causing ffmpeg errors. If videos are too large for the payload, use the `video_download_url` fallback in `handler.py` (send a publicly accessible URL instead).
- The status endpoint at `/api/video/<video_id>/status` hardcodes `total: 1` and simplified clip counts — it's simplified since clips are now tracked inside the GPU worker, not locally.

## Environment

- `.env` file must be in **repo root** (not `backend/`) — `app.py` loads it from `Path(__file__).parent.parent / ".env"`.
- Required env vars: `RUNPOD_API_KEY`, `RUNPOD_ENDPOINT_ID`
- `ffmpeg` must be in PATH on both the local machine and inside the GPU worker container.
