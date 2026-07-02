# AGENTS.md — TikTok Caption Removal Pipeline

## Architecture

Two isolated components:
- **`backend/`** — Flask web UI + job orchestration (local). No ML here.
- **`gpu_worker/`** — RunPod Serverless handler. PySceneDetect + PaddleOCR + ProPainter (all GPU). Deployed separately.

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
- **Scene detection** uses PySceneDetect (`ContentDetector`) inside `handler.py`, not TransNetV2. The local `backend/pipeline/` has no `scene_detect.py` module despite README mentioning it — the file doesn't exist in the current checkout. `gpu_worker/download_weights.py` (which fetches TransNetV2 weights to a Network Volume) is a leftover from an earlier architecture and isn't called by anything in the current build/deploy path.
- **ProPainter weights** are downloaded automatically on GPU worker cold start — `handler.py`'s `_download_weights()` fetches the three `.pth` files from GitHub releases into `/app/weights` inside the container. No RunPod Network Volume is required; `DEPLOY.md`'s Network Volume setup steps are stale for the current Dockerfile.
- **Video transfer: Google Drive upload, not base64** — `app.py`'s `upload_to_drive()` uploads the video via the Google Drive API (OAuth flow, token cached at `data/drive_token.json`), makes it public, and sends the resulting link as `video_download_url` in the RunPod payload. The GPU worker (`handler.py`) downloads it to `/tmp/` before processing. `handler.py` still accepts a `video_data_b64` field and a raw `video_path` field, but the live backend never sends those — treat them as legacy/manual-testing inputs only. OAuth client ID/secret are hardcoded in `app.py`, not read from `.env`.
- The status endpoint at `/api/video/<video_id>/status` hardcodes `total: 1` and simplified clip counts — it's simplified since clips are now tracked inside the GPU worker, not locally.

## Environment

- `.env` file must be in **repo root** (not `backend/`) — `app.py` loads it from `Path(__file__).parent.parent / ".env"`.
- Required env vars: `RUNPOD_API_KEY`, `RUNPOD_ENDPOINT_ID`
- `ffmpeg` must be in PATH on both the local machine and inside the GPU worker container.
