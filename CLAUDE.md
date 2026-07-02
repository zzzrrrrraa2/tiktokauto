# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A pipeline that removes captions/subtitles from video: a user uploads a video, draws a bounding box over
the caption region, and a GPU worker detects text in that region (OCR) and inpaints it out frame-by-frame.

Two isolated components, deployed separately:
- **`backend/`** — Flask web UI + job orchestration. Runs locally. No ML/GPU code here.
- **`gpu_worker/`** — RunPod Serverless handler. Runs scene detection, OCR, and inpainting. Deployed as a
  Docker container to RunPod.

The backend sends the **entire uploaded video** to the GPU worker as a single RunPod job. The GPU worker
(`gpu_worker/handler.py`) does everything else internally: scene detection, splitting into clips, per-clip
OCR + mask generation + inpainting, and re-concatenation. `backend/pipeline/ffmpeg_utils.py` exists but its
split/concat functions are **not** used by the current `app.py` flow — that logic is duplicated inside
`handler.py` and runs GPU-side.

## Commands

```bash
# Backend (local)
cd backend
pip install -r requirements.txt
python app.py                  # http://localhost:5000, config from ../config.yaml

# GPU Worker — build and push to a registry, then create a RunPod Serverless endpoint from it
cd gpu_worker
docker build -t caption-removal-worker:v1 .
docker tag caption-removal-worker:v1 <registry>/caption-removal-worker:v1
docker push <registry>/caption-removal-worker:v1
```

There is no test suite, linter, or CI configured in this repo.

## Video transfer: private Google Drive files + short-lived OAuth token

`app.py`'s `process_video_pipeline()` uploads the source video to Google Drive as a **private** file
(`upload_to_drive()`), then sends the GPU worker `video_file_id` plus `drive_access_token` (the backend's
OAuth access token, force-refreshed at pipeline start so it's valid for ~1h). The worker downloads the input
via the Drive REST API, and at the end **uploads the result back to Drive** and returns `result_file_id`.
The backend downloads that file to `data/final/<video_id>_final.mp4` and best-effort deletes both Drive
files. Nothing is made publicly readable anymore, and results of any size avoid RunPod's 20 MB response
limit.

`handler.py` also still supports `video_download_url`, `video_data_b64` and local `video_path` inputs for
manual testing. Without a `drive_access_token`, results ≤15 MB are returned inline as `result_b64`; larger
results without a token are an error.

OAuth client ID/secret for the Drive upload are hardcoded in `app.py` (not read from `.env`); the token is
cached at `data/drive_token.json` after the first interactive consent (`flow.run_local_server(port=8080)`).
Jobs longer than ~1h will fail at the result-upload step when the access token expires.

## Scene detection: PySceneDetect, not TransNetV2

Despite what `README.md` says, `handler.py`'s `detect_scenes()` uses **PySceneDetect**
(`scenedetect.detectors.ContentDetector`), not TransNetV2. Scenes longer than
`processing.clip_max_duration` (config.yaml) are subdivided so per-clip GPU memory stays bounded.
`backend/pipeline/scene_detect.py` referenced in `README.md`'s directory tree does not exist.

## GPU worker weight loading

The `Dockerfile` bakes the three ProPainter `.pth` files **and** the PaddleOCR models into the image at
build time — no Network Volume, no cold-start downloads. `handler.py` keeps `_download_weights()` at import
time as a safety net (it's a no-op when the files exist). `DEPLOY.md`'s "Create Network Volume" / "Populate
Weights" sections are stale.

## GPU worker processing details

- ProPainter runs on a **crop around the ROI** (`compute_crop_box`, ROI + 96 px margin, dims multiple of 8),
  not the full frame — the inpainted crop is composited back. Keep `CROP_MARGIN` comfortably larger than
  total mask dilation.
- Clips are split **re-encoded** (`libx264`, CFR) rather than stream-copied, so cuts are frame-accurate and
  the concat demuxer is safe. Audio is dropped at split time and muxed back from the original in the final
  step (`mux_audio`).
- OCR runs every 5th frame (masks are reused between keyframes), on a BGR crop, via `ocr.predict()`
  (PaddleOCR 3.x API — no `cls`/`use_angle_cls`/`use_gpu` params, results read from `res["dt_polys"]`).
  Clips with no detected text skip ProPainter entirely.
- The handler cleans `/tmp` work dirs in a `finally` block — warm workers don't accumulate disk usage.

## Configuration priority

Credentials load from **both `.env` and `config.yaml`**, with `.env` taking precedence for RunPod keys
(`app.py`):

```python
api_key=os.environ.get("RUNPOD_API_KEY", "")
endpoint_id=os.environ.get("RUNPOD_ENDPOINT_ID", config["runpod"]["endpoint_id"])
```

`.env` must live in the **repo root** (not `backend/`) — `app.py` loads it via
`Path(__file__).parent.parent / ".env"`. Required vars: `RUNPOD_API_KEY`, `RUNPOD_ENDPOINT_ID`.

`config.yaml` (copy from `config.yaml`'s own header comment / `.env.example` pattern) controls
`worker.concurrency`, `worker.timeout_per_clip`, `scene_detection.threshold`/`min_scene_length`,
`processing.mask_dilation`, and `server.port`/`host`/`debug`.

## Other quirks

- **`data/queue.db`** (SQLite, `backend/pipeline/job_queue.py`) is auto-created at runtime and persists
  between restarts. Delete it to reset all job state. It tracks per-video status and a `clips` table that is
  largely vestigial now — clip-level tracking actually happens inside the GPU worker, not in this DB.
- **`data/` subdirectories** (`uploads/`, `clips/`, `masks/`, `results/`, `final/`) are auto-created by
  `app.py` / the GPU worker at runtime.
- The `/api/video/<video_id>/status` endpoint hardcodes `total: 1` and simplified clip counts, since clip
  progress isn't visible to the local backend during a run — the whole video is one opaque RunPod job.
- `ffmpeg`/`ffprobe` must be on `PATH` locally for `backend/pipeline/ffmpeg_utils.py` (used for
  upload-time probing) — the GPU worker installs its own `ffmpeg` via the Dockerfile's `apt-get`.
- GPU worker Python/CUDA deps are version-pinned tightly and order-dependent in the `Dockerfile` (PyTorch
  cu124 wheel index, then PaddlePaddle-GPU from its own cu118 wheel index, then PaddleOCR 3.x). Recent commit
  history shows several fixes here (PaddleOCR 2.x→3.x API migration, dropping deprecated `use_gpu`,
  matplotlib/`MPLBACKEND=Agg` for headless rendering) — if you touch this file, verify the install order
  still resolves without pulling conflicting `nvidia-*` packages.
