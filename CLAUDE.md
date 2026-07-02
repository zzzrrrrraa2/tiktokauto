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
```

**GPU worker deploys via GitHub Actions, not local Docker.** Any push touching `gpu_worker/**` triggers
`.github/workflows/build.yml`, which builds the image and pushes it to Docker Hub as
`shokiii/caption-removal-worker:<tag>` (needs the `DOCKERHUB_TOKEN` repo secret). The tag is **hardcoded
in the workflow** (currently `v9`) — to ship a new image version, bump the tag in `build.yml` *and*
update the RunPod Serverless endpoint to reference the new tag. A local `docker build` in `gpu_worker/`
works too if Docker is available, but the CI path is how this repo actually deploys.

There is no test suite or linter configured in this repo; the only CI is the image build above.

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
time as a safety net (it's a no-op when the files exist). `DEPLOY.md` says the opposite (weights fetched on
cold start) — the Dockerfile is the source of truth.

## GPU worker processing details

- ProPainter runs on a **crop around the ROI** (`compute_crop_box`, ROI + 96 px margin, dims multiple of 8),
  not the full frame — the inpainted crop is composited back. Keep `CROP_MARGIN` comfortably larger than
  total mask dilation.
- Clips are split **re-encoded** (`libx264`, CFR) rather than stream-copied, so cuts are frame-accurate and
  the concat demuxer is safe. Audio is dropped at split time and muxed back from the original in the final
  step (`mux_audio`).
- **OCR → mask is per-word max-extent via a sliding temporal window.** `run_ocr()` OCRs **every** frame's
  BGR ROI crop (`OCR_STEP = 1`) via `ocr.predict()` (PaddleOCR 3.x API — no `cls`/`use_angle_cls`/`use_gpu`
  params, results read from `res["dt_polys"]`). Each detected polygon is scaled by `OVERSHOOT_SCALE`
  (1.15×) around its **own** bbox center (captions pop in ~0%→110%→100%), and a frame's mask is the union
  of detections within `MASK_WINDOW` (±5) frames. Rationale: captions are one-word pop-ins living ~6-12
  frames @ 30fps, so every frame of a word's life — including pop-in/out frames OCR can't see — is within
  the window of its peak-size detection and gets masked at the word's biggest extent, while short words
  ("at") get small holes, long words ("readiness") wide ones, and caption-free stretches no mask at all.
  Clips with no detected text anywhere skip ProPainter entirely (`has_text` check in `process_clip`).
- **PaddleOCR CPU fallback**: if Paddle GPU construction/inference throws (e.g. wheel lacks CUDA kernels
  for the GPU model RunPod scheduled), `_ocr_predict()` rebuilds the OCR singleton on CPU and continues —
  slower, but the job survives. `gpu_diagnostic()` logs the GPU model, compute capability, and a torch
  CUDA kernel check at the start of every job; check job logs for this when debugging GPU issues.
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

`config.yaml` (checked in; its header comment says "copy this to config.yaml" but it *is* the live config)
controls `worker.concurrency`, `worker.timeout_per_clip`, `scene_detection.threshold`/`min_scene_length`,
`processing.mask_dilation`, and `server.port`/`host`/`debug`.

## Other quirks

- **Trust the code over the prose docs.** `README.md`, `AGENTS.md`, and `DEPLOY.md` all still describe the
  old transfer flow (public Drive download URL / `video_download_url`) instead of the current private-file
  `video_file_id` + `drive_access_token` flow, and `README.md`/`DEPLOY.md` reference
  `gpu_worker/download_weights.py`, which does not exist in the checkout.
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
