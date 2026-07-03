import os
import sys
import shutil
import threading
import time
import uuid
import yaml
import requests
import io
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from dotenv import load_dotenv
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseUpload

load_dotenv(Path(__file__).parent.parent / ".env")
from flask import Flask, request, jsonify, render_template, send_from_directory, url_for
from flask_cors import CORS

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pipeline.ffmpeg_utils import (probe_video, extract_frame, split_video, concat_clips,
                                   mux_audio, extract_audio, pair_swap_order)
from pipeline.dubbing import dub_audio
from pipeline.scene_detect import detect_scenes, subdivide_segments
from pipeline.runpod_client import RunPodClient
from pipeline.job_queue import JobQueue

app = Flask(__name__)
CORS(app)

BASE_DIR = Path(os.path.dirname(os.path.abspath(__file__))).parent
DATA_DIR = BASE_DIR / "data"

with open(BASE_DIR / "config.yaml") as f:
    config = yaml.safe_load(f)

app.config["MAX_CONTENT_LENGTH"] = config["server"]["max_upload_size_mb"] * 1024 * 1024

queue = JobQueue(str(DATA_DIR / "queue.db"))

runpod = RunPodClient(
    api_key=os.environ.get("RUNPOD_API_KEY", ""),
    endpoint_id=os.environ.get("RUNPOD_ENDPOINT_ID", config["runpod"]["endpoint_id"])
)


OAUTH_CLIENT_ID = "738384489698-nipjpv09aunhmvgpv8iu509rkvee0ngs.apps.googleusercontent.com"
OAUTH_CLIENT_SECRET = "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"
TOKEN_PATH = DATA_DIR / "drive_token.json"
SCOPES = ["https://www.googleapis.com/auth/drive.file"]


def _get_drive_creds():
    creds = None
    if TOKEN_PATH.exists():
        with open(TOKEN_PATH) as f:
            creds = Credentials.from_authorized_user_info(json.load(f), SCOPES)

    # Always refresh when possible: the access token is handed to the GPU worker,
    # which needs it to stay valid for the whole job (~1h lifetime from refresh).
    if creds and creds.refresh_token:
        try:
            creds.refresh(Request())
            TOKEN_PATH.parent.mkdir(parents=True, exist_ok=True)
            with open(TOKEN_PATH, "w") as f:
                f.write(creds.to_json())
        except Exception:
            pass

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            client_config = {
                "installed": {
                    "client_id": OAUTH_CLIENT_ID,
                    "client_secret": OAUTH_CLIENT_SECRET,
                    "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                    "token_uri": "https://oauth2.googleapis.com/token",
                    "redirect_uris": ["http://localhost"]
                }
            }
            flow = InstalledAppFlow.from_client_config(client_config, SCOPES)
            creds = flow.run_local_server(port=8080, open_browser=True)

        TOKEN_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(TOKEN_PATH, "w") as f:
            f.write(creds.to_json())

    return creds


def _get_drive_service(creds=None):
    return build("drive", "v3", credentials=creds or _get_drive_creds())


def upload_to_drive(file_path: str, filename: str, creds=None) -> str:
    """Upload a file to Google Drive (kept private). Returns the file id."""
    service = _get_drive_service(creds)

    with open(file_path, "rb") as fh:
        media = MediaIoBaseUpload(io.BytesIO(fh.read()), mimetype="video/mp4", resumable=True)
        file_metadata = {"name": filename}
        uploaded = service.files().create(body=file_metadata, media_body=media, fields="id").execute()

    return uploaded["id"]


def download_from_drive(file_id: str, dest_path: str, creds):
    """Stream a Drive file to disk using the OAuth access token."""
    resp = requests.get(
        f"https://www.googleapis.com/drive/v3/files/{file_id}?alt=media",
        headers={"Authorization": f"Bearer {creds.token}"},
        stream=True, timeout=600
    )
    resp.raise_for_status()
    with open(dest_path, "wb") as f:
        for chunk in resp.iter_content(chunk_size=1024 * 1024):
            f.write(chunk)


def delete_from_drive(file_id: str, creds):
    try:
        _get_drive_service(creds).files().delete(fileId=file_id).execute()
    except Exception:
        pass  # cleanup is best-effort; orphaned files only cost Drive quota


def _job_payload(file_id: str, roi: dict, job_video_id: str, token: str) -> dict:
    """RunPod job payload — identical shape whether file_id is a whole video or
    a single clip (the worker pipeline handles both)."""
    proc = config["processing"]
    return {
        "video_file_id": file_id,
        "drive_access_token": token,
        "roi": roi,
        "video_id": job_video_id,
        "mask_dilation": proc["mask_dilation"],
        "mask_scale_x": proc.get("mask_scale_x", 1.45),
        "mask_scale_y": proc.get("mask_scale_y", 1.15),
        "mask_pad_x": proc.get("mask_pad_x", 8),
        "debug_masks": proc.get("debug_masks", True),
        "scene_threshold": config["scene_detection"]["threshold"],
        "min_scene_length": config["scene_detection"]["min_scene_length"],
        "max_clip_duration": proc["clip_max_duration"]
    }


def process_video_pipeline(video_id: str):
    """Background pipeline dispatcher."""
    log = lambda msg: print(f"[BACKEND] [{video_id}] {msg}")
    try:
        log("Starting pipeline")

        video = queue.get_video(video_id)
        if not video:
            log("ERROR: video not found in queue")
            return

        roi = {
            "x": video["roi_x"],
            "y": video["roi_y"],
            "w": video["roi_w"],
            "h": video["roi_h"]
        }
        log(f"Video loaded — path={video['video_path']} duration={video['duration']}s {video['width']}x{video['height']} roi={roi}")

        queue.update_video_status(video_id, "gpu_processing")

        if config["processing"].get("parallel_clips", True):
            _process_parallel_clips(video_id, video, roi, log)
        else:
            _process_whole_video(video_id, video, roi, log)

    except Exception as e:
        log(f"ERROR: {e}")
        queue.update_video_status(video_id, "failed")
        queue._log(video_id, None, "pipeline_error", str(e))


def _process_parallel_clips(video_id: str, video: dict, roi: dict, log):
    """Split locally, fan one RunPod job out per clip (endpoint Max Workers is
    the concurrency cap), reassemble locally. The GPU worker is unchanged — it
    just receives a short video."""
    video_path = video["video_path"]
    fps = video["fps"] or probe_video(video_path)["fps"]

    work_dir = DATA_DIR / "clips" / video_id
    shutil.rmtree(work_dir, ignore_errors=True)
    work_dir.mkdir(parents=True, exist_ok=True)

    # Kick off dubbing immediately — it runs at ElevenLabs while the clips go
    # through split/upload/RunPod, and is joined right before the final mux.
    dub_future = None
    dub_executor = None
    if config["processing"].get("dub_enabled", False):
        eleven_key = os.environ.get("ELEVENLABS_API_KEY", "")
        if not eleven_key:
            log("WARNING: dub_enabled but ELEVENLABS_API_KEY not set — skipping dubbing")
        else:
            audio_src = extract_audio(video_path, str(work_dir / "orig_audio.mp3"))
            if not audio_src:
                log("WARNING: video has no audio stream — skipping dubbing")
            else:
                target_lang = config["processing"].get("dub_target_lang", "de")
                log(f"Starting ElevenLabs dubbing to '{target_lang}' in background...")
                dub_executor = ThreadPoolExecutor(max_workers=1)
                dub_future = dub_executor.submit(
                    dub_audio, audio_src, target_lang, eleven_key,
                    str(work_dir / f"dubbed_{target_lang}.mp3"),
                    timeout=config["processing"].get("dub_timeout", 900))

    log("Detecting scenes locally...")
    cuts = detect_scenes(video_path,
                         threshold=config["scene_detection"]["threshold"],
                         min_scene_length=config["scene_detection"]["min_scene_length"])
    segments = [(cuts[i], cuts[i + 1]) for i in range(len(cuts) - 1)]
    segments = subdivide_segments(segments, config["processing"]["clip_max_duration"])

    log(f"Splitting into {len(segments)} clips locally...")
    clip_paths = split_video(video_path, segments, str(work_dir), fps)
    n = len(clip_paths)
    queue.clear_clips(video_id)
    clip_ids = queue.create_clips_batch(video_id, segments, clip_paths)

    creds = _get_drive_creds()
    token = creds.token

    log(f"Uploading {n} clips to Drive...")
    with ThreadPoolExecutor(max_workers=4) as pool:
        file_ids = list(pool.map(
            lambda iv: upload_to_drive(iv[1], f"{video_id}_c{iv[0]+1:04d}.mp4", creds),
            enumerate(clip_paths)))

    jobs = {}
    for i, fid in enumerate(file_ids):
        job_id = runpod.submit_job(_job_payload(fid, roi, f"{video_id}_c{i+1:04d}", token))
        jobs[i] = job_id
        queue.update_clip_status(clip_ids[i], "gpu_processing", runpod_job_id=job_id)
    log(f"Submitted {n} RunPod jobs — endpoint Max Workers caps the parallelism")

    per_clip_timeout = config["worker"]["timeout_per_clip"] * 4

    def wait_and_fetch(i):
        """Wait for clip i's job, retrying once on failure. Downloads the
        result (and debug overlay, best-effort). Returns (i, result, debug)."""
        for attempt in (1, 2):
            job_id = jobs[i] if attempt == 1 else runpod.submit_job(
                _job_payload(file_ids[i], roi, f"{video_id}_c{i+1:04d}", token))
            try:
                status = runpod.wait_for_job(job_id, poll_interval=3.0,
                                             timeout=per_clip_timeout)
            except TimeoutError:
                if attempt == 2:
                    queue.update_clip_status(clip_ids[i], "failed",
                                             error_message="timed out twice")
                    raise
                log(f"clip {i+1}/{n}: timed out, retrying")
                continue
            out = status.get("output") or {}
            if status["status"] != "COMPLETED" or out.get("error"):
                err = out.get("error") or status["status"]
                if attempt == 2:
                    queue.update_clip_status(clip_ids[i], "failed", error_message=str(err))
                    raise RuntimeError(f"clip {i+1}/{n} failed twice: {err}")
                log(f"clip {i+1}/{n}: failed ({err}), retrying")
                continue

            result_path = str(work_dir / f"result_{i+1:04d}.mp4")
            download_from_drive(out["result_file_id"], result_path, creds)
            delete_from_drive(out["result_file_id"], creds)

            debug_path = None
            if out.get("debug_file_id"):
                try:
                    debug_path = str(work_dir / f"debug_{i+1:04d}.mp4")
                    download_from_drive(out["debug_file_id"], debug_path, creds)
                    delete_from_drive(out["debug_file_id"], creds)
                except Exception as e:
                    log(f"clip {i+1}/{n}: debug download failed (non-fatal): {e}")
                    debug_path = None

            queue.update_clip_status(clip_ids[i], "completed",
                                     runpod_job_id=job_id, result_path=result_path)
            log(f"clip {i+1}/{n} completed (worker {out.get('worker_version', '?')})")
            return i, result_path, debug_path

    try:
        with ThreadPoolExecutor(max_workers=min(n, 16)) as pool:
            results = list(pool.map(wait_and_fetch, range(n)))
    finally:
        for fid in file_ids:
            delete_from_drive(fid, creds)

    log("All clips done — concatenating and muxing audio...")
    results.sort(key=lambda t: t[0])
    if config["processing"].get("shuffle_clips", True):
        order = pair_swap_order(len(results))
        results = [results[j] for j in order]
        log(f"Shuffled clip order (video only, audio unchanged): {[j + 1 for j in order]}")
    final_dir = DATA_DIR / "final"
    final_dir.mkdir(parents=True, exist_ok=True)

    video_only = str(work_dir / "video_only.mp4")
    concat_clips([r[1] for r in results], video_only)

    audio_source = video_path  # fallback: original audio
    if dub_future is not None:
        try:
            audio_source = dub_future.result()
            log("Dubbing done — using dubbed audio track")
        except Exception as e:
            log(f"WARNING: dubbing failed, falling back to original audio: {e}")
            queue._log(video_id, None, "dubbing_warning", str(e))
    if dub_executor is not None:
        dub_executor.shutdown(wait=False)

    final_path = str(final_dir / f"{video_id}_final.mp4")
    mux_audio(video_only, audio_source, final_path)

    debug_paths = [r[2] for r in results if r[2]]
    if debug_paths:
        try:
            concat_clips(debug_paths, str(final_dir / f"{video_id}_debug.mp4"))
            log(f"Debug overlay saved to {final_dir / f'{video_id}_debug.mp4'}")
        except Exception as e:
            log(f"Debug concat failed (non-fatal): {e}")

    queue.set_video_result(video_id, final_path)
    log(f"Pipeline complete — result saved to {final_path}")
    shutil.rmtree(work_dir, ignore_errors=True)


def _process_whole_video(video_id: str, video: dict, roi: dict, log):
    """Legacy single-job path (processing.parallel_clips: false): the whole
    video is one RunPod job and the worker splits/concats internally."""
    video_path = video["video_path"]
    file_size_mb = os.path.getsize(video_path) / (1024 * 1024)
    log(f"Uploading video to Google Drive: {video_path} ({file_size_mb:.1f} MB)")

    creds = _get_drive_creds()
    video_file_id = upload_to_drive(video_path, f"{video_id}.mp4", creds)
    log(f"Uploaded — Drive file id: {video_file_id}")

    payload = _job_payload(video_file_id, roi, video_id, creds.token)
    timeout = config["worker"]["timeout_per_clip"] * 20
    log(f"Sending to RunPod — timeout={timeout}s")

    try:
        result = runpod.run_sync(payload, timeout=timeout)
    finally:
        delete_from_drive(video_file_id, creds)
    log(f"RunPod response received — keys={list(result.keys()) if result else 'None'}")

    if result and result.get("error"):
        raise RuntimeError(f"GPU worker error: {result['error']}")

    result_file_id = (result or {}).get("result_file_id", "")
    if not result_file_id:
        raise RuntimeError("No result file id from GPU worker")

    final_dir = DATA_DIR / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    final_path = str(final_dir / f"{video_id}_final.mp4")

    log(f"Downloading result from Drive (file id {result_file_id})")
    download_from_drive(result_file_id, final_path, creds)
    delete_from_drive(result_file_id, creds)

    debug_file_id = (result or {}).get("debug_file_id", "")
    if debug_file_id:
        try:
            debug_path = str(final_dir / f"{video_id}_debug.mp4")
            download_from_drive(debug_file_id, debug_path, creds)
            delete_from_drive(debug_file_id, creds)
            log(f"Debug overlay saved to {debug_path}")
        except Exception as e:
            log(f"Debug overlay download failed (non-fatal): {e}")

    log(f"Pipeline complete — result saved to {final_path}")
    queue.set_video_result(video_id, final_path)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/upload", methods=["POST"])
def upload_video():
    if "video" not in request.files:
        return jsonify({"error": "No video file"}), 400

    file = request.files["video"]
    if file.filename == "":
        return jsonify({"error": "No filename"}), 400

    ext = Path(file.filename).suffix or ".mp4"
    safe_name = f"{uuid.uuid4().hex[:8]}{ext}"
    upload_dir = DATA_DIR / "uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)
    video_path = str(upload_dir / safe_name)
    file.save(video_path)

    info = probe_video(video_path)

    video_id = queue.create_video(
        original_filename=file.filename,
        video_path=video_path,
        duration=info["duration"],
        width=info["width"],
        height=info["height"],
        fps=info["fps"]
    )

    return jsonify({
        "video_id": video_id,
        "filename": file.filename,
        "duration": info["duration"],
        "width": info["width"],
        "height": info["height"],
        "fps": info["fps"],
        "video_url": f"/data/uploads/{safe_name}"
    })


@app.route("/api/video/<video_id>/roi", methods=["POST"])
def set_roi(video_id):
    data = request.json
    queue.set_roi(
        video_id,
        x=int(data["x"]),
        y=int(data["y"]),
        w=int(data["w"]),
        h=int(data["h"])
    )
    return jsonify({"status": "ok"})


@app.route("/api/video/<video_id>/start", methods=["POST"])
def start_processing(video_id):
    video = queue.get_video(video_id)
    if not video:
        return jsonify({"error": "Video not found"}), 404

    if not all(video.get(f"roi_{k}") is not None for k in ("x", "y", "w", "h")):
        return jsonify({"error": "ROI not set"}), 400

    queue.update_video_status(video_id, "queued")
    thread = threading.Thread(target=process_video_pipeline, args=(video_id,))
    thread.daemon = True
    thread.start()

    return jsonify({"status": "started", "video_id": video_id})


@app.route("/api/video/<video_id>/status", methods=["GET"])
def video_status(video_id):
    video = queue.get_video(video_id)
    if not video:
        return jsonify({"status": "not_found"}), 404

    # Parallel-clip runs have real per-clip rows; the overall status always
    # comes from the video row (clips finish before the final concat/mux does).
    summary = queue.get_video_status_summary(video_id)
    if summary["total"] > 0:
        summary["status"] = video["status"]
        return jsonify(summary)

    return jsonify({
        "status": video["status"],
        "total": 1,
        "completed": 1 if video["status"] == "completed" else 0,
        "processing": 1 if video["status"] in ("gpu_processing", "queued") else 0,
        "failed": 1 if video["status"] == "failed" else 0,
        "pending": 0,
        "clips": []
    })


@app.route("/api/video/<video_id>/cancel", methods=["POST"])
def cancel_video(video_id):
    queue.cancel_video(video_id)
    return jsonify({"status": "cancelled"})


@app.route("/api/video/<video_id>/logs", methods=["GET"])
def video_logs(video_id):
    logs = queue.get_logs(video_id)
    return jsonify(logs)


@app.route("/api/videos", methods=["GET"])
def list_videos():
    videos = queue.list_videos()
    return jsonify(videos)


@app.route("/data/<path:filepath>")
def serve_data(filepath):
    return send_from_directory(str(DATA_DIR), filepath)


if __name__ == "__main__":
    app.run(
        host=config["server"]["host"],
        port=config["server"]["port"],
        debug=config["server"]["debug"]
    )
