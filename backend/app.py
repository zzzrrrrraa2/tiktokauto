import os
import sys
import threading
import time
import uuid
import yaml
import requests
import io
import json
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

from pipeline.ffmpeg_utils import probe_video, extract_frame
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


def process_video_pipeline(video_id: str):
    """Background pipeline: upload video to file host, send download URL to GPU worker."""
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

        video_path = video["video_path"]
        file_size_mb = os.path.getsize(video_path) / (1024 * 1024)
        log(f"Uploading video to Google Drive: {video_path} ({file_size_mb:.1f} MB)")

        creds = _get_drive_creds()
        video_file_id = upload_to_drive(video_path, f"{video_id}.mp4", creds)
        log(f"Uploaded — Drive file id: {video_file_id}")

        payload = {
            "video_file_id": video_file_id,
            "drive_access_token": creds.token,
            "roi": roi,
            "video_id": video_id,
            "mask_dilation": config["processing"]["mask_dilation"],
            "mask_scale_x": config["processing"].get("mask_scale_x", 1.45),
            "mask_scale_y": config["processing"].get("mask_scale_y", 1.15),
            "mask_pad_x": config["processing"].get("mask_pad_x", 8),
            "debug_masks": config["processing"].get("debug_masks", True),
            "scene_threshold": config["scene_detection"]["threshold"],
            "min_scene_length": config["scene_detection"]["min_scene_length"],
            "max_clip_duration": config["processing"]["clip_max_duration"]
        }
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
        if result_file_id:
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
        else:
            log("ERROR: no result_file_id in RunPod response")
            queue.update_video_status(video_id, "failed")
            queue._log(video_id, None, "pipeline_error", "No result file id from GPU worker")

    except Exception as e:
        log(f"ERROR: {e}")
        queue.update_video_status(video_id, "failed")
        queue._log(video_id, None, "pipeline_error", str(e))


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
