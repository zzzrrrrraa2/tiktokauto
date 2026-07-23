#!/usr/bin/env python3
"""RunPod Serverless Handler — Full Video Caption Removal Pipeline.

Input payload:
  {
    "video_file_id": "<Google Drive file id>",        # preferred (with drive_access_token)
    "drive_access_token": "<short-lived OAuth token>", # used to download input + upload result
    "video_download_url" | "video_data_b64" | "video_path": ...,  # fallbacks / manual testing
    "roi": {"x", "y", "w", "h"},                       # caption region in source pixel coords
    "video_id": str,
    "mask_dilation": int,
    "scene_threshold": float,
    "min_scene_length": float,
    "max_clip_duration": float
  }

Pipeline (all on GPU):
  1. PySceneDetect → scene boundaries (capped at max_clip_duration per clip)
  2. ffmpeg → split video into frame-accurate clips (video only, re-encoded)
  3. Per clip: PaddleOCR on ROI → text masks → ProPainter inpainting on a crop
     around the ROI only (memory- and speed-bounded), composited back full-res
  4. ffmpeg → concatenate clips, mux original audio back in
  5. Upload result to Google Drive (or return base64 for small manual tests)
"""

import os
import sys
import json
import math
import time
import base64
import shutil
import fractions
import subprocess
import hashlib
import re
import numpy as np
import cv2
from PIL import Image
import torch
import torchvision
import warnings
warnings.filterwarnings("ignore")

import runpod

WEIGHTS_DIR = "/app/weights"

PROPAINTER_DIR = os.path.join(WEIGHTS_DIR, "propainter")
PROPAINTER_REPO = "/app/propainter_repo"
PROPAINTER_MODEL_PATH = os.path.join(PROPAINTER_DIR, "ProPainter.pth")
RAFT_MODEL_PATH = os.path.join(PROPAINTER_DIR, "raft-things.pth")
FLOW_COMPLETE_PATH = os.path.join(PROPAINTER_DIR, "recurrent_flow_completion.pth")

WEIGHTS_URLS = {
    PROPAINTER_MODEL_PATH: "https://github.com/sczhou/ProPainter/releases/download/v0.1.0/ProPainter.pth",
    RAFT_MODEL_PATH: "https://github.com/sczhou/ProPainter/releases/download/v0.1.0/raft-things.pth",
    FLOW_COMPLETE_PATH: "https://github.com/sczhou/ProPainter/releases/download/v0.1.0/recurrent_flow_completion.pth",
}


def _download_weights():
    os.makedirs(PROPAINTER_DIR, exist_ok=True)
    for path, url in WEIGHTS_URLS.items():
        if not os.path.exists(path):
            print(f"[GPU_WORKER] Downloading {os.path.basename(path)}...", flush=True)
            subprocess.run(["wget", "-q", "-O", path, url], check=True)
            print(f"[GPU_WORKER] Downloaded {os.path.basename(path)} ({os.path.getsize(path) / 1024**2:.1f} MB)", flush=True)

_download_weights()

sys.path.insert(0, PROPAINTER_REPO)

from model.modules.flow_comp_raft import RAFT_bi
from model.recurrent_flow_completion import RecurrentFlowCompleteNet
from model.propainter import InpaintGenerator
from core.utils import to_tensors

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_HALF = DEVICE.type == "cuda"

# Logged at job start and returned in the output so it's always provable which
# image build actually served a job. Bump together with the CI image tag.
WORKER_VERSION = "v14"
PAYLOAD_SCHEMA_VERSION = 1

DRIVE_API = "https://www.googleapis.com/drive/v3"
DRIVE_UPLOAD = "https://www.googleapis.com/upload/drive/v3/files"

# Inpainting crop margin around the ROI. Must comfortably exceed total mask
# dilation (pre-dilation + ProPainter flow dilation ≈ 1.5 * mask_dilation px).
CROP_MARGIN = 96
CROP_MIN_SIZE = 160

# Captions here are fast, animated, one-word pop-ins (~6-12 frames @ 30fps).
# OCR every frame; each frame's mask is the union of detections within
# MASK_WINDOW frames on either side, so every word is masked at its biggest
# extent for its whole lifetime — see run_ocr() for rationale.
OCR_STEP = 1
MASK_WINDOW = 5

# Per-polygon mask expansion. Horizontal is deliberately much larger than
# vertical: the DB text detector expands its shrunken text core by a constant
# offset proportional to text HEIGHT (area*ratio/perimeter ≈ H/2 for wide
# boxes), so detected boxes systematically clip word sides regardless of word
# width. Overridable per job via mask_scale_x/mask_scale_y/mask_pad_x inputs
# (plumbed from config.yaml by the backend — tune there, no image rebuild).
MASK_SCALE_X = 1.45
MASK_SCALE_Y = 1.15
MASK_PAD_X = 8


# ---------------------------------------------------------------------------
# Google Drive transfer (plain REST — no client libraries needed on the worker)
# ---------------------------------------------------------------------------

def sha256_file(path: str, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        while True:
            chunk = source.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def structured_error(code: str, retryable: bool, stage: str, message: str,
                     job_id: str = None, signature: str = None) -> dict:
    return {
        "code": code,
        "retryable": bool(retryable),
        "stage": stage,
        "message": str(message)[:2000],
        "worker_version": WORKER_VERSION,
        "job_id": job_id,
        "signature": signature or hashlib.sha256(
            f"{stage}:{code}:{message}".encode()).hexdigest()[:20],
    }


def drive_file_metadata(file_id: str, token: str):
    import requests as req
    response = req.get(
        f"{DRIVE_API}/files/{file_id}",
        params={"fields": "id,name,size,md5Checksum,appProperties,trashed"},
        headers={"Authorization": f"Bearer {token}"},
        timeout=60,
    )
    if response.status_code == 404:
        return None
    response.raise_for_status()
    return response.json()


def drive_download(file_id: str, token: str, dest_path: str,
                   expected_size: int = None, expected_sha256: str = None):
    import requests as req
    partial = f"{dest_path}.partial"
    offset = os.path.getsize(partial) if os.path.isfile(partial) else 0
    headers = {"Authorization": f"Bearer {token}"}
    if offset:
        headers["Range"] = f"bytes={offset}-"
    with req.get(f"{DRIVE_API}/files/{file_id}?alt=media",
                 headers=headers, stream=True, timeout=600) as r:
        r.raise_for_status()
        mode = "ab" if offset and r.status_code == 206 else "wb"
        with open(partial, mode) as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
    size = os.path.getsize(partial)
    if expected_size is not None and size != int(expected_size):
        raise ValueError(
            f"input size mismatch: expected {expected_size}, received {size}")
    digest = sha256_file(partial)
    if expected_sha256 and digest.lower() != expected_sha256.lower():
        raise ValueError("input SHA-256 mismatch")
    os.replace(partial, dest_path)
    return {"size": size, "sha256": digest}


def _session_offset(response) -> int:
    match = re.search(r"bytes=0-(\d+)", response.headers.get("Range", ""))
    return int(match.group(1)) + 1 if match else 0


def drive_upload_reserved(file_path: str, file_id: str, session_url: str,
                          token: str, name: str = None) -> dict:
    """Resume the backend-created session for a pre-generated Drive file ID."""
    import requests as req
    size = os.path.getsize(file_path)
    digest = sha256_file(file_path)
    if not session_url:
        response = req.post(
            f"{DRIVE_UPLOAD}?uploadType=resumable&fields=id,size,md5Checksum",
            headers={
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json; charset=UTF-8",
                "X-Upload-Content-Type": "video/mp4",
                "X-Upload-Content-Length": str(size),
            },
            json={"id": file_id, "name": name or f"{file_id}.mp4"},
            timeout=60,
        )
        if response.status_code == 409:
            existing = drive_file_metadata(file_id, token)
            if existing and int(existing.get("size") or 0) > 0:
                return {"id": file_id, "size": int(existing["size"]),
                        "sha256": None, "already_existed": True}
        response.raise_for_status()
        session_url = response.headers["Location"]

    query = req.put(
        session_url,
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Length": "0",
            "Content-Range": f"bytes */{size}",
        },
        timeout=60,
    )
    if query.status_code in (200, 201):
        return {"id": file_id, "size": size, "sha256": digest}
    if query.status_code == 308:
        offset = _session_offset(query)
    elif query.status_code in (404, 410):
        # Recreate using the same deterministic ID. A 409 is checked as
        # possible prior success, so this cannot create a duplicate.
        return drive_upload_reserved(file_path, file_id, "", token, name)
    else:
        query.raise_for_status()

    chunk_size = 8 * 1024 * 1024
    with open(file_path, "rb") as source:
        source.seek(offset)
        while offset < size:
            data = source.read(min(chunk_size, size - offset))
            end = offset + len(data) - 1
            response = req.put(
                session_url,
                data=data,
                headers={
                    "Authorization": f"Bearer {token}",
                    "Content-Type": "video/mp4",
                    "Content-Length": str(len(data)),
                    "Content-Range": f"bytes {offset}-{end}/{size}",
                },
                timeout=600,
            )
            if response.status_code == 308:
                offset = _session_offset(response)
                source.seek(offset)
                continue
            response.raise_for_status()
            result = response.json()
            if result.get("id") and result["id"] != file_id:
                raise ValueError("Drive returned a different reserved file ID")
            return {"id": file_id, "size": size, "sha256": digest}
    raise RuntimeError("reserved Drive upload ended before completion")


def drive_upload(file_path: str, name: str, token: str) -> str:
    """Resumable upload; returns the new file id."""
    import requests as req
    size = os.path.getsize(file_path)
    r = req.post(
        f"{DRIVE_UPLOAD}?uploadType=resumable&fields=id",
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json; charset=UTF-8",
            "X-Upload-Content-Type": "video/mp4",
            "X-Upload-Content-Length": str(size),
        },
        json={"name": name},
        timeout=60,
    )
    r.raise_for_status()
    session_url = r.headers["Location"]
    with open(file_path, "rb") as f:
        r2 = req.put(
            session_url,
            data=f,
            headers={"Content-Type": "video/mp4", "Content-Length": str(size)},
            timeout=1800,
        )
    r2.raise_for_status()
    return r2.json()["id"]


# ---------------------------------------------------------------------------
# Video utilities
# ---------------------------------------------------------------------------

def probe_video(video_path: str) -> dict:
    cmd = [
        "ffprobe", "-v", "quiet", "-print_format", "json",
        "-show_format", "-show_streams", video_path
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    data = json.loads(result.stdout)
    video_stream = None
    for stream in data.get("streams", []):
        if stream.get("codec_type") == "video":
            video_stream = stream
            break
    if video_stream is None:
        raise ValueError(f"No video stream in {video_path}")
    return {
        "width": int(video_stream["width"]),
        "height": int(video_stream["height"]),
        "fps": float(fractions.Fraction(video_stream.get("r_frame_rate", "30/1"))),
        "duration": float(data["format"].get("duration", 0)),
    }


def detect_scenes(video_path: str, threshold: float = 27.0,
                  min_scene_length: float = 1.5) -> list:
    """Detect scene boundaries with PySceneDetect. Returns sorted cut timestamps
    always starting at 0.0 and ending at the video duration — a video with no
    cuts yields [0.0, duration]."""
    from scenedetect import open_video, SceneManager
    from scenedetect.detectors import ContentDetector

    duration = probe_video(video_path)["duration"]

    video = open_video(video_path)
    scene_manager = SceneManager()
    scene_manager.add_detector(ContentDetector(threshold=threshold))
    scene_manager.detect_scenes(video)
    scene_list = scene_manager.get_scene_list()

    cuts = [0.0]
    for _start, end in scene_list:
        t = end.get_seconds()
        if t - cuts[-1] >= min_scene_length and t < duration - 0.05:
            cuts.append(t)
    cuts.append(duration)
    # Merge a trailing scene shorter than the minimum into its predecessor
    if len(cuts) > 2 and cuts[-1] - cuts[-2] < min_scene_length:
        del cuts[-2]
    return cuts


def subdivide_segments(segments: list, max_duration: float) -> list:
    """Split any segment longer than max_duration into equal-length chunks
    so per-clip memory stays bounded."""
    out = []
    for start, end in segments:
        d = end - start
        if d <= max_duration + 0.01:
            out.append((start, end))
            continue
        n = math.ceil(d / max_duration)
        step = d / n
        for i in range(n):
            out.append((start + i * step, min(end, start + (i + 1) * step)))
    return out


def split_video(input_path: str, segments: list, output_dir: str, fps: float) -> list:
    """Split video into frame-accurate clips (re-encoded, video-only).
    Audio is muxed back from the original after concatenation."""
    os.makedirs(output_dir, exist_ok=True)
    clip_paths = []
    for i, (start, end) in enumerate(segments):
        duration = end - start
        clip_path = os.path.join(output_dir, f"clip_{i+1:04d}.mp4")
        cmd = [
            "ffmpeg", "-y",
            "-ss", f"{start:.3f}",
            "-i", input_path,
            "-t", f"{duration:.3f}",
            "-an",
            "-vsync", "cfr", "-r", f"{fps}",
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
            "-pix_fmt", "yuv420p",
            clip_path
        ]
        proc = subprocess.run(cmd, capture_output=True)
        if proc.returncode != 0:
            raise RuntimeError(f"ffmpeg split failed for segment {i}: {proc.stderr.decode(errors='replace')[-2000:]}")
        clip_paths.append(clip_path)
    return clip_paths


def concat_clips(clip_paths: list, output_path: str):
    concat_file = os.path.join(os.path.dirname(output_path), "_concat_list.txt")
    with open(concat_file, "w") as f:
        for path in clip_paths:
            f.write(f"file '{os.path.abspath(path)}'\n")
    cmd = [
        "ffmpeg", "-y",
        "-f", "concat", "-safe", "0",
        "-i", concat_file,
        "-c", "copy",
        output_path
    ]
    proc = subprocess.run(cmd, capture_output=True)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg concat failed: {proc.stderr.decode(errors='replace')[-2000:]}")
    os.remove(concat_file)


def mux_audio(video_path: str, audio_source_path: str, output_path: str):
    """Copy the processed video stream and add the original audio (if any)."""
    cmd = [
        "ffmpeg", "-y",
        "-i", video_path,
        "-i", audio_source_path,
        "-map", "0:v:0", "-map", "1:a:0?",
        "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
        "-shortest", "-movflags", "+faststart",
        output_path
    ]
    proc = subprocess.run(cmd, capture_output=True)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg audio mux failed: {proc.stderr.decode(errors='replace')[-2000:]}")


def load_video_frames(video_path: str):
    """Returns (frames THWC uint8 numpy array, fps)."""
    vframes, _aframes, info = torchvision.io.read_video(
        filename=video_path, pts_unit="sec", output_format="THWC"
    )
    return vframes.numpy(), float(info["video_fps"])


def write_video(frames, fps: float, output_path: str):
    import imageio
    imageio.mimwrite(
        output_path, frames, fps=fps,
        codec="libx264", quality=8,
        pixelformat="yuv420p", macro_block_size=2,
    )


def write_debug_video(frames, masks, raw_masks, roi: dict, fps: float, output_path: str):
    """Overlay the final inpaint mask (red, 40% alpha) and the raw OCR
    detections (green outlines) on the original frames. Frame-by-frame so the
    clip isn't duplicated in memory; must run BEFORE compositing mutates
    `frames` in place."""
    import imageio
    x, y = roi["x"], roi["y"]
    writer = imageio.get_writer(
        output_path, fps=fps,
        codec="libx264", quality=8,
        pixelformat="yuv420p", macro_block_size=2,
    )
    try:
        for i in range(len(frames)):
            img = frames[i].copy()
            hot = masks[i] > 0
            img[hot] = (img[hot] * 0.6 + np.array([255, 0, 0]) * 0.4).astype(np.uint8)
            if raw_masks is not None and raw_masks[i].any():
                contours, _ = cv2.findContours(raw_masks[i], cv2.RETR_EXTERNAL,
                                               cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(img, contours, -1, (0, 255, 0), 2, offset=(x, y))
            writer.append_data(img)
    finally:
        writer.close()


# ---------------------------------------------------------------------------
# OCR → text masks
# ---------------------------------------------------------------------------

_OCR = None
_OCR_ON_CPU = False


def _get_ocr(cpu: bool = False):
    global _OCR, _OCR_ON_CPU
    if _OCR is None or (cpu and not _OCR_ON_CPU):
        from paddleocr import PaddleOCR
        kwargs = dict(
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
            lang="en",
            text_det_thresh=0.3,
            text_det_box_thresh=0.4,
            text_det_unclip_ratio=2.0
        )
        if cpu:
            kwargs["device"] = "cpu"
        _OCR = PaddleOCR(**kwargs)
        _OCR_ON_CPU = cpu
    return _OCR


def _ocr_predict(img):
    """Run OCR on the default (GPU) instance; if Paddle fails at construction
    or inference (e.g. missing CUDA kernels for this GPU model), rebuild the
    singleton on CPU and continue — slower, but the job survives."""
    try:
        return _get_ocr().predict(img)
    except ImportError:
        raise
    except Exception as e:
        if _OCR_ON_CPU:
            raise
        print(f"[GPU_WORKER] PaddleOCR GPU inference failed ({type(e).__name__}: {e}); "
              f"falling back to CPU OCR", flush=True)
        return _get_ocr(cpu=True).predict(img)


def clamp_roi(roi: dict, width: int, height: int) -> dict:
    x = min(max(int(roi["x"]), 0), max(0, width - 1))
    y = min(max(int(roi["y"]), 0), max(0, height - 1))
    w = max(1, min(int(roi["w"]), width - x))
    h = max(1, min(int(roi["h"]), height - y))
    return {"x": x, "y": y, "w": w, "h": h}


def widen_roi(roi: dict, width: int, height: int) -> dict:
    """Working ROI from the user-drawn box: honor only its vertical band.

    Word width routinely exceeds the drawn box (users draw it around a typical
    word; long words are 2-3x wider), and everything downstream — the OCR crop,
    the mask arrays, the inpainting crop — is hard-clipped to the working ROI.
    Through v12 that silently capped every mask at the drawn width. So: full
    frame width horizontally, drawn band plus a 15% margin vertically."""
    margin = int(round(roi["h"] * 0.15))
    y = max(0, int(roi["y"]) - margin)
    h = min(height - y, int(roi["h"]) + 2 * margin)
    return {"x": 0, "y": y, "w": width, "h": h}


def run_ocr(frames, roi: dict, scale_x: float = MASK_SCALE_X,
            scale_y: float = MASK_SCALE_Y, pad_x: float = MASK_PAD_X,
            return_debug: bool = False):
    """Per-word max-extent masking. Returns one full-frame uint8 mask (0/255)
    per frame.

    Captions here are fast, animated, one-word pop-ins (~6-12 frames @ 30fps,
    scaling ~0%→110%→100%), and word sizes vary hugely ("readiness" vs "at").
    OCR runs on every frame's ROI crop; each detected polygon is expanded
    anisotropically around its own bbox center — width by scale_x plus a
    constant pad_x per side (DB detection under-covers word sides by a
    roughly constant amount, see the MASK_SCALE_X comment), height by scale_y
    (covers the pop animation's ~110% overshoot). A frame's mask is then the
    union of detections within MASK_WINDOW frames on either side — a word is
    only ever MASK_WINDOW frames from its peak-size detection, so every frame
    of its life (including pop-in/pop-out frames where OCR sees nothing or a
    tiny box) is masked at the word's biggest extent, while short words get
    proportionally small holes and caption-free stretches get no mask at all.

    With return_debug=True, also returns the raw (unexpanded, unwindowed)
    per-frame detection mini-masks (n, roi_h, roi_w) for the debug overlay.
    """
    try:
        import paddleocr  # noqa: F401 — availability check only
    except ImportError:
        masks = _fallback_text_detection(frames, roi)
        return (masks, None) if return_debug else masks

    x, y, w, h = roi["x"], roi["y"], roi["w"], roi["h"]
    frame_h, frame_w = frames[0].shape[:2]
    n = len(frames)

    # Pass 1: OCR each frame's ROI crop into an ROI-sized mini-mask.
    roi_masks = np.zeros((n, h, w), dtype=np.uint8)
    raw_masks = np.zeros((n, h, w), dtype=np.uint8) if return_debug else None
    for i in range(0, n, OCR_STEP):
        roi_bgr = cv2.cvtColor(frames[i][y:y+h, x:x+w], cv2.COLOR_RGB2BGR)
        result = _ocr_predict(roi_bgr)
        for res in result or []:
            polys = res.get("dt_polys") if hasattr(res, "get") else None
            if polys is None:
                continue
            for box in polys:
                pts = np.asarray(box, dtype=np.float32).reshape(-1, 2)
                if raw_masks is not None:
                    cv2.fillPoly(raw_masks[i], [np.round(pts).astype(np.int32)], 255)
                cx, cy = (pts.min(axis=0) + pts.max(axis=0)) / 2.0
                pts[:, 0] = cx + (pts[:, 0] - cx) * scale_x + np.sign(pts[:, 0] - cx) * pad_x
                pts[:, 1] = cy + (pts[:, 1] - cy) * scale_y
                cv2.fillPoly(roi_masks[i], [np.round(pts).astype(np.int32)], 255)

    # Pass 2: per-frame union over the temporal window, embedded full-frame.
    mask_frames = []
    for i in range(n):
        lo, hi = max(0, i - MASK_WINDOW), min(n, i + MASK_WINDOW + 1)
        mask = np.zeros((frame_h, frame_w), dtype=np.uint8)
        mask[y:y+h, x:x+w] = roi_masks[lo:hi].max(axis=0)
        mask_frames.append(mask)

    return (mask_frames, raw_masks) if return_debug else mask_frames


def _fallback_text_detection(frames, roi: dict) -> list:
    x, y, w, h = roi["x"], roi["y"], roi["w"], roi["h"]
    mask_frames = []

    for i in range(len(frames)):
        frame_np = frames[i]
        mask = np.zeros(frame_np.shape[:2], dtype=np.uint8)
        roi_region = frame_np[y:y+h, x:x+w]

        gray = cv2.cvtColor(roi_region, cv2.COLOR_RGB2GRAY)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 3))
        dilated = cv2.dilate(binary, kernel, iterations=3)
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            cx, cy, cw, ch = cv2.boundingRect(cnt)
            if cw > 30 and ch > 8:
                cv2.rectangle(mask, (x + cx, y + cy), (x + cx + cw, y + cy + ch), 255, -1)

        mask_frames.append(mask)

    return mask_frames


# ---------------------------------------------------------------------------
# ProPainter inpainting (on a crop around the ROI)
# ---------------------------------------------------------------------------

def compute_crop_box(roi: dict, width: int, height: int,
                     margin: int = CROP_MARGIN, min_size: int = CROP_MIN_SIZE):
    """Crop window covering ROI + margin, dimensions multiples of 8,
    clamped to the frame. Returns (x0, y0, x1, y1)."""
    def fit(lo, hi, limit):
        lo = max(0, lo - margin)
        hi = min(limit, hi + margin)
        size = max(hi - lo, min(min_size, limit))
        size = min(size, limit)
        size -= size % 8
        lo = max(0, min(lo, limit - size))
        return lo, lo + size

    x0, x1 = fit(roi["x"], roi["x"] + roi["w"], width)
    y0, y1 = fit(roi["y"], roi["y"] + roi["h"], height)
    return x0, y0, x1, y1


def run_propainter(crop_frames: list, crop_masks: list,
                   fix_raft, fix_flow_complete, model,
                   mask_dilation: int = 8) -> list:
    """Inpaint masked regions. crop_frames: PIL RGB images, crop_masks: PIL L
    masks — all the same size with dimensions divisible by 8.
    Returns the completed frames as uint8 numpy arrays."""
    video_length = len(crop_frames)
    w, h = crop_frames[0].size

    kernel = np.ones((3, 3), np.uint8)
    masks_dilated = []
    for m in crop_masks:
        arr = np.array(m)
        if mask_dilation > 0:
            arr = cv2.dilate(arr, kernel, iterations=mask_dilation)
        masks_dilated.append(Image.fromarray(((arr > 0) * 255).astype(np.uint8)))

    frames_t = to_tensors()(crop_frames).unsqueeze(0) * 2 - 1
    flow_masks_t = to_tensors()(masks_dilated).unsqueeze(0)
    masks_dilated_t = flow_masks_t
    frames_t = frames_t.to(DEVICE)
    flow_masks_t = flow_masks_t.to(DEVICE)
    masks_dilated_t = masks_dilated_t.to(DEVICE)

    ref_stride = 10
    neighbor_length = 10
    subvideo_length = 50

    with torch.no_grad():
        if frames_t.size(-1) <= 640:
            short_clip_len = 12
        elif frames_t.size(-1) <= 720:
            short_clip_len = 8
        else:
            short_clip_len = 4

        if frames_t.size(1) > short_clip_len:
            gt_flows_f_list, gt_flows_b_list = [], []
            for f in range(0, video_length, short_clip_len):
                end_f = min(video_length, f + short_clip_len)
                if f == 0:
                    flows_f, flows_b = fix_raft(frames_t[:, f:end_f], iters=20)
                else:
                    flows_f, flows_b = fix_raft(frames_t[:, f-1:end_f], iters=20)
                gt_flows_f_list.append(flows_f)
                gt_flows_b_list.append(flows_b)
                torch.cuda.empty_cache()
            gt_flows_f = torch.cat(gt_flows_f_list, dim=1)
            gt_flows_b = torch.cat(gt_flows_b_list, dim=1)
            gt_flows_bi = (gt_flows_f, gt_flows_b)
        else:
            gt_flows_bi = fix_raft(frames_t, iters=20)
            torch.cuda.empty_cache()

        if USE_HALF:
            frames_t = frames_t.half()
            flow_masks_t = flow_masks_t.half()
            masks_dilated_t = flow_masks_t
            gt_flows_bi = (gt_flows_bi[0].half(), gt_flows_bi[1].half())
            fix_flow_complete = fix_flow_complete.half()
            model = model.half()

        flow_length = gt_flows_bi[0].size(1)
        if flow_length > subvideo_length:
            pred_flows_f, pred_flows_b = [], []
            pad_len = 5
            for f in range(0, flow_length, subvideo_length):
                s_f = max(0, f - pad_len)
                e_f = min(flow_length, f + subvideo_length + pad_len)
                pad_len_s = max(0, f) - s_f
                pad_len_e = e_f - min(flow_length, f + subvideo_length)
                pred_flows_bi_sub, _ = fix_flow_complete.forward_bidirect_flow(
                    (gt_flows_bi[0][:, s_f:e_f], gt_flows_bi[1][:, s_f:e_f]),
                    flow_masks_t[:, s_f:e_f+1])
                pred_flows_bi_sub = fix_flow_complete.combine_flow(
                    (gt_flows_bi[0][:, s_f:e_f], gt_flows_bi[1][:, s_f:e_f]),
                    pred_flows_bi_sub,
                    flow_masks_t[:, s_f:e_f+1])
                pred_flows_f.append(pred_flows_bi_sub[0][:, pad_len_s:e_f-s_f-pad_len_e])
                pred_flows_b.append(pred_flows_bi_sub[1][:, pad_len_s:e_f-s_f-pad_len_e])
                torch.cuda.empty_cache()
            pred_flows_f = torch.cat(pred_flows_f, dim=1)
            pred_flows_b = torch.cat(pred_flows_b, dim=1)
            pred_flows_bi = (pred_flows_f, pred_flows_b)
        else:
            pred_flows_bi, _ = fix_flow_complete.forward_bidirect_flow(gt_flows_bi, flow_masks_t)
            pred_flows_bi = fix_flow_complete.combine_flow(gt_flows_bi, pred_flows_bi, flow_masks_t)
            torch.cuda.empty_cache()

        masked_frames = frames_t * (1 - masks_dilated_t)
        subvideo_length_img_prop = min(100, subvideo_length)
        if video_length > subvideo_length_img_prop:
            updated_frames, updated_masks = [], []
            pad_len = 10
            for f in range(0, video_length, subvideo_length_img_prop):
                s_f = max(0, f - pad_len)
                e_f = min(video_length, f + subvideo_length_img_prop + pad_len)
                pad_len_s = max(0, f) - s_f
                pad_len_e = e_f - min(video_length, f + subvideo_length_img_prop)
                b, t, _, _, _ = masks_dilated_t[:, s_f:e_f].size()
                pred_flows_bi_sub = (pred_flows_bi[0][:, s_f:e_f-1], pred_flows_bi[1][:, s_f:e_f-1])
                prop_imgs_sub, updated_local_masks_sub = model.img_propagation(
                    masked_frames[:, s_f:e_f], pred_flows_bi_sub,
                    masks_dilated_t[:, s_f:e_f], "nearest")
                updated_frames_sub = frames_t[:, s_f:e_f] * (1 - masks_dilated_t[:, s_f:e_f]) + \
                    prop_imgs_sub.view(b, t, 3, h, w) * masks_dilated_t[:, s_f:e_f]
                updated_masks_sub = updated_local_masks_sub.view(b, t, 1, h, w)
                updated_frames.append(updated_frames_sub[:, pad_len_s:e_f-s_f-pad_len_e])
                updated_masks.append(updated_masks_sub[:, pad_len_s:e_f-s_f-pad_len_e])
                torch.cuda.empty_cache()
            updated_frames = torch.cat(updated_frames, dim=1)
            updated_masks = torch.cat(updated_masks, dim=1)
        else:
            b, t, _, _, _ = masks_dilated_t.size()
            prop_imgs, updated_local_masks = model.img_propagation(
                masked_frames, pred_flows_bi, masks_dilated_t, "nearest")
            updated_frames = frames_t * (1 - masks_dilated_t) + prop_imgs.view(b, t, 3, h, w) * masks_dilated_t
            updated_masks = updated_local_masks.view(b, t, 1, h, w)
            torch.cuda.empty_cache()

    ori_frames = [np.array(f).astype(np.uint8) for f in crop_frames]
    comp_frames = [None] * video_length
    neighbor_stride = neighbor_length // 2

    if video_length > subvideo_length:
        ref_num = subvideo_length // ref_stride
    else:
        ref_num = -1

    for f in range(0, video_length, neighbor_stride):
        neighbor_ids = [
            i for i in range(max(0, f - neighbor_stride),
                             min(video_length, f + neighbor_stride + 1))
        ]
        ref_ids = []
        if ref_num == -1:
            for i in range(0, video_length, ref_stride):
                if i not in neighbor_ids:
                    ref_ids.append(i)
        else:
            start_idx = max(0, f - ref_stride * (ref_num // 2))
            end_idx = min(video_length, f + ref_stride * (ref_num // 2))
            for i in range(start_idx, end_idx, ref_stride):
                if i not in neighbor_ids:
                    if len(ref_ids) > ref_num:
                        break
                    ref_ids.append(i)

        selected_imgs = updated_frames[:, neighbor_ids + ref_ids, :, :, :]
        selected_masks = masks_dilated_t[:, neighbor_ids + ref_ids, :, :, :]
        selected_update_masks = updated_masks[:, neighbor_ids + ref_ids, :, :, :]
        selected_pred_flows_bi = (
            pred_flows_bi[0][:, neighbor_ids[:-1], :, :, :],
            pred_flows_bi[1][:, neighbor_ids[:-1], :, :, :]
        )

        with torch.no_grad():
            l_t = len(neighbor_ids)
            pred_img = model(selected_imgs, selected_pred_flows_bi,
                             selected_masks, selected_update_masks, l_t)
            pred_img = pred_img.view(-1, 3, h, w)
            pred_img = (pred_img + 1) / 2
            pred_img = pred_img.cpu().permute(0, 2, 3, 1).numpy() * 255
            binary_masks = masks_dilated_t[0, neighbor_ids, :, :, :].cpu().permute(
                0, 2, 3, 1).numpy().astype(np.uint8)

            for i in range(len(neighbor_ids)):
                idx = neighbor_ids[i]
                img = np.array(pred_img[i]).astype(np.uint8) * binary_masks[i] \
                    + ori_frames[idx] * (1 - binary_masks[i])
                if comp_frames[idx] is None:
                    comp_frames[idx] = img
                else:
                    comp_frames[idx] = comp_frames[idx].astype(np.float32) * 0.5 + img.astype(np.float32) * 0.5
                comp_frames[idx] = comp_frames[idx].astype(np.uint8)

        torch.cuda.empty_cache()

    return comp_frames


def process_clip(clip_path: str, roi: dict, clip_id: str, mask_dilation: int,
                 fix_raft, fix_flow_complete, propainter_model,
                 work_dir: str, log,
                 mask_scale_x: float = MASK_SCALE_X,
                 mask_scale_y: float = MASK_SCALE_Y,
                 mask_pad_x: float = MASK_PAD_X,
                 debug: bool = False):
    """Full processing of a single clip: OCR → mask → crop → ProPainter → composite.
    Returns (result_path, debug_path-or-None)."""
    frames, fps = load_video_frames(clip_path)
    n, frame_h, frame_w = frames.shape[0], frames.shape[1], frames.shape[2]

    roi_c = clamp_roi(roi, frame_w, frame_h)
    raw_masks = None
    if debug:
        masks, raw_masks = run_ocr(frames, roi_c, mask_scale_x, mask_scale_y,
                                   mask_pad_x, return_debug=True)
    else:
        masks = run_ocr(frames, roi_c, mask_scale_x, mask_scale_y, mask_pad_x)

    pre_dilate = max(1, mask_dilation // 2)
    kernel = np.ones((3, 3), np.uint8)
    masks = [cv2.dilate(m, kernel, iterations=pre_dilate) if m.any() else m for m in masks]

    result_path = os.path.join(work_dir, f"{clip_id}_out.mp4")

    debug_path = None
    if debug:
        debug_path = os.path.join(work_dir, f"{clip_id}_debug.mp4")
        write_debug_video(frames, masks, raw_masks, roi_c, fps, debug_path)

    has_text = any(m.any() for m in masks)
    if has_text and n >= 2:
        x0, y0, x1, y1 = compute_crop_box(roi_c, frame_w, frame_h)
        log(f"{clip_id}: {n} frames, inpainting crop ({x0},{y0})-({x1},{y1})")
        crop_frames = [Image.fromarray(frames[i, y0:y1, x0:x1]) for i in range(n)]
        crop_masks = [Image.fromarray(masks[i][y0:y1, x0:x1]) for i in range(n)]
        comp_frames = run_propainter(
            crop_frames, crop_masks,
            fix_raft, fix_flow_complete, propainter_model,
            mask_dilation=mask_dilation
        )
        for i in range(n):
            frames[i, y0:y1, x0:x1] = comp_frames[i]
        del crop_frames, crop_masks, comp_frames
        torch.cuda.empty_cache()
    else:
        log(f"{clip_id}: {n} frames, no text detected — passing through")

    write_video(frames, fps, result_path)
    return result_path, debug_path


fix_raft = None
fix_flow_complete = None
propainter_model = None


def load_propainter_models():
    fix_raft = RAFT_bi(RAFT_MODEL_PATH, DEVICE)
    fix_flow_complete = RecurrentFlowCompleteNet(FLOW_COMPLETE_PATH)
    for p in fix_flow_complete.parameters():
        p.requires_grad = False
    fix_flow_complete.to(DEVICE)
    fix_flow_complete.eval()
    model = InpaintGenerator(model_path=PROPAINTER_MODEL_PATH).to(DEVICE)
    model.eval()
    return fix_raft, fix_flow_complete, model


def init_models():
    global fix_raft, fix_flow_complete, propainter_model
    if fix_raft is None:
        fix_raft, fix_flow_complete, propainter_model = load_propainter_models()


def gpu_diagnostic(log):
    """Log which GPU this worker landed on and prove torch CUDA kernels run.
    The endpoint pool mixes 8 GPU models — this line is how per-model failures
    get correlated across jobs."""
    if not torch.cuda.is_available():
        log("WARNING: torch.cuda not available — inpainting will run on CPU (very slow)")
        return
    name = torch.cuda.get_device_name(0)
    cap = torch.cuda.get_device_capability(0)
    check = (torch.randn(8, 8, device="cuda") @ torch.randn(8, 8, device="cuda")).sum().item()
    log(f"GPU: {name} (sm_{cap[0]}{cap[1]}) — torch CUDA kernel check OK ({check:+.2f})")


def validate_job_input(job_input: dict, job_id: str = None) -> dict:
    mode = job_input.get("mode", "legacy")
    if mode not in ("clip", "legacy"):
        return structured_error(
            "invalid_mode", False, "validate", f"unsupported mode: {mode}", job_id)
    if mode != "clip":
        return None
    if int(job_input.get("schema_version", 0)) != PAYLOAD_SCHEMA_VERSION:
        return structured_error(
            "schema_mismatch", False, "validate",
            f"expected schema {PAYLOAD_SCHEMA_VERSION}", job_id)
    expected_worker = job_input.get("expected_worker_version")
    if expected_worker != WORKER_VERSION:
        return structured_error(
            "version_mismatch", False, "validate",
            f"expected worker {expected_worker}, running {WORKER_VERSION}", job_id)
    required = (
        "video_file_id", "drive_access_token", "input_size", "input_sha256",
        "result_file_id", "result_upload_session", "roi",
    )
    missing = [name for name in required if not job_input.get(name)]
    if missing:
        return structured_error(
            "invalid_payload", False, "validate",
            f"missing required fields: {', '.join(missing)}", job_id)
    roi = job_input.get("roi")
    if not isinstance(roi, dict) or any(
            key not in roi for key in ("x", "y", "w", "h")):
        return structured_error(
            "invalid_roi", False, "validate", "roi requires x, y, w, h", job_id)
    return None


def classify_worker_exception(exc: BaseException, stage: str,
                              job_id: str = None) -> dict:
    response = getattr(exc, "response", None)
    status = getattr(response, "status_code", None)
    if status == 429 or (status and status >= 500):
        return structured_error(
            "provider_transient", True, stage, str(exc), job_id,
            signature=f"http:{status}")
    if status in (401, 403):
        return structured_error(
            "drive_authorization", False, stage, str(exc), job_id,
            signature=f"http:{status}")
    if isinstance(exc, (TimeoutError, ConnectionError)):
        return structured_error(
            "network_timeout", True, stage, str(exc), job_id)
    text = str(exc).lower()
    if "cuda" in text or "out of memory" in text or "model" in text:
        return structured_error(
            "deterministic_model_failure", False, stage, str(exc), job_id)
    if isinstance(exc, (ValueError, KeyError, json.JSONDecodeError)):
        return structured_error(
            "invalid_or_corrupt_media", False, stage, str(exc), job_id)
    return structured_error("worker_failure", False, stage, str(exc), job_id)


def handler(job):
    """RunPod serverless handler — full pipeline."""
    job_input = job["input"]
    job_id = job.get("id")

    video_id = job_input.get("video_id", "unknown")
    log = lambda msg: print(f"[GPU_WORKER] [{video_id}] {msg}", flush=True)

    log(f"Worker {WORKER_VERSION} — handler invoked, input keys: {list(job_input.keys())}")
    validation_error = validate_job_input(job_input, job_id)
    if validation_error:
        return {
            "status": "failed",
            "worker_version": WORKER_VERSION,
            "job_id": job_id,
            "error": validation_error,
        }

    roi = job_input["roi"]
    mode = job_input.get("mode", "legacy")
    mask_dilation = int(job_input.get("mask_dilation", 8))
    scene_threshold = float(job_input.get("scene_threshold", 27.0))
    min_scene_length = float(job_input.get("min_scene_length", 1.5))
    max_clip_duration = float(job_input.get("max_clip_duration", 12.0))
    mask_scale_x = float(job_input.get("mask_scale_x", MASK_SCALE_X))
    mask_scale_y = float(job_input.get("mask_scale_y", MASK_SCALE_Y))
    mask_pad_x = float(job_input.get("mask_pad_x", MASK_PAD_X))
    debug_masks = bool(job_input.get("debug_masks", True))
    drive_token = job_input.get("drive_access_token", "")
    log(f"Params — roi={roi} mask_dilation={mask_dilation} scene_threshold={scene_threshold} "
        f"min_scene_length={min_scene_length} max_clip_duration={max_clip_duration} "
        f"mask_scale_x={mask_scale_x} mask_scale_y={mask_scale_y} mask_pad_x={mask_pad_x} "
        f"debug_masks={debug_masks} drive_token={'yes' if drive_token else 'no'}")

    # --- Acquire the source video ---------------------------------------
    video_file_id = job_input.get("video_file_id", "")
    video_download_url = job_input.get("video_download_url", "")
    video_data_b64 = job_input.get("video_data_b64", "")
    downloaded = False

    if mode == "clip" and drive_token:
        existing = drive_file_metadata(job_input["result_file_id"], drive_token)
        if (existing and not existing.get("trashed")
                and int(existing.get("size") or 0) > 0):
            log("Reserved result already exists; skipping recomputation")
            return {
                "status": "completed",
                "video_id": video_id,
                "worker_version": WORKER_VERSION,
                "job_id": job_id,
                "result_file_id": job_input["result_file_id"],
                "result_size": int(existing["size"]),
                "cache_hit": "reserved_drive_result",
                "timings": {"total_seconds": 0.0},
            }

    if video_file_id and drive_token:
        video_path = os.path.join("/tmp", f"{video_id}_{int(time.time())}.mp4")
        log(f"Downloading video from Drive (file id {video_file_id})...")
        if mode == "clip":
            metadata = drive_file_metadata(video_file_id, drive_token)
            if not metadata or metadata.get("trashed"):
                return {
                    "status": "failed",
                    "worker_version": WORKER_VERSION,
                    "job_id": job_id,
                    "error": structured_error(
                        "input_missing", False, "validate",
                        "Drive input does not exist", job_id),
                }
            if int(metadata.get("size") or 0) != int(job_input["input_size"]):
                return {
                    "status": "failed",
                    "worker_version": WORKER_VERSION,
                    "job_id": job_id,
                    "error": structured_error(
                        "input_size_mismatch", False, "validate",
                        "Drive input size differs from the reserved artifact", job_id),
                }
        drive_download(
            video_file_id,
            drive_token,
            video_path,
            expected_size=job_input.get("input_size"),
            expected_sha256=job_input.get("input_sha256"),
        )
        downloaded = True
    elif video_download_url:
        video_path = os.path.join("/tmp", f"{video_id}_{int(time.time())}.mp4")
        log(f"Downloading video from URL: {video_download_url}")
        import requests as req
        resp = req.get(video_download_url, stream=True, timeout=300)
        resp.raise_for_status()
        with open(video_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=1024 * 1024):
                f.write(chunk)
        downloaded = True
    elif video_data_b64:
        video_path = os.path.join("/tmp", f"{video_id}_{int(time.time())}.mp4")
        log(f"Decoding base64 video ({len(video_data_b64)} chars)...")
        with open(video_path, "wb") as f:
            f.write(base64.b64decode(video_data_b64))
        downloaded = True
    else:
        video_path = job_input.get("video_path", "")
        log(f"Using direct video_path: {video_path}")

    if not video_path or not os.path.exists(video_path):
        return {
            "status": "failed",
            "worker_version": WORKER_VERSION,
            "job_id": job_id,
            "error": structured_error(
                "input_missing", False, "validate",
                f"Video file not found: {video_path}", job_id),
        }
    log(f"Source video ready: {video_path} ({os.path.getsize(video_path) / 1024**2:.1f} MB)")

    work_dir = os.path.join("/tmp", f"work_{video_id}_{int(time.time())}")
    os.makedirs(work_dir, exist_ok=True)

    try:
        gpu_diagnostic(log)
        log("Loading ProPainter models...")
        init_models()
        log("Models loaded")

        t_start = time.time()
        info = probe_video(video_path)
        fps = info["fps"]

        roi_user = roi
        roi = widen_roi(roi, info["width"], info["height"])
        log(f"ROI widened {roi_user} -> {roi} (drawn box marks the caption band; "
            f"word width routinely exceeds it)")

        if mode == "clip":
            t_clip = time.time()
            result_path, debug_path = process_clip(
                video_path, roi, video_id, mask_dilation,
                fix_raft, fix_flow_complete, propainter_model,
                work_dir, log,
                mask_scale_x=mask_scale_x,
                mask_scale_y=mask_scale_y,
                mask_pad_x=mask_pad_x,
                debug=debug_masks,
            )
            process_seconds = time.time() - t_clip
            result = drive_upload_reserved(
                result_path,
                job_input["result_file_id"],
                job_input["result_upload_session"],
                drive_token,
                f"{video_id}_result.mp4",
            )
            output = {
                "status": "completed",
                "video_id": video_id,
                "worker_version": WORKER_VERSION,
                "job_id": job_id,
                "result_file_id": result["id"],
                "result_size": result["size"],
                "result_sha256": result.get("sha256"),
                "timings": {
                    "process_seconds": round(process_seconds, 3),
                    "total_seconds": round(time.time() - t_start, 3),
                },
            }
            if debug_path and job_input.get("debug_file_id"):
                debug = drive_upload_reserved(
                    debug_path,
                    job_input["debug_file_id"],
                    job_input.get("debug_upload_session", ""),
                    drive_token,
                    f"{video_id}_debug.mp4",
                )
                output.update({
                    "debug_file_id": debug["id"],
                    "debug_size": debug["size"],
                    "debug_sha256": debug.get("sha256"),
                })
            log(f"Clip mode complete in {time.time() - t_start:.1f}s")
            return output

        # Step 1: scene detection
        log("Step 1: Scene detection...")
        cuts = detect_scenes(video_path, threshold=scene_threshold,
                             min_scene_length=min_scene_length)
        segments = [(cuts[i], cuts[i + 1]) for i in range(len(cuts) - 1)]
        segments = subdivide_segments(segments, max_clip_duration)
        log(f"Scene detection done — {len(segments)} segments (after max-duration split)")

        # Step 2: split into clips
        log("Step 2: Splitting video into clips...")
        clip_dir = os.path.join(work_dir, "clips")
        clip_paths = split_video(video_path, segments, clip_dir, fps)
        log(f"Splitting done — {len(clip_paths)} clips created")

        # Step 3: process each clip
        result_clips = []
        debug_clips = []
        for i, clip_path in enumerate(clip_paths):
            clip_id = f"clip_{i+1:04d}"
            log(f"Step 3.{i+1}/{len(clip_paths)}: Processing {clip_id}...")
            t_clip = time.time()
            result_path, debug_path = process_clip(
                clip_path, roi, clip_id, mask_dilation,
                fix_raft, fix_flow_complete, propainter_model,
                work_dir, log,
                mask_scale_x=mask_scale_x, mask_scale_y=mask_scale_y,
                mask_pad_x=mask_pad_x, debug=debug_masks
            )
            log(f"{clip_id} done in {time.time() - t_clip:.1f}s")
            result_clips.append(result_path)
            if debug_path:
                debug_clips.append(debug_path)

        # Step 4: concatenate + restore audio
        log("Step 4: Concatenating clips and muxing audio...")
        video_only_path = os.path.join(work_dir, "video_only.mp4")
        concat_clips(result_clips, video_only_path)
        final_path = os.path.join(work_dir, "final.mp4")
        mux_audio(video_only_path, video_path, final_path)
        final_size = os.path.getsize(final_path)
        log(f"Final video ready ({final_size / 1024**2:.1f} MB)")

        t_total = time.time() - t_start

        # Step 5: deliver the result
        output = {
            "video_id": video_id,
            "worker_version": WORKER_VERSION,
            "num_clips": len(clip_paths),
            "total_time": round(t_total, 1),
            "status": "completed",
        }
        if drive_token:
            log("Step 5: Uploading result to Google Drive...")
            result_name = f"{video_id}_final.mp4"
            output["result_file_id"] = drive_upload(final_path, result_name, drive_token)
            output["result_name"] = result_name
            log(f"Uploaded result — file id {output['result_file_id']}")
            if debug_clips:
                try:
                    debug_video_path = os.path.join(work_dir, "debug.mp4")
                    concat_clips(debug_clips, debug_video_path)
                    output["debug_file_id"] = drive_upload(
                        debug_video_path, f"{video_id}_debug.mp4", drive_token)
                    log(f"Uploaded debug overlay — file id {output['debug_file_id']}")
                except Exception as e:
                    log(f"Debug overlay upload failed (non-fatal): {e}")
        elif final_size <= 15 * 1024 * 1024:
            log("Step 5: No Drive token — returning result inline (base64)")
            with open(final_path, "rb") as f:
                output["result_b64"] = base64.b64encode(f.read()).decode("ascii")
        else:
            return {"error": f"Result is {final_size / 1024**2:.1f} MB — too large to return inline "
                             "and no drive_access_token was provided"}

        log(f"Pipeline complete — total time: {t_total:.1f}s")
        return output

    except BaseException as exc:
        error = classify_worker_exception(
            exc, "clip" if mode == "clip" else "legacy", job_id)
        log(f"FAILED {error['code']} at {error['stage']}: {error['message']}")
        return {
            "status": "failed",
            "video_id": video_id,
            "worker_version": WORKER_VERSION,
            "job_id": job_id,
            "error": error,
        }
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)
        if downloaded:
            try:
                os.remove(video_path)
            except OSError:
                pass
        torch.cuda.empty_cache()


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
