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

DRIVE_API = "https://www.googleapis.com/drive/v3"
DRIVE_UPLOAD = "https://www.googleapis.com/upload/drive/v3/files"

# Inpainting crop margin around the ROI. Must comfortably exceed total mask
# dilation (pre-dilation + ProPainter flow dilation ≈ 1.5 * mask_dilation px).
CROP_MARGIN = 96
CROP_MIN_SIZE = 160


# ---------------------------------------------------------------------------
# Google Drive transfer (plain REST — no client libraries needed on the worker)
# ---------------------------------------------------------------------------

def drive_download(file_id: str, token: str, dest_path: str):
    import requests as req
    with req.get(f"{DRIVE_API}/files/{file_id}?alt=media",
                 headers={"Authorization": f"Bearer {token}"},
                 stream=True, timeout=600) as r:
        r.raise_for_status()
        with open(dest_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                f.write(chunk)


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


# ---------------------------------------------------------------------------
# OCR → text masks
# ---------------------------------------------------------------------------

_OCR = None


def _get_ocr():
    global _OCR
    if _OCR is None:
        from paddleocr import PaddleOCR
        _OCR = PaddleOCR(
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
            lang="en",
            text_det_thresh=0.3,
            text_det_box_thresh=0.4,
            text_det_unclip_ratio=1.5
        )
    return _OCR


def clamp_roi(roi: dict, width: int, height: int) -> dict:
    x = min(max(int(roi["x"]), 0), max(0, width - 1))
    y = min(max(int(roi["y"]), 0), max(0, height - 1))
    w = max(1, min(int(roi["w"]), width - x))
    h = max(1, min(int(roi["h"]), height - y))
    return {"x": x, "y": y, "w": w, "h": h}


def run_ocr(frames, roi: dict) -> list:
    """Run PaddleOCR text detection on keyframes within the ROI.
    Returns one full-frame uint8 mask (0/255) per frame."""
    try:
        ocr = _get_ocr()
    except ImportError:
        return _fallback_text_detection(frames, roi)

    x, y, w, h = roi["x"], roi["y"], roi["w"], roi["h"]
    mask_frames = []
    ocr_step = 5
    last_mask = None

    for i in range(len(frames)):
        frame_np = frames[i]
        mask = np.zeros(frame_np.shape[:2], dtype=np.uint8)

        if i % ocr_step == 0:
            roi_bgr = cv2.cvtColor(frame_np[y:y+h, x:x+w], cv2.COLOR_RGB2BGR)
            result = ocr.predict(roi_bgr)
            for res in result or []:
                polys = res.get("dt_polys") if hasattr(res, "get") else None
                if polys is None:
                    continue
                for box in polys:
                    box = np.asarray(box, dtype=np.int32).reshape(-1, 2)
                    box[:, 0] += x
                    box[:, 1] += y
                    cv2.fillPoly(mask, [box], 255)
            last_mask = mask
        elif last_mask is not None:
            mask = last_mask.copy()

        mask_frames.append(mask)

    return mask_frames


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
                 work_dir: str, log) -> str:
    """Full processing of a single clip: OCR → mask → crop → ProPainter → composite."""
    frames, fps = load_video_frames(clip_path)
    n, frame_h, frame_w = frames.shape[0], frames.shape[1], frames.shape[2]

    roi_c = clamp_roi(roi, frame_w, frame_h)
    masks = run_ocr(frames, roi_c)

    pre_dilate = max(1, mask_dilation // 2)
    kernel = np.ones((3, 3), np.uint8)
    masks = [cv2.dilate(m, kernel, iterations=pre_dilate) if m.any() else m for m in masks]

    result_path = os.path.join(work_dir, f"{clip_id}_out.mp4")

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
    return result_path


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


def handler(job):
    """RunPod serverless handler — full pipeline."""
    job_input = job["input"]

    video_id = job_input.get("video_id", "unknown")
    log = lambda msg: print(f"[GPU_WORKER] [{video_id}] {msg}", flush=True)

    log(f"Handler invoked — input keys: {list(job_input.keys())}")

    roi = job_input["roi"]
    mask_dilation = int(job_input.get("mask_dilation", 8))
    scene_threshold = float(job_input.get("scene_threshold", 27.0))
    min_scene_length = float(job_input.get("min_scene_length", 1.5))
    max_clip_duration = float(job_input.get("max_clip_duration", 12.0))
    drive_token = job_input.get("drive_access_token", "")
    log(f"Params — roi={roi} mask_dilation={mask_dilation} scene_threshold={scene_threshold} "
        f"min_scene_length={min_scene_length} max_clip_duration={max_clip_duration} "
        f"drive_token={'yes' if drive_token else 'no'}")

    # --- Acquire the source video ---------------------------------------
    video_file_id = job_input.get("video_file_id", "")
    video_download_url = job_input.get("video_download_url", "")
    video_data_b64 = job_input.get("video_data_b64", "")
    downloaded = False

    if video_file_id and drive_token:
        video_path = os.path.join("/tmp", f"{video_id}_{int(time.time())}.mp4")
        log(f"Downloading video from Drive (file id {video_file_id})...")
        drive_download(video_file_id, drive_token, video_path)
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
        return {"error": f"Video file not found: {video_path}"}
    log(f"Source video ready: {video_path} ({os.path.getsize(video_path) / 1024**2:.1f} MB)")

    work_dir = os.path.join("/tmp", f"work_{video_id}_{int(time.time())}")
    os.makedirs(work_dir, exist_ok=True)

    try:
        log("Loading ProPainter models...")
        init_models()
        log("Models loaded")

        t_start = time.time()
        info = probe_video(video_path)
        fps = info["fps"]

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
        for i, clip_path in enumerate(clip_paths):
            clip_id = f"clip_{i+1:04d}"
            log(f"Step 3.{i+1}/{len(clip_paths)}: Processing {clip_id}...")
            t_clip = time.time()
            result_path = process_clip(
                clip_path, roi, clip_id, mask_dilation,
                fix_raft, fix_flow_complete, propainter_model,
                work_dir, log
            )
            log(f"{clip_id} done in {time.time() - t_clip:.1f}s")
            result_clips.append(result_path)

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
        elif final_size <= 15 * 1024 * 1024:
            log("Step 5: No Drive token — returning result inline (base64)")
            with open(final_path, "rb") as f:
                output["result_b64"] = base64.b64encode(f.read()).decode("ascii")
        else:
            return {"error": f"Result is {final_size / 1024**2:.1f} MB — too large to return inline "
                             "and no drive_access_token was provided"}

        log(f"Pipeline complete — total time: {t_total:.1f}s")
        return output

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
