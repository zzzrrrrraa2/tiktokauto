#!/usr/bin/env python3
"""RunPod Serverless Handler — Full Video Caption Removal Pipeline.

Receives: {video_data_b64 or video_download_url or video_path, roi: {x,y,w,h}, fps, video_id, mask_dilation, scene_threshold, min_scene_length}

Pipeline (all on GPU):
  1. PySceneDetect → detect scene boundaries
  2. ffmpeg → split video into clips
  3. Per clip: PaddleOCR → mask generation → ProPainter inpainting
  4. ffmpeg → concatenate all clips
  5. Return result video path
"""

import os
import sys
import json
import time
import subprocess
import shutil
import base64
import fractions
import numpy as np
import cv2
from PIL import Image
from pathlib import Path
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
            subprocess.run(["wget", "-q", "--show-progress", "-O", path, url], check=True)
            print(f"[GPU_WORKER] Downloaded {os.path.basename(path)} ({os.path.getsize(path) / 1024**2:.1f} MB)", flush=True)

_download_weights()

sys.path.insert(0, PROPAINTER_REPO)

from model.modules.flow_comp_raft import RAFT_bi
from model.recurrent_flow_completion import RecurrentFlowCompleteNet
from model.propainter import InpaintGenerator
from core.utils import to_tensors

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_HALF = DEVICE.type == "cuda"


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
    return {
        "width": int(video_stream["width"]),
        "height": int(video_stream["height"]),
        "fps": float(fractions.Fraction(video_stream.get("r_frame_rate", "30/1"))),
        "duration": float(data["format"].get("duration", 0)),
    }


def detect_scenes(video_path: str, threshold: float = 27.0,
                   min_scene_length: float = 1.5) -> list[float]:
    """Detect scene boundaries using PySceneDetect. Returns list of cut timestamps."""
    from scenedetect import open_video, SceneManager
    from scenedetect.detectors import ContentDetector

    video = open_video(video_path)
    scene_manager = SceneManager()
    scene_manager.add_detector(ContentDetector(threshold=threshold))
    scene_manager.detect_scenes(video)

    scene_list = scene_manager.get_scene_list()
    cuts = [0.0]
    for start, end in scene_list:
        cuts.append(end.get_seconds())

    fps = probe_video(video_path)["fps"]
    min_frames = int(min_scene_length * fps)
    merged = [cuts[0]]
    for i in range(1, len(cuts) - 1):
        if (cuts[i] - merged[-1]) * fps >= min_frames:
            merged.append(cuts[i])
    merged.append(cuts[-1])

    return merged


def split_video(input_path: str, segments: list[tuple[float, float]],
                output_dir: str) -> list[str]:
    """Split video into clips. Returns clip paths."""
    os.makedirs(output_dir, exist_ok=True)
    clip_paths = []
    for i, (start, end) in enumerate(segments):
        duration = end - start
        clip_path = os.path.join(output_dir, f"clip_{i+1:04d}.mp4")
        cmd = [
            "ffmpeg", "-y",
            "-ss", str(start),
            "-i", input_path,
            "-t", str(duration),
            "-c", "copy",
            "-avoid_negative_ts", "make_zero",
            clip_path
        ]
        subprocess.run(cmd, capture_output=True, check=True)
        clip_paths.append(clip_path)
    return clip_paths


def concat_clips(clip_paths: list[str], output_path: str):
    """Concatenate clips using ffmpeg concat demuxer."""
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
    subprocess.run(cmd, capture_output=True, check=True)
    os.remove(concat_file)


def load_frames_as_pil(video_path: str) -> tuple:
    vframes, aframes, info = torchvision.io.read_video(
        filename=video_path, pts_unit="sec"
    )
    frames = [Image.fromarray(f.numpy()) for f in vframes]
    fps = info["video_fps"]
    return frames, fps


def run_ocr(frames: list, roi: dict) -> list:
    """Run PaddleOCR on keyframes within ROI. Returns list of mask arrays."""
    try:
        from paddleocr import PaddleOCR
    except ImportError:
        return _fallback_text_detection(frames, roi)

    ocr = PaddleOCR(
        use_angle_cls=False,
        lang="en",
        show_log=False,
        det_db_thresh=0.3,
        det_db_box_thresh=0.4,
        det_db_unclip_ratio=1.5
    )

    x, y, w, h = roi["x"], roi["y"], roi["w"], roi["h"]
    mask_frames = []
    ocr_step = 5
    last_mask = None

    for i, frame in enumerate(frames):
        frame_np = np.array(frame)
        mask = np.zeros((frame_np.shape[0], frame_np.shape[1]), dtype=np.uint8)
        roi_region = frame_np[y:y+h, x:x+w]

        if roi_region.size == 0:
            mask_frames.append(mask)
            continue

        if i % ocr_step == 0:
            result = ocr.ocr(roi_region, cls=False)
            last_mask = mask.copy()
            if result and result[0]:
                for line in result[0]:
                    box = line[0]
                    box = np.array(box).astype(np.int32)
                    box[:, 0] += x
                    box[:, 1] += y
                    cv2.fillPoly(mask, [box], 255)
                    cv2.fillPoly(last_mask, [box], 255)
            mask_frames.append(mask)
        elif last_mask is not None:
            mask_frames.append(last_mask.copy())
        else:
            mask_frames.append(mask)

    return mask_frames


def _fallback_text_detection(frames: list, roi: dict) -> list:
    x, y, w, h = roi["x"], roi["y"], roi["w"], roi["h"]
    mask_frames = []

    for frame in frames:
        frame_np = np.array(frame)
        mask = np.zeros((frame_np.shape[0], frame_np.shape[1]), dtype=np.uint8)
        roi_region = frame_np[y:y+h, x:x+w]
        if roi_region.size == 0:
            mask_frames.append(mask)
            continue

        gray = cv2.cvtColor(roi_region, cv2.COLOR_RGB2GRAY)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 3))
        dilated = cv2.dilate(binary, kernel, iterations=3)
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            cx, cy, cw, ch = cv2.boundingRect(cnt)
            if cw > 30 and ch > 8:
                cv2.rectangle(mask,
                    (x + cx, y + cy),
                    (x + cx + cw, y + cy + ch),
                    255, -1)

        mask_frames.append(mask)

    return mask_frames


def dilate_masks(masks: list, iterations: int = 4) -> list:
    import scipy.ndimage
    dilated = []
    for mask in masks:
        if iterations > 0:
            d = scipy.ndimage.binary_dilation(mask, iterations=iterations).astype(np.uint8)
        else:
            d = (mask > 0).astype(np.uint8)
        dilated.append(Image.fromarray(d * 255))
    return dilated


def run_propainter(clip_path: str, mask_dir: str, output_dir: str,
                   fix_raft, fix_flow_complete, model,
                   mask_dilation: int = 8) -> str:
    """Run ProPainter inpainting. Returns result video path."""
    frames, fps = load_frames_as_pil(clip_path)
    video_length = len(frames)

    h, w = frames[0].size[1], frames[0].size[0]
    process_h = h - h % 8
    process_w = w - w % 8
    frames = [f.resize((process_w, process_h)) for f in frames]

    mask_files = sorted(Path(mask_dir).glob("*.png"))
    masks_pil = [Image.open(p).convert("L").resize((process_w, process_h), Image.NEAREST)
                 for p in mask_files]

    import scipy.ndimage
    flow_masks = []
    masks_dilated = []
    for m in masks_pil:
        m_arr = np.array(m)
        flow_m = scipy.ndimage.binary_dilation(m_arr, iterations=mask_dilation).astype(np.uint8)
        flow_masks.append(Image.fromarray(flow_m * 255))
        dilated_m = scipy.ndimage.binary_dilation(m_arr, iterations=mask_dilation).astype(np.uint8)
        masks_dilated.append(Image.fromarray(dilated_m * 255))

    frames_t = to_tensors()(frames).unsqueeze(0) * 2 - 1
    flow_masks_t = to_tensors()(flow_masks).unsqueeze(0)
    masks_dilated_t = to_tensors()(masks_dilated).unsqueeze(0)
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
            masks_dilated_t = masks_dilated_t.half()
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

    ori_frames = [np.array(f).astype(np.uint8) for f in frames]
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

    os.makedirs(output_dir, exist_ok=True)
    result_path = os.path.join(output_dir, "inpaint_out.mp4")

    import imageio
    imageio.mimwrite(result_path, comp_frames, fps=fps, quality=7)

    return result_path


def process_clip(clip_path: str, roi: dict, clip_id: str, mask_dilation: int,
                 fix_raft, fix_flow_complete, propainter_model,
                 work_dir: str) -> str:
    """Full processing of a single clip: OCR → mask → ProPainter."""
    clip_work = os.path.join(work_dir, clip_id)
    os.makedirs(clip_work, exist_ok=True)

    frames, fps = load_frames_as_pil(clip_path)

    mask_arrays = run_ocr(frames, roi)
    masks_pil = dilate_masks(mask_arrays, iterations=mask_dilation // 2)

    mask_dir = os.path.join(clip_work, "masks")
    os.makedirs(mask_dir, exist_ok=True)
    for i, m in enumerate(masks_pil):
        m.save(os.path.join(mask_dir, f"mask_{i:06d}.png"))

    output_dir = os.path.join(clip_work, "output")
    result_path = run_propainter(
        clip_path, mask_dir, output_dir,
        fix_raft, fix_flow_complete, propainter_model,
        mask_dilation=mask_dilation
    )

    torch.cuda.empty_cache()
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
    mask_dilation = job_input.get("mask_dilation", 8)
    scene_threshold = job_input.get("scene_threshold", 27.0)
    min_scene_length = job_input.get("min_scene_length", 1.5)
    log(f"Params — roi={roi} mask_dilation={mask_dilation} scene_threshold={scene_threshold} min_scene_length={min_scene_length}")

    video_data_b64 = job_input.get("video_data_b64", "")
    video_download_url = job_input.get("video_download_url", "")

    if video_data_b64:
        b64_len = len(video_data_b64)
        log(f"Decoding base64 video ({b64_len} chars)...")
        video_path = os.path.join("/tmp", f"{video_id}_{int(time.time())}.mp4")
        with open(video_path, "wb") as f:
            f.write(base64.b64decode(video_data_b64))
        file_size = os.path.getsize(video_path) / (1024 * 1024)
        log(f"Video decoded to {video_path} ({file_size:.1f} MB)")
    elif video_download_url:
        log(f"Downloading video from URL: {video_download_url}")
        video_path = os.path.join("/tmp", f"{video_id}_{int(time.time())}.mp4")
        import requests as req
        resp = req.get(video_download_url, stream=True, timeout=300)
        resp.raise_for_status()
        with open(video_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=8192):
                f.write(chunk)
        file_size = os.path.getsize(video_path) / (1024 * 1024)
        log(f"Video downloaded to {video_path} ({file_size:.1f} MB)")
    else:
        video_path = job_input.get("video_path", "")
        log(f"Using direct video_path: {video_path}")

    if not video_path or not os.path.exists(video_path):
        log(f"ERROR: Video file not found at {video_path}")
        return {"error": f"Video file not found: {video_path}"}

    work_dir = f"/tmp/{video_id}"
    os.makedirs(work_dir, exist_ok=True)

    log("Loading ProPainter models...")
    init_models()
    log("Models loaded")

    t_start = time.time()

    # Step 1: Scene detection
    log("Step 1: Scene detection...")
    cuts = detect_scenes(video_path, threshold=scene_threshold,
                         min_scene_length=min_scene_length)
    segments = [(cuts[i], cuts[i + 1]) for i in range(len(cuts) - 1)]
    log(f"Scene detection done — {len(cuts)-1} cuts found, {len(segments)} segments")

    # Step 2: Split video into clips
    log("Step 2: Splitting video into clips...")
    clip_dir = os.path.join(work_dir, "clips")
    clip_paths = split_video(video_path, segments, clip_dir)
    log(f"Splitting done — {len(clip_paths)} clips created")

    # Step 3: Process each clip
    result_clips = []
    for i, clip_path in enumerate(clip_paths):
        clip_id = f"clip_{i+1:04d}"
        clip_size = os.path.getsize(clip_path) / (1024 * 1024)
        log(f"Step 3.{i+1}/{len(clip_paths)}: Processing clip {clip_id} ({clip_size:.1f} MB)...")
        t_clip = time.time()
        result_path = process_clip(
            clip_path, roi, clip_id, mask_dilation,
            fix_raft, fix_flow_complete, propainter_model,
            work_dir
        )
        log(f"Clip {clip_id} done in {time.time() - t_clip:.1f}s")
        result_clips.append(result_path)

    # Step 4: Concatenate results
    log("Step 4: Concatenating clips...")
    final_path = os.path.join(work_dir, "final.mp4")
    concat_clips(result_clips, final_path)
    final_size = os.path.getsize(final_path) / (1024 * 1024)
    log(f"Concatenation done — final video: {final_path} ({final_size:.1f} MB)")

    t_total = time.time() - t_start
    log(f"Pipeline complete — total time: {t_total:.1f}s")

    torch.cuda.empty_cache()

    return {
        "result_path": final_path,
        "video_id": video_id,
        "num_clips": len(clip_paths),
        "total_time": round(t_total, 1),
        "status": "completed"
    }


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
