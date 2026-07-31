"""Polish headline extraction helpers used by the first clip job.

This module deliberately has no PaddleOCR dependency so its geometry and colour
logic can be unit-tested without loading the worker's GPU stack.
"""

from __future__ import annotations

import math
import re
from typing import Callable

import cv2
import numpy as np


def _result_items(results) -> list[dict]:
    items = []
    for result in results or []:
        data = result
        if not hasattr(data, "get") and hasattr(result, "json"):
            data = result.json() if callable(result.json) else result.json
        if not hasattr(data, "get"):
            continue
        if isinstance(data.get("res"), dict):
            data = data["res"]
        texts = data.get("rec_texts") or []
        scores = data.get("rec_scores") or []
        boxes = data.get("rec_polys")
        if boxes is None:
            boxes = data.get("dt_polys")
        for index, text in enumerate(texts):
            if index >= len(scores) or boxes is None or index >= len(boxes):
                continue
            text = " ".join(str(text).split())
            if not text:
                continue
            pts = np.asarray(boxes[index], dtype=np.float32).reshape(-1, 2)
            items.append({
                "text": text,
                "confidence": float(scores[index]),
                "points": pts,
                "x0": float(pts[:, 0].min()),
                "y0": float(pts[:, 1].min()),
                "x1": float(pts[:, 0].max()),
                "y1": float(pts[:, 1].max()),
            })
    return items


def _reading_order(items: list[dict]) -> list[dict]:
    """Order OCR words top-to-bottom, then left-to-right within each line."""
    ordered = sorted(items, key=lambda item: (item["y0"], item["x0"]))
    lines: list[list[dict]] = []
    for item in ordered:
        cy = (item["y0"] + item["y1"]) / 2
        height = max(1.0, item["y1"] - item["y0"])
        target = None
        for line in lines:
            line_cy = sum((x["y0"] + x["y1"]) / 2 for x in line) / len(line)
            line_h = max(max(1.0, x["y1"] - x["y0"]) for x in line)
            if abs(cy - line_cy) <= max(height, line_h) * 0.55:
                target = line
                break
        (target if target is not None else lines.append([]) or lines[-1]).append(item)
    return [
        item
        for line in sorted(lines, key=lambda line: min(x["y0"] for x in line))
        for item in sorted(line, key=lambda item: item["x0"])
    ]


def _word_geometry(items: list[dict]) -> list[dict]:
    """Approximate per-word boxes when recognition returns one box per line."""
    words = []
    for item in items:
        matches = list(re.finditer(r"\S+", item["text"]))
        if len(matches) <= 1:
            words.append(item)
            continue
        text_length = max(1, len(item["text"]))
        span = item["x1"] - item["x0"]
        for match in matches:
            x0 = item["x0"] + span * match.start() / text_length
            x1 = item["x0"] + span * match.end() / text_length
            word = dict(item)
            word.update({
                "text": match.group(),
                "x0": x0,
                "x1": x1,
                "points": np.asarray([
                    [x0, item["y0"]], [x1, item["y0"]],
                    [x1, item["y1"]], [x0, item["y1"]],
                ], dtype=np.float32),
            })
            words.append(word)
    return words


def classify_colour(rgb: np.ndarray, points: np.ndarray) -> str | None:
    """Classify title glyph pixels as the source's white or red group."""
    height, width = rgb.shape[:2]
    polygon = np.round(points).astype(np.int32)
    polygon[:, 0] = np.clip(polygon[:, 0], 0, width - 1)
    polygon[:, 1] = np.clip(polygon[:, 1], 0, height - 1)
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.fillPoly(mask, [polygon], 255)
    pixels = rgb[mask > 0].astype(np.int16)
    if not len(pixels):
        return None
    red = (
        (pixels[:, 0] >= 145)
        & (pixels[:, 0] >= pixels[:, 1] * 1.30)
        & (pixels[:, 0] >= pixels[:, 2] * 1.20)
    )
    white = (
        (pixels.min(axis=1) >= 155)
        & ((pixels.max(axis=1) - pixels.min(axis=1)) <= 75)
    )
    red_count = int(red.sum())
    white_count = int(white.sum())
    minimum = max(2, int(math.ceil(len(pixels) * 0.01)))
    if max(red_count, white_count) < minimum:
        return None
    return "red" if red_count > white_count else "white"


def extract_headline(
    frames: np.ndarray,
    settings: dict,
    predict: Callable[[np.ndarray], object],
) -> dict:
    """Select the best of three Polish OCR readings containing both colours."""
    if frames is None or not len(frames):
        return {"status": "failed", "reason": "no_frames"}
    count, height, width = frames.shape[:3]
    sample_count = max(1, int(settings.get("sample_frames", 3)))
    indices = sorted(set(
        int(round(i * (count - 1) / max(1, sample_count - 1)))
        for i in range(sample_count)
    ))
    top_fraction = float(settings.get("search_top_fraction", 0.16))
    search_height = max(1, min(height, round(height * top_fraction)))
    minimum_confidence = float(settings.get("min_confidence", 0.70))
    candidates = []
    for frame_index in indices:
        rgb = frames[frame_index, :search_height]
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        items = _reading_order(_word_geometry(_result_items(predict(bgr))))
        for item in items:
            item["colour"] = classify_colour(rgb, item["points"])
        items = [item for item in items if item["colour"]]
        colours = {item["colour"] for item in items}
        if not items or colours != {"white", "red"}:
            continue
        confidence = sum(item["confidence"] for item in items) / len(items)
        if confidence < minimum_confidence:
            continue
        segments = []
        for item in items:
            if segments and segments[-1]["color"] == item["colour"]:
                segments[-1]["text"] += " " + item["text"]
            else:
                segments.append({"color": item["colour"], "text": item["text"]})
        candidates.append({
            "status": "ok",
            "segments": segments,
            "white_pl": " ".join(
                item["text"] for item in items if item["colour"] == "white"),
            "red_pl": " ".join(
                item["text"] for item in items if item["colour"] == "red"),
            "order": [segment["color"] for segment in segments],
            "confidence": round(confidence, 4),
            "bounds": {
                "x": round(min(item["x0"] for item in items)),
                "y": round(min(item["y0"] for item in items)),
                "w": round(max(item["x1"] for item in items)
                           - min(item["x0"] for item in items)),
                "h": round(max(item["y1"] for item in items)
                           - min(item["y0"] for item in items)),
            },
            "sample_frame": frame_index,
        })
    if not candidates:
        return {
            "status": "failed",
            "reason": "no_confident_white_red_reading",
            "sampled_frames": indices,
        }
    return max(candidates, key=lambda item: item["confidence"])
