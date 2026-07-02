"""Scene detection for the backend's parallel-clip fan-out.

Ported verbatim from gpu_worker/handler.py so backend-side splits produce the
same segments the worker would have produced internally. Keep the two in sync
if detection logic changes.
"""
import math

from .ffmpeg_utils import probe_video


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
