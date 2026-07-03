import subprocess
import json
import os
import fractions
from pathlib import Path
from typing import Optional


def probe_video(video_path: str) -> dict:
    """Get video metadata using ffprobe."""
    cmd = [
        "ffprobe",
        "-v", "quiet",
        "-print_format", "json",
        "-show_format",
        "-show_streams",
        video_path
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe failed: {result.stderr}")

    data = json.loads(result.stdout)
    video_stream = None
    for stream in data.get("streams", []):
        if stream.get("codec_type") == "video":
            video_stream = stream
            break

    if not video_stream:
        raise ValueError("No video stream found")

    return {
        "width": int(video_stream["width"]),
        "height": int(video_stream["height"]),
        "fps": float(fractions.Fraction(video_stream.get("r_frame_rate", "30/1"))),
        "duration": float(data["format"].get("duration", 0)),
        "codec": video_stream.get("codec_name", "h264"),
        "bitrate": int(data["format"].get("bit_rate", 0)),
    }


def extract_frame(video_path: str, time_seconds: float, output_path: str):
    """Extract a single frame at given time."""
    cmd = [
        "ffmpeg",
        "-y",
        "-ss", str(time_seconds),
        "-i", video_path,
        "-vframes", "1",
        "-q:v", "2",
        output_path
    ]
    subprocess.run(cmd, capture_output=True, check=True)


def extract_frames(video_path: str, output_dir: str, fps: Optional[float] = None) -> list[str]:
    """Extract all frames from video. Returns list of frame paths."""
    os.makedirs(output_dir, exist_ok=True)
    fps_filter = f"fps={fps}," if fps else ""
    cmd = [
        "ffmpeg",
        "-y",
        "-i", video_path,
        "-vf", f"{fps_filter}format=rgb24",
        f"{output_dir}/frame_%06d.png"
    ]
    subprocess.run(cmd, capture_output=True, check=True)
    frames = sorted(Path(output_dir).glob("frame_*.png"))
    return [str(f) for f in frames]


def frames_to_video(frame_dir: str, output_path: str, fps: float, codec: str = "libx264", crf: int = 18):
    """Convert frame sequence to video."""
    cmd = [
        "ffmpeg",
        "-y",
        "-framerate", str(fps),
        "-i", f"{frame_dir}/frame_%06d.png",
        "-c:v", codec,
        "-crf", str(crf),
        "-pix_fmt", "yuv420p",
        output_path
    ]
    subprocess.run(cmd, capture_output=True, check=True)


def split_video(input_path: str, segments: list[tuple[float, float]], output_dir: str,
                fps: float) -> list[str]:
    """Split video into frame-accurate clips (re-encoded, video-only, CFR).
    Ported from gpu_worker/handler.py — stream-copy cuts snap to keyframes and
    are unsafe to concat; re-encoding keeps cuts exact and the concat demuxer
    happy. Audio is dropped here and muxed back from the original at the end."""
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
            raise RuntimeError(f"ffmpeg split failed for segment {i}: "
                               f"{proc.stderr.decode(errors='replace')[-2000:]}")
        clip_paths.append(clip_path)
    return clip_paths


def concat_clips(clip_paths: list[str], output_path: str):
    """Concatenate clips into one video using ffmpeg concat demuxer."""
    concat_file = os.path.join(os.path.dirname(output_path), "_concat_list.txt")
    with open(concat_file, "w") as f:
        for path in clip_paths:
            f.write(f"file '{os.path.abspath(path)}'\n")

    cmd = [
        "ffmpeg",
        "-y",
        "-f", "concat",
        "-safe", "0",
        "-i", concat_file,
        "-c", "copy",
        output_path
    ]
    subprocess.run(cmd, capture_output=True, check=True)
    os.remove(concat_file)


def extract_audio(video_path: str, output_path: str) -> Optional[str]:
    """Extract the audio track as mp3 (small upload for the dubbing API).
    Returns output_path, or None if the source has no audio stream."""
    probe = subprocess.run(
        ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_streams",
         "-select_streams", "a", video_path],
        capture_output=True, text=True)
    if probe.returncode != 0:
        raise RuntimeError(f"ffprobe failed: {probe.stderr}")
    if not json.loads(probe.stdout).get("streams"):
        return None

    cmd = [
        "ffmpeg", "-y",
        "-i", video_path,
        "-vn",
        "-c:a", "libmp3lame", "-b:a", "192k",
        output_path
    ]
    proc = subprocess.run(cmd, capture_output=True)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg audio extract failed: "
                           f"{proc.stderr.decode(errors='replace')[-2000:]}")
    return output_path


def pair_swap_order(n: int) -> list[int]:
    """0-based output order that keeps clip 0 first and swaps each subsequent
    pair: n=6 -> [0, 2, 1, 4, 3, 5]; n=7 -> [0, 2, 1, 4, 3, 6, 5]. An odd
    tail clip stays in place; n <= 2 is identity."""
    order = [0] if n > 0 else []
    i = 1
    while i + 1 < n:
        order += [i + 1, i]
        i += 2
    if i < n:
        order.append(i)
    return order


def blur_clip_edges(clip_path: str, output_path: str, fps: float,
                    blur_duration: float = 0.15,
                    blur_start: bool = True, blur_end: bool = True,
                    strength: int = 10):
    """Re-encode a clip with a short blur on its first/last blur_duration
    seconds, smoothing the hard cuts created by clip shuffling. Duration is
    unchanged, so audio timing is unaffected."""
    conds = []
    if blur_start:
        conds.append(f"lt(t,{blur_duration:.3f})")
    if blur_end:
        dur = probe_video(clip_path)["duration"]
        conds.append(f"gt(t,{dur - blur_duration:.3f})")
    if not conds:
        raise ValueError("blur_start and blur_end are both False")
    vf = (f"boxblur=luma_radius={strength}:luma_power=2:"
          f"chroma_radius={strength // 2}:"
          f"enable='{'+'.join(conds)}'")
    cmd = [
        "ffmpeg", "-y",
        "-i", clip_path,
        "-vf", vf,
        "-an",
        "-vsync", "cfr", "-r", f"{fps}",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
        "-pix_fmt", "yuv420p",
        output_path
    ]
    proc = subprocess.run(cmd, capture_output=True)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg edge blur failed: "
                           f"{proc.stderr.decode(errors='replace')[-2000:]}")


def mux_audio(video_path: str, audio_source_path: str, output_path: str,
              subtitles_path: Optional[str] = None):
    """Copy the processed video stream and add the original audio (if any).
    Ported from gpu_worker/handler.py — the '?' makes the audio map optional so
    audio-less sources still mux cleanly. With subtitles_path set, burns the
    ASS captions in (requires a re-encode instead of the stream copy); the
    subprocess runs with cwd = the .ass file's dir so the libass filter arg
    needs no Windows drive-colon escaping."""
    if subtitles_path:
        cwd = os.path.dirname(os.path.abspath(subtitles_path))
        video_args = ["-vf", f"ass={os.path.basename(subtitles_path)}",
                      "-c:v", "libx264", "-preset", "medium", "-crf", "18",
                      "-pix_fmt", "yuv420p"]
    else:
        cwd = None
        video_args = ["-c:v", "copy"]
    cmd = [
        "ffmpeg", "-y",
        "-i", os.path.abspath(video_path),
        "-i", os.path.abspath(audio_source_path),
        "-map", "0:v:0", "-map", "1:a:0?",
        *video_args,
        "-c:a", "aac", "-b:a", "192k",
        "-shortest", "-movflags", "+faststart",
        os.path.abspath(output_path)
    ]
    proc = subprocess.run(cmd, capture_output=True, cwd=cwd)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg audio mux failed: "
                           f"{proc.stderr.decode(errors='replace')[-2000:]}")


def resize_video(input_path: str, output_path: str, width: int, height: int):
    """Resize video to target dimensions."""
    cmd = [
        "ffmpeg",
        "-y",
        "-i", input_path,
        "-vf", f"scale={width}:{height}",
        "-c:v", "libx264",
        "-crf", "18",
        "-preset", "fast",
        "-c:a", "copy",
        output_path
    ]
    subprocess.run(cmd, capture_output=True, check=True)
