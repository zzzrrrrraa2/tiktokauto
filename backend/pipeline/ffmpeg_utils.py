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


def split_video(input_path: str, segments: list[tuple[float, float]], output_dir: str) -> list[str]:
    """Split video into clips. segments = [(start, end), ...]. Returns clip paths."""
    os.makedirs(output_dir, exist_ok=True)
    clip_paths = []
    for i, (start, end) in enumerate(segments):
        duration = end - start
        clip_path = os.path.join(output_dir, f"clip_{i+1:04d}.mp4")
        cmd = [
            "ffmpeg",
            "-y",
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
