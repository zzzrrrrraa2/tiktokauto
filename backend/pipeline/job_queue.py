import sqlite3
import json
import uuid
import time
from enum import Enum
from datetime import datetime
from typing import Optional


class JobStatus(str, Enum):
    QUEUED = "queued"
    SCENE_DETECTION = "scene_detection"
    SPLITTING = "splitting"
    GPU_PROCESSING = "gpu_processing"
    CONCATENATING = "concatenating"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class JobQueue:
    """SQLite-based job queue for video processing pipeline."""

    def __init__(self, db_path: str = "data/queue.db"):
        self.db_path = db_path
        self._init_db()

    def _get_conn(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self):
        with self._get_conn() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS videos (
                    video_id TEXT PRIMARY KEY,
                    original_filename TEXT NOT NULL,
                    video_path TEXT NOT NULL,
                    roi_x INTEGER,
                    roi_y INTEGER,
                    roi_w INTEGER,
                    roi_h INTEGER,
                    duration REAL,
                    width INTEGER,
                    height INTEGER,
                    fps REAL,
                    status TEXT DEFAULT 'uploaded',
                    created_at TEXT DEFAULT (datetime('now')),
                    updated_at TEXT DEFAULT (datetime('now'))
                );

                CREATE TABLE IF NOT EXISTS clips (
                    clip_id TEXT PRIMARY KEY,
                    video_id TEXT NOT NULL,
                    clip_index INTEGER NOT NULL,
                    clip_path TEXT,
                    start_time REAL,
                    end_time REAL,
                    duration REAL,
                    runpod_job_id TEXT,
                    result_path TEXT,
                    status TEXT DEFAULT 'pending',
                    error_message TEXT,
                    created_at TEXT DEFAULT (datetime('now')),
                    updated_at TEXT DEFAULT (datetime('now')),
                    FOREIGN KEY (video_id) REFERENCES videos(video_id)
                );

                CREATE TABLE IF NOT EXISTS job_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    video_id TEXT,
                    clip_id TEXT,
                    event TEXT NOT NULL,
                    details TEXT,
                    created_at TEXT DEFAULT (datetime('now'))
                );
            """)

    def create_video(self, original_filename: str, video_path: str,
                     duration: float, width: int, height: int, fps: float) -> str:
        video_id = str(uuid.uuid4())[:8]
        with self._get_conn() as conn:
            conn.execute(
                """INSERT INTO videos (video_id, original_filename, video_path, duration, width, height, fps, status)
                   VALUES (?, ?, ?, ?, ?, ?, ?, 'uploaded')""",
                (video_id, original_filename, video_path, duration, width, height, fps)
            )
        self._log(video_id, None, "video_uploaded",
                   f"File: {original_filename}, Duration: {duration}s, {width}x{height}")
        return video_id

    def set_roi(self, video_id: str, x: int, y: int, w: int, h: int):
        with self._get_conn() as conn:
            conn.execute(
                "UPDATE videos SET roi_x=?, roi_y=?, roi_w=?, roi_h=?, updated_at=datetime('now') WHERE video_id=?",
                (x, y, w, h, video_id)
            )
        self._log(video_id, None, "roi_set", f"ROI: x={x} y={y} w={w} h={h}")

    def get_video(self, video_id: str) -> Optional[dict]:
        with self._get_conn() as conn:
            row = conn.execute("SELECT * FROM videos WHERE video_id=?", (video_id,)).fetchone()
            return dict(row) if row else None

    def list_videos(self, limit: int = 20) -> list[dict]:
        with self._get_conn() as conn:
            rows = conn.execute(
                "SELECT * FROM videos ORDER BY created_at DESC LIMIT ?", (limit,)
            ).fetchall()
            return [dict(r) for r in rows]

    def create_clip(self, video_id: str, clip_index: int, start_time: float,
                    end_time: float, clip_path: str = None) -> str:
        clip_id = f"{video_id}_c{clip_index:04d}"
        duration = end_time - start_time
        with self._get_conn() as conn:
            conn.execute(
                """INSERT INTO clips (clip_id, video_id, clip_index, clip_path, start_time, end_time, duration, status)
                   VALUES (?, ?, ?, ?, ?, ?, ?, 'pending')""",
                (clip_id, video_id, clip_index, clip_path, start_time, end_time, duration)
            )
        return clip_id

    def create_clips_batch(self, video_id: str, segments: list[tuple[float, float]],
                           clip_paths: list[str]) -> list[str]:
        clip_ids = []
        with self._get_conn() as conn:
            for i, ((start, end), path) in enumerate(zip(segments, clip_paths)):
                clip_id = f"{video_id}_c{i+1:04d}"
                duration = end - start
                conn.execute(
                    """INSERT OR REPLACE INTO clips (clip_id, video_id, clip_index, clip_path, start_time, end_time, duration, status)
                       VALUES (?, ?, ?, ?, ?, ?, ?, 'pending')""",
                    (clip_id, video_id, i + 1, path, start, end, duration)
                )
                clip_ids.append(clip_id)
        self._log(video_id, None, "clips_created", f"{len(clip_ids)} clips")
        return clip_ids

    def clear_clips(self, video_id: str):
        """Remove all clip rows for a video (fresh rows are created per run;
        stale ones from a previous run with more clips would skew progress)."""
        with self._get_conn() as conn:
            conn.execute("DELETE FROM clips WHERE video_id=?", (video_id,))

    def update_clip_status(self, clip_id: str, status: str, runpod_job_id: str = None,
                           result_path: str = None, error_message: str = None):
        with self._get_conn() as conn:
            conn.execute(
                """UPDATE clips SET status=?, runpod_job_id=?, result_path=?, error_message=?, updated_at=datetime('now')
                   WHERE clip_id=?""",
                (status, runpod_job_id, result_path, error_message, clip_id)
            )

    def get_clips(self, video_id: str) -> list[dict]:
        with self._get_conn() as conn:
            rows = conn.execute(
                "SELECT * FROM clips WHERE video_id=? ORDER BY clip_index",
                (video_id,)
            ).fetchall()
            return [dict(r) for r in rows]

    def get_clip(self, clip_id: str) -> Optional[dict]:
        with self._get_conn() as conn:
            row = conn.execute("SELECT * FROM clips WHERE clip_id=?", (clip_id,)).fetchone()
            return dict(row) if row else None

    def get_video_status_summary(self, video_id: str) -> dict:
        clips = self.get_clips(video_id)
        if not clips:
            video = self.get_video(video_id)
            return {"status": "uploaded", "total": 0, "completed": 0, "processing": 0, "failed": 0, "pending": 0, "clips": []}

        statuses = {
            "completed": sum(1 for c in clips if c["status"] == "completed"),
            "failed": sum(1 for c in clips if c["status"] == "failed"),
            "processing": sum(1 for c in clips if c["status"] in ("processing", "gpu_processing")),
            "pending": sum(1 for c in clips if c["status"] == "pending"),
        }

        if statuses["failed"] > 0:
            overall = "failed"
        elif statuses["completed"] == len(clips):
            overall = "completed"
        elif statuses["processing"] > 0 or statuses["pending"] > 0:
            overall = "processing"
        else:
            overall = "unknown"

        return {
            "status": overall,
            "total": len(clips),
            "completed": statuses["completed"],
            "processing": statuses["processing"],
            "failed": statuses["failed"],
            "pending": statuses["pending"],
            "clips": clips
        }

    def update_video_status(self, video_id: str, status: str):
        with self._get_conn() as conn:
            conn.execute(
                "UPDATE videos SET status=?, updated_at=datetime('now') WHERE video_id=?",
                (status, video_id)
            )

    def set_video_result(self, video_id: str, final_path: str):
        with self._get_conn() as conn:
            conn.execute(
                "UPDATE videos SET status='completed', updated_at=datetime('now') WHERE video_id=?",
                (video_id,)
            )
        self._log(video_id, None, "video_completed", f"Result: {final_path}")

    def cancel_video(self, video_id: str):
        with self._get_conn() as conn:
            conn.execute("UPDATE videos SET status='cancelled' WHERE video_id=?", (video_id,))
            conn.execute(
                "UPDATE clips SET status='cancelled' WHERE video_id=? AND status IN ('pending', 'processing')",
                (video_id,)
            )
        self._log(video_id, None, "video_cancelled", "")

    def _log(self, video_id: str, clip_id: Optional[str], event: str, details: str):
        with self._get_conn() as conn:
            conn.execute(
                "INSERT INTO job_log (video_id, clip_id, event, details) VALUES (?, ?, ?, ?)",
                (video_id, clip_id, event, details)
            )

    def get_logs(self, video_id: str, limit: int = 50) -> list[dict]:
        with self._get_conn() as conn:
            rows = conn.execute(
                "SELECT * FROM job_log WHERE video_id=? ORDER BY created_at DESC LIMIT ?",
                (video_id, limit)
            ).fetchall()
            return [dict(r) for r in rows]
