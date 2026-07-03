"""ElevenLabs Dubbing API client.

Sends the extracted audio track (not the video — smaller upload, no watermark
concerns) and polls until the dubbed track is ready. The API transcribes,
translates, and re-voices the audio while preserving the original timing, so
the returned track can be muxed over the video 1:1.
"""

import time
import requests

ELEVEN_BASE = "https://api.elevenlabs.io/v1"


class DubbingError(RuntimeError):
    pass


def dub_audio(audio_path: str, target_lang: str, api_key: str,
              output_path: str, timeout: int = 900,
              poll_interval: float = 5.0) -> str:
    """Dub an audio file into target_lang via the ElevenLabs Dubbing API.
    Returns output_path. Raises DubbingError on failure or timeout."""
    headers = {"xi-api-key": api_key}

    with open(audio_path, "rb") as f:
        resp = requests.post(
            f"{ELEVEN_BASE}/dubbing",
            headers=headers,
            files={"file": ("audio.mp3", f, "audio/mpeg")},
            data={
                "target_lang": target_lang,
                "source_lang": "auto",
                "num_speakers": "0",
                "watermark": "false",
            },
            timeout=300,
        )
    if resp.status_code != 200:
        raise DubbingError(f"dubbing job create failed "
                           f"({resp.status_code}): {resp.text[:500]}")
    dubbing_id = resp.json()["dubbing_id"]

    try:
        deadline = time.time() + timeout
        while True:
            status_resp = requests.get(
                f"{ELEVEN_BASE}/dubbing/{dubbing_id}",
                headers=headers, timeout=60)
            status_resp.raise_for_status()
            info = status_resp.json()
            status = info.get("status")
            if status == "dubbed":
                break
            if status == "failed":
                raise DubbingError(f"dubbing job failed: "
                                   f"{info.get('error', 'unknown error')}")
            if time.time() > deadline:
                raise DubbingError(f"dubbing job timed out after {timeout}s "
                                   f"(last status: {status})")
            time.sleep(poll_interval)

        audio_resp = requests.get(
            f"{ELEVEN_BASE}/dubbing/{dubbing_id}/audio/{target_lang}",
            headers=headers, stream=True, timeout=300)
        if audio_resp.status_code != 200:
            raise DubbingError(f"dubbed audio download failed "
                               f"({audio_resp.status_code}): "
                               f"{audio_resp.text[:500]}")
        with open(output_path, "wb") as f:
            for chunk in audio_resp.iter_content(chunk_size=1024 * 1024):
                f.write(chunk)
    finally:
        try:
            requests.delete(f"{ELEVEN_BASE}/dubbing/{dubbing_id}",
                            headers=headers, timeout=30)
        except Exception:
            pass  # cleanup is best-effort

    return output_path
