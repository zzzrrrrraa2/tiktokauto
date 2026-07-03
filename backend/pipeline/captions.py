"""One-word TikTok-style animated captions.

Transcribes the final audio track with faster-whisper (word timestamps) and
renders an ASS subtitle file: one word at a time, ALL CAPS, centered in the
user's drawn ROI (where the original captions were inpainted out), with a
pop-in scale animation. Burned into the video by mux_audio via libass.
"""

import string
from typing import Optional

_PUNCT = string.punctuation + "„""«»…¿¡"


def transcribe_words(audio_path: str, model_size: str = "small",
                     language: Optional[str] = None) -> list[tuple[str, float, float]]:
    """Transcribe audio to a flat [(WORD, start_s, end_s), ...] list.
    language=None lets whisper auto-detect (used when dubbing fell back to
    the original audio)."""
    from faster_whisper import WhisperModel  # lazy: ~60s model load, only when enabled

    model = WhisperModel(model_size, device="cpu", compute_type="int8")
    segments, _info = model.transcribe(audio_path, word_timestamps=True,
                                       language=language, vad_filter=True)
    words = []
    for seg in segments:
        for w in seg.words or []:
            text = w.word.strip().strip(_PUNCT).upper()
            if text:
                words.append((text, w.start, w.end))
    return words


def _ass_time(t: float) -> str:
    t = max(t, 0.0)
    cs = int(round(t * 100))
    h, rem = divmod(cs, 360000)
    m, rem = divmod(rem, 6000)
    s, cs = divmod(rem, 100)
    return f"{h}:{m:02d}:{s:02d}.{cs:02d}"


def build_ass(words: list[tuple[str, float, float]], video_w: int, video_h: int,
              roi: dict, font: str, font_size: int, out_path: str) -> str:
    """Write an ASS file: each word visible start→end (extended to the next
    word when the gap is small), pop animation 75% → 115% → 100%."""
    cx = round(roi["x"] + roi["w"] / 2)
    cy = round(roi["y"] + roi["h"] / 2)

    header = f"""[Script Info]
ScriptType: v4.00+
PlayResX: {video_w}
PlayResY: {video_h}
WrapStyle: 2
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Word,{font},{font_size},&H00FFFFFF,&H00FFFFFF,&H80000000,&H80000000,-1,0,0,0,100,100,0,0,1,1,2,5,10,10,10,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    lines = []
    for i, (text, start, end) in enumerate(words):
        if i + 1 < len(words):
            gap = words[i + 1][1] - end
            if 0 < gap < 0.3:
                end = words[i + 1][1]  # hold until next word: no flicker
        end = max(end, start + 0.10)
        dur_ms = int((end - start) * 1000)
        t1 = min(90, dur_ms // 2)
        t2 = min(180, dur_ms)
        tags = (f"{{\\an5\\pos({cx},{cy})\\fscx75\\fscy75"
                f"\\t(0,{t1},\\fscx115\\fscy115)"
                f"\\t({t1},{t2},\\fscx100\\fscy100)}}")
        lines.append(f"Dialogue: 0,{_ass_time(start)},{_ass_time(end)},"
                     f"Word,,0,0,0,,{tags}{text}")

    with open(out_path, "w", encoding="utf-8-sig") as f:
        f.write(header + "\n".join(lines) + "\n")
    return out_path
