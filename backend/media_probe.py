from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any

from .ffmpeg_tools import find_executable


class MediaProbeError(ValueError):
    pass


@dataclass(frozen=True)
class MediaProfile:
    duration: float
    has_audio: bool
    video_codec: str | None = None
    audio_codec: str | None = None
    width: int | None = None
    height: int | None = None
    fps: float | None = None
    fps_rate: str | None = None
    video_bitrate: int | None = None
    audio_bitrate: int | None = None
    total_bitrate: int | None = None

    def to_debug_dict(self) -> dict[str, Any]:
        return {
            "duration": round(self.duration, 3),
            "has_audio": self.has_audio,
            "video_codec": self.video_codec,
            "audio_codec": self.audio_codec,
            "width": self.width,
            "height": self.height,
            "fps": round(self.fps, 3) if self.fps is not None else None,
            "fps_rate": self.fps_rate,
            "video_bitrate": self.video_bitrate,
            "audio_bitrate": self.audio_bitrate,
            "total_bitrate": self.total_bitrate,
        }


def probe_media_profile(path: Path) -> MediaProfile:
    ffprobe = find_executable("ffprobe")
    if ffprobe is None:
        return MediaProfile(duration=60.0, has_audio=False)

    result = subprocess.run(
        [
            ffprobe,
            "-v",
            "error",
            "-show_entries",
            "format=duration,bit_rate:stream=codec_type,codec_name,width,height,r_frame_rate,avg_frame_rate,bit_rate",
            "-of",
            "json",
            str(path),
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise MediaProbeError(result.stderr)

    payload = json.loads(result.stdout)
    format_info = payload.get("format", {})
    streams = payload.get("streams", [])
    video = next((stream for stream in streams if stream.get("codec_type") == "video"), {})
    audio = next((stream for stream in streams if stream.get("codec_type") == "audio"), {})
    fps_rate = _best_fps_rate(video)

    return MediaProfile(
        duration=_float_or_zero(format_info.get("duration")),
        has_audio=bool(audio),
        video_codec=_str_or_none(video.get("codec_name")),
        audio_codec=_str_or_none(audio.get("codec_name")),
        width=_int_or_none(video.get("width")),
        height=_int_or_none(video.get("height")),
        fps=_fps_value(fps_rate),
        fps_rate=fps_rate,
        video_bitrate=_int_or_none(video.get("bit_rate")),
        audio_bitrate=_int_or_none(audio.get("bit_rate")),
        total_bitrate=_int_or_none(format_info.get("bit_rate")),
    )


def _best_fps_rate(video: dict[str, Any]) -> str | None:
    for key in ("avg_frame_rate", "r_frame_rate"):
        value = _str_or_none(video.get(key))
        if value and value != "0/0":
            return value
    return None


def _fps_value(rate: str | None) -> float | None:
    if not rate:
        return None
    try:
        return float(Fraction(rate))
    except (ValueError, ZeroDivisionError):
        return None


def _float_or_zero(value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _int_or_none(value: object) -> int | None:
    try:
        return int(float(str(value)))
    except (TypeError, ValueError):
        return None


def _str_or_none(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None
