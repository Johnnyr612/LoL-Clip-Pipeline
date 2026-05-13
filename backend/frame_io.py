from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import config


@dataclass(frozen=True)
class FrameBundle:
    full_frames: np.ndarray
    minimap_frames: np.ndarray
    timestamps_full: np.ndarray
    timestamps_mini: np.ndarray
    audio_path: Path | None


class FrameDecodeError(RuntimeError):
    pass


def extract_audio(input_path: Path, job_id: str) -> Path | None:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        return None
    temp_dir = config.TEMP_DIR / job_id
    temp_dir.mkdir(parents=True, exist_ok=True)
    wav_path = temp_dir / "audio.wav"
    result = subprocess.run(
        [ffmpeg, "-y", "-i", str(input_path), "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1", str(wav_path)],
        capture_output=True,
        text=True,
    )
    return wav_path if result.returncode == 0 else None


def decode_video(input_path: Path, job_id: str) -> FrameBundle:
    return _opencv_decode(input_path, job_id)


def _opencv_decode(input_path: Path, job_id: str) -> FrameBundle:
    try:
        import cv2
    except Exception as exc:  # noqa: BLE001
        raise FrameDecodeError(f"OpenCV video decoding unavailable: {exc}") from exc

    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise FrameDecodeError(f"could not open video: {input_path}")
    source_fps = cap.get(cv2.CAP_PROP_FPS) or 120.0
    full_step = max(1, round(source_fps / 2.0))
    mini_step = max(1, round(source_fps / 4.0))
    full_frames: list[np.ndarray] = []
    mini_frames: list[np.ndarray] = []
    ts_full: list[float] = []
    ts_mini: list[float] = []
    idx = 0
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        if idx % full_step == 0:
            rgb = cv2.cvtColor(cv2.resize(bgr, (1920, 1080)), cv2.COLOR_BGR2RGB)
            full_frames.append(rgb)
            ts_full.append(idx / source_fps)
        if idx % mini_step == 0:
            h, w = bgr.shape[:2]
            crop = bgr[int(h * config.MINIMAP_CROP_Y_PCT) : h, int(w * config.MINIMAP_CROP_X_PCT) : w]
            mini_frames.append(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
            ts_mini.append(idx / source_fps)
        idx += 1
    cap.release()
    audio_path = extract_audio(input_path, job_id)
    return FrameBundle(
        np.asarray(full_frames, dtype=np.uint8),
        np.asarray(mini_frames, dtype=np.uint8),
        np.asarray(ts_full, dtype=np.float32),
        np.asarray(ts_mini, dtype=np.float32),
        audio_path,
    )
