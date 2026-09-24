from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import config
from .ffmpeg_tools import find_executable


@dataclass(frozen=True)
class FrameBundle:
    full_frames: np.ndarray
    minimap_frames: np.ndarray
    timestamps_full: np.ndarray
    timestamps_mini: np.ndarray
    audio_path: Path | None
    vjepa_frames: object | None = None


class FrameDecodeError(RuntimeError):
    pass


def extract_audio(input_path: Path, job_id: str) -> Path | None:
    ffmpeg = find_executable("ffmpeg")
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


def decode_video(input_path: Path, job_id: str, *, vjepa_config=None, duration=None,
                 include_minimap=True, include_audio=True) -> FrameBundle:
    return _opencv_decode(input_path, job_id, vjepa_config=vjepa_config, duration=duration,
                          include_minimap=include_minimap, include_audio=include_audio)


def _opencv_decode(input_path: Path, job_id: str, *, vjepa_config=None, duration=None,
                   include_minimap=True, include_audio=True) -> FrameBundle:
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
    shared = None
    needed = set()
    try:
        if vjepa_config is not None:
            from .vjepa_runtime import DecodedWindows, window_frame_indexes
            count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            indexes = window_frame_indexes(duration, source_fps, count, vjepa_config)
            needed = set(indexes.flatten().tolist())
            shared = DecodedWindows({}, indexes, vjepa_config, duration)
        idx = 0
        # Advance the decoder once. Retrieve/resize only frames needed by a consumer.
        while cap.grab():
            take_full = idx % full_step == 0
            take_mini = include_minimap and idx % mini_step == 0
            take_vjepa = idx in needed
            if take_full or take_mini or take_vjepa:
                ok, bgr = cap.retrieve()
                if not ok:
                    raise FrameDecodeError(f"could not retrieve frame {idx}: {input_path}")
                if take_full:
                    rgb = cv2.cvtColor(cv2.resize(bgr, (1920, 1080)), cv2.COLOR_BGR2RGB)
                    full_frames.append(rgb)
                    ts_full.append(idx / source_fps)
                if take_mini:
                    h, w = bgr.shape[:2]
                    crop = bgr[int(h * config.MINIMAP_CROP_Y_PCT):h, int(w * config.MINIMAP_CROP_X_PCT):w]
                    mini_frames.append(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
                    ts_mini.append(idx / source_fps)
                if take_vjepa:
                    # Resize the original frame, NOT the 1080p analysis copy.
                    frame = cv2.resize(bgr, (vjepa_config.image_size, vjepa_config.image_size), interpolation=cv2.INTER_LINEAR)
                    shared.frames[idx] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            idx += 1
        if shared is not None and needed != set(shared.frames):
            raise FrameDecodeError("Video ended before all required V-JEPA frames were decoded")
    finally:
        cap.release()
    audio_path = extract_audio(input_path, job_id) if include_audio else None
    return FrameBundle(
        np.asarray(full_frames, dtype=np.uint8),
        np.asarray(mini_frames, dtype=np.uint8),
        np.asarray(ts_full, dtype=np.float32),
        np.asarray(ts_mini, dtype=np.float32),
        audio_path,
        shared,
    )
