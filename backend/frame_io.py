from __future__ import annotations

import ctypes
import json
import shutil
import subprocess
import sys
import tempfile
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


def _decoder_library_path() -> Path:
    suffix = ".dll" if sys.platform.startswith("win") else ".so"
    return config.PROJECT_ROOT / "frame_decoder" / f"frame_decoder{suffix}"


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
    lib_path = _decoder_library_path()
    if not lib_path.exists():
        return _opencv_fallback_decode(input_path, job_id)

    # The native decoder is invoked in a subprocess so a C++ crash cannot kill FastAPI.
    script = Path(tempfile.gettempdir()) / f"lol_decode_{job_id}.py"
    script.write_text(
        "from backend.frame_io import _decode_with_ctypes; import json, sys; print(json.dumps(_decode_with_ctypes(sys.argv[1], sys.argv[2])))",
        encoding="utf-8",
    )
    out_dir = config.TEMP_DIR / job_id / "frames"
    out_dir.mkdir(parents=True, exist_ok=True)
    result = subprocess.run([sys.executable, str(script), str(input_path), str(out_dir)], cwd=config.PROJECT_ROOT, capture_output=True, text=True)
    if result.returncode != 0:
        raise FrameDecodeError(result.stderr)
    payload = json.loads(result.stdout)
    audio_path = extract_audio(input_path, job_id)
    return FrameBundle(
        np.load(payload["full"]),
        np.load(payload["mini"]),
        np.load(payload["timestamps_full"]),
        np.load(payload["timestamps_mini"]),
        audio_path,
    )


def _decode_with_ctypes(input_path: str, out_dir: str) -> dict[str, str]:
    class DecodedFrames(ctypes.Structure):
        _fields_ = [
            ("data", ctypes.POINTER(ctypes.c_uint8)),
            ("n_frames", ctypes.c_int),
            ("height", ctypes.c_int),
            ("width", ctypes.c_int),
            ("channels", ctypes.c_int),
        ]

    lib = ctypes.CDLL(str(_decoder_library_path()))
    lib.decode_full_frames.argtypes = [ctypes.c_char_p, ctypes.c_float, ctypes.c_int, ctypes.c_int]
    lib.decode_full_frames.restype = DecodedFrames
    lib.decode_minimap_frames.argtypes = [ctypes.c_char_p, ctypes.c_float, ctypes.c_float, ctypes.c_float]
    lib.decode_minimap_frames.restype = DecodedFrames
    lib.free_frame_buffer.argtypes = [ctypes.POINTER(ctypes.c_uint8)]
    lib.free_frame_buffer.restype = None

    out = Path(out_dir)
    full = lib.decode_full_frames(str(input_path).encode("utf-8"), ctypes.c_float(2.0), 1920, 1080)
    mini = lib.decode_minimap_frames(
        str(input_path).encode("utf-8"),
        ctypes.c_float(4.0),
        ctypes.c_float(config.MINIMAP_CROP_X_PCT),
        ctypes.c_float(config.MINIMAP_CROP_Y_PCT),
    )
    try:
        full_path = out / "full.npy"
        mini_path = out / "mini.npy"
        ts_full_path = out / "timestamps_full.npy"
        ts_mini_path = out / "timestamps_mini.npy"
        np.save(full_path, _result_to_array(full))
        np.save(mini_path, _result_to_array(mini))
        np.save(ts_full_path, np.arange(full.n_frames, dtype=np.float32) / 2.0)
        np.save(ts_mini_path, np.arange(mini.n_frames, dtype=np.float32) / 4.0)
    finally:
        if full.data:
            lib.free_frame_buffer(full.data)
        if mini.data:
            lib.free_frame_buffer(mini.data)
    return {
        "full": str(full_path),
        "mini": str(mini_path),
        "timestamps_full": str(ts_full_path),
        "timestamps_mini": str(ts_mini_path),
    }


def _result_to_array(result: ctypes.Structure) -> np.ndarray:
    if not result.data or result.n_frames <= 0:
        return np.empty((0, result.height, result.width, result.channels), dtype=np.uint8)
    size = result.n_frames * result.height * result.width * result.channels
    flat = np.ctypeslib.as_array(result.data, shape=(size,))
    return flat.reshape((result.n_frames, result.height, result.width, result.channels)).copy()


def _opencv_fallback_decode(input_path: Path, job_id: str) -> FrameBundle:
    try:
        import cv2
    except Exception as exc:  # noqa: BLE001
        raise FrameDecodeError(f"frame_decoder binary missing and OpenCV fallback unavailable: {exc}") from exc

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
