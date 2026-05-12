from __future__ import annotations

import math
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from . import config


class EncoderError(RuntimeError):
    def __init__(self, ffmpeg_stderr: str):
        super().__init__(ffmpeg_stderr)
        self.ffmpeg_stderr = ffmpeg_stderr


@dataclass(frozen=True)
class CropSegment:
    start: float
    end: float
    crop_x: int


def _run_ffmpeg(args: list[str]) -> None:
    result = subprocess.run(args, capture_output=True, text=True)
    if result.returncode != 0:
        raise EncoderError(result.stderr)


def quantize_crop_trajectory(crops: Sequence[tuple[int, int, int, int]], timestamps: Sequence[float]) -> list[CropSegment]:
    if len(crops) == 0 or len(timestamps) == 0:
        return []
    segments: list[CropSegment] = []
    start = float(timestamps[0])
    current_x = crops[0][0]
    for idx in range(1, len(crops)):
        crop_x = crops[idx][0]
        if abs(crop_x - current_x) >= config.CROP_QUANTIZE_THRESHOLD:
            segments.append(CropSegment(start, float(timestamps[idx]), current_x))
            start = float(timestamps[idx])
            current_x = crop_x
    segments.append(CropSegment(start, float(timestamps[-1]), current_x))
    return segments


def _ffmpeg_expr(value: float) -> str:
    if math.isclose(value, round(value), abs_tol=1e-6):
        return str(int(round(value)))
    return f"{value:.4f}".rstrip("0").rstrip(".")


def _escape_filter_expr(expr: str) -> str:
    return expr.replace(",", r"\,")


def _linear_crop_expression(
    crops: Sequence[tuple[int, int, int, int]],
    timestamps: Sequence[float],
    clip_start: float,
    clip_end: float,
) -> str:
    duration = max(0.0, clip_end - clip_start)
    if len(crops) == 0 or len(timestamps) == 0:
        return "555"

    points: list[tuple[float, int]] = []
    last_x: int | None = None
    for crop, timestamp in zip(crops, timestamps):
        t = max(0.0, min(duration, float(timestamp) - clip_start))
        x = int(crop[0])
        if last_x is not None and x == last_x and points and t - points[-1][0] < 1.0:
            continue
        points.append((t, x))
        last_x = x

    if not points:
        return "555"
    if points[0][0] > 0.001:
        points.insert(0, (0.0, points[0][1]))
    if points[-1][0] < duration:
        points.append((duration, points[-1][1]))

    expr = _ffmpeg_expr(points[-1][1])
    for idx in range(len(points) - 2, -1, -1):
        t0, x0 = points[idx]
        t1, x1 = points[idx + 1]
        if math.isclose(t1, t0, abs_tol=1e-6):
            segment_expr = _ffmpeg_expr(x1)
        else:
            slope = (x1 - x0) / (t1 - t0)
            segment_expr = f"{_ffmpeg_expr(x0)}+({_ffmpeg_expr(slope)})*(t-{_ffmpeg_expr(t0)})"
        expr = f"if(lte(t,{_ffmpeg_expr(t1)}),{segment_expr},{expr})"
    return _escape_filter_expr(expr)


class VideoEncoder:
    def encode(
        self,
        job_id: str,
        source: Path,
        clip_start: float,
        clip_end: float,
        crops: Sequence[tuple[int, int, int, int]],
        crop_timestamps: Sequence[float],
    ) -> Path:
        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg is None:
            raise EncoderError("ffmpeg executable not found on PATH")
        temp_dir = config.TEMP_DIR / job_id
        output_dir = config.OUTPUT_DIR
        temp_dir.mkdir(parents=True, exist_ok=True)
        output_dir.mkdir(parents=True, exist_ok=True)
        final_output = output_dir / f"{job_id}_final.mp4"

        try:
            clip_duration = max(0.0, clip_end - clip_start)
            crop_x = _linear_crop_expression(crops, crop_timestamps, clip_start, clip_end)
            vf = f"crop={config.CROP_W}:{config.CROP_H}:x={crop_x}:y=0,scale={config.OUTPUT_WIDTH}:{config.OUTPUT_HEIGHT}:flags=lanczos"
            _run_ffmpeg(
                [
                    ffmpeg,
                    "-y",
                    "-ss",
                    str(clip_start),
                    "-t",
                    str(clip_duration),
                    "-i",
                    str(source),
                    "-vf",
                    vf,
                    "-r",
                    str(config.OUTPUT_FPS),
                    "-c:v",
                    "libx264",
                    "-crf",
                    str(config.FFMPEG_CRF),
                    "-preset",
                    config.FFMPEG_PRESET,
                    "-c:a",
                    "aac",
                    "-b:a",
                    "192k",
                    str(final_output),
                ]
            )
        except Exception:
            raise
        else:
            shutil.rmtree(temp_dir, ignore_errors=True)
        return final_output
