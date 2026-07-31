from __future__ import annotations

import math
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from . import config
from .media_probe import MediaProfile


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


def _bitrate_label(bits_per_second: int | None) -> str:
    if bits_per_second is None or bits_per_second <= 0:
        return ""
    if bits_per_second % 1_000_000 == 0:
        return f"{bits_per_second // 1_000_000}M"
    if bits_per_second >= 1_000_000:
        return f"{bits_per_second / 1_000_000:.1f}M"
    return str(bits_per_second)


def _bitrate_value(label: str) -> int | None:
    text = label.strip().lower()
    if not text:
        return None
    multiplier = 1
    if text.endswith("m"):
        multiplier = 1_000_000
        text = text[:-1]
    elif text.endswith("k"):
        multiplier = 1_000
        text = text[:-1]
    try:
        return int(float(text) * multiplier)
    except ValueError:
        return None


def _rate_at_least(configured: str, selected_bitrate: str) -> str:
    if not configured:
        return selected_bitrate
    configured_value = _bitrate_value(configured)
    selected_value = _bitrate_value(selected_bitrate)
    if configured_value is not None and selected_value is not None and configured_value < selected_value:
        return selected_bitrate
    return configured


def _matched_source_bitrate(source_profile: MediaProfile | None = None) -> int | None:
    if not config.FFMPEG_MATCH_SOURCE_ENCODING or source_profile is None:
        return None
    source_bitrate = source_profile.video_bitrate or source_profile.total_bitrate
    if source_bitrate is None or source_bitrate <= 0:
        return None
    multiplier = max(1.0, float(config.FFMPEG_SOURCE_BITRATE_MULTIPLIER))
    return int(round(source_bitrate * multiplier))


def _selected_video_bitrate(source_profile: MediaProfile | None = None) -> str:
    matched_bitrate = _matched_source_bitrate(source_profile)
    if matched_bitrate is not None:
        label = _bitrate_label(matched_bitrate)
        if label:
            return label
    return config.FFMPEG_VIDEO_BITRATE


def _selected_output_fps(source_profile: MediaProfile | None = None) -> str:
    if not config.FFMPEG_MATCH_SOURCE_ENCODING or source_profile is None or source_profile.fps is None:
        return str(config.OUTPUT_FPS)
    return f"{source_profile.fps:.3f}".rstrip("0").rstrip(".")


def _audio_encode_args(source_profile: MediaProfile | None = None) -> list[str]:
    if (
        config.FFMPEG_MATCH_SOURCE_ENCODING
        and source_profile is not None
        and (source_profile.audio_codec or "").lower() == "aac"
    ):
        return ["-c:a", "copy"]
    return ["-c:a", "aac", "-b:a", config.FFMPEG_AUDIO_BITRATE]


def describe_encode_settings(source_profile: MediaProfile | None = None) -> dict:
    bitrate = _selected_video_bitrate(source_profile)
    fps = _selected_output_fps(source_profile)
    return {
        "match_source_encoding": config.FFMPEG_MATCH_SOURCE_ENCODING,
        "encoder": config.FFMPEG_VIDEO_ENCODER.lower(),
        "fps": fps,
        "target_video_bitrate": bitrate or None,
        "maxrate": _rate_at_least(config.FFMPEG_VIDEO_MAXRATE, bitrate) if bitrate else None,
        "bufsize": _rate_at_least(config.FFMPEG_VIDEO_BUFSIZE, bitrate) if bitrate else None,
        "source_bitrate_multiplier": config.FFMPEG_SOURCE_BITRATE_MULTIPLIER if config.FFMPEG_MATCH_SOURCE_ENCODING else None,
        "crf": None if bitrate or config.FFMPEG_VIDEO_ENCODER.lower() == "h264_nvenc" else config.FFMPEG_CRF,
        "preset": config.FFMPEG_NVENC_PRESET if config.FFMPEG_VIDEO_ENCODER.lower() == "h264_nvenc" else config.FFMPEG_PRESET,
        "rate_control": config.FFMPEG_NVENC_RC if config.FFMPEG_VIDEO_ENCODER.lower() == "h264_nvenc" else None,
        "audio_codec": "copy" if _audio_encode_args(source_profile) == ["-c:a", "copy"] else "aac",
        "audio_bitrate": None if _audio_encode_args(source_profile) == ["-c:a", "copy"] else config.FFMPEG_AUDIO_BITRATE,
        "pixel_format": "yuv420p",
        "output_width": config.OUTPUT_WIDTH,
        "output_height": config.OUTPUT_HEIGHT,
    }


def _video_encode_args(source_profile: MediaProfile | None = None) -> list[str]:
    encoder = config.FFMPEG_VIDEO_ENCODER.lower()
    args = ["-c:v", encoder]
    if encoder == "h264_nvenc":
        args.extend(["-preset", config.FFMPEG_NVENC_PRESET, "-rc", config.FFMPEG_NVENC_RC])
        if config.FFMPEG_NVENC_CQ:
            args.extend(["-cq", config.FFMPEG_NVENC_CQ])
    else:
        args.extend(["-preset", config.FFMPEG_PRESET])

    bitrate = _selected_video_bitrate(source_profile)
    if bitrate:
        args.extend(["-b:v", bitrate])
        args.extend(["-maxrate", _rate_at_least(config.FFMPEG_VIDEO_MAXRATE, bitrate)])
        args.extend(["-bufsize", _rate_at_least(config.FFMPEG_VIDEO_BUFSIZE, bitrate)])
    elif encoder != "h264_nvenc":
        args.extend(["-crf", str(config.FFMPEG_CRF)])
    args.extend(["-pix_fmt", "yuv420p"])
    return args


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


def _simplify_points(points: list[tuple[float, float]], tolerance: float = 2.0) -> list[tuple[float, float]]:
    """Drop points that lie on (or near) the straight line between their
    neighbors, so the filter expression stays short without changing the pan."""
    if len(points) <= 2:
        return list(points)
    keep = [points[0]]
    for idx in range(1, len(points) - 1):
        t0, x0 = keep[-1]
        t1, x1 = points[idx]
        t2, x2 = points[idx + 1]
        span = max(t2 - t0, 1e-6)
        predicted = x0 + (x2 - x0) * (t1 - t0) / span
        if abs(predicted - x1) > tolerance:
            keep.append(points[idx])
    keep.append(points[-1])
    return keep


def _step_crop_expression(
    crops: Sequence[tuple[int, int, int, int]],
    timestamps: Sequence[float],
    clip_start: float,
    clip_end: float,
) -> str:
    """Piecewise-constant x(t): the crop holds each position and cuts
    instantly to the next, with no sliding."""
    duration = max(0.0, clip_end - clip_start)
    if len(crops) == 0 or len(timestamps) == 0:
        return "555"

    segments: list[tuple[float, float]] = []  # (start_time, x)
    for crop, timestamp in zip(crops, timestamps):
        t = max(0.0, min(duration, float(timestamp) - clip_start))
        x = float(crop[0])
        if not segments or abs(x - segments[-1][1]) >= 1.0:
            segments.append((t, x))
    if segments[0][0] > 0.001:
        segments[0] = (0.0, segments[0][1])

    max_points = max(2, config.CROP_EXPR_MAX_POINTS)
    if len(segments) > max_points:
        indices = sorted({round(i * (len(segments) - 1) / (max_points - 1)) for i in range(max_points)})
        segments = [segments[i] for i in indices]

    expr = _ffmpeg_expr(segments[-1][1])
    for idx in range(len(segments) - 2, -1, -1):
        boundary = segments[idx + 1][0]
        expr = f"if(lt(t,{_ffmpeg_expr(boundary)}),{_ffmpeg_expr(segments[idx][1])},{expr})"
    return _escape_filter_expr(expr)


def _crop_x_expression(
    crops: Sequence[tuple[int, int, int, int]],
    timestamps: Sequence[float],
    clip_start: float,
    clip_end: float,
) -> str:
    if config.CROP_TRANSITION == "cut":
        return _step_crop_expression(crops, timestamps, clip_start, clip_end)
    return _linear_crop_expression(crops, timestamps, clip_start, clip_end)


def _linear_crop_expression(
    crops: Sequence[tuple[int, int, int, int]],
    timestamps: Sequence[float],
    clip_start: float,
    clip_end: float,
) -> str:
    """Build a piecewise-linear x(t) expression so the crop pans smoothly
    between keyframes instead of jumping in steps."""
    duration = max(0.0, clip_end - clip_start)
    if len(crops) == 0 or len(timestamps) == 0:
        return "555"

    points: list[tuple[float, float]] = []
    for crop, timestamp in zip(crops, timestamps):
        t = max(0.0, min(duration, float(timestamp) - clip_start))
        x = float(crop[0])
        if points and abs(t - points[-1][0]) < 1e-4:
            points[-1] = (t, x)
        else:
            points.append((t, x))

    if not points:
        return "555"
    if points[0][0] > 0.001:
        points.insert(0, (0.0, points[0][1]))
    if points[-1][0] < duration:
        points.append((duration, points[-1][1]))

    points = _simplify_points(points)
    if all(abs(x - points[0][1]) < 0.5 for _, x in points):
        return _ffmpeg_expr(points[0][1])
    max_points = max(2, config.CROP_EXPR_MAX_POINTS)
    if len(points) > max_points:
        indices = sorted({round(i * (len(points) - 1) / (max_points - 1)) for i in range(max_points)})
        points = [points[i] for i in indices]

    expr = _ffmpeg_expr(points[-1][1])
    for idx in range(len(points) - 2, -1, -1):
        t0, x0 = points[idx]
        t1, x1 = points[idx + 1]
        span = max(t1 - t0, 1e-3)
        if abs(x1 - x0) < 0.5:
            segment = _ffmpeg_expr(x0)
        else:
            segment = (
                f"({_ffmpeg_expr(x0)}+({_ffmpeg_expr(x1 - x0)})"
                f"*(t-{_ffmpeg_expr(t0)})/{_ffmpeg_expr(span)})"
            )
        expr = f"if(lt(t,{_ffmpeg_expr(t1)}),{segment},{expr})"
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
        source_profile: MediaProfile | None = None,
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
            crop_x = _crop_x_expression(crops, crop_timestamps, clip_start, clip_end)
            output_fps = _selected_output_fps(source_profile)
            gop_size = str(max(1, int(round(float(output_fps)))))
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
                    output_fps,
                    *_video_encode_args(source_profile),
                    "-g",
                    gop_size,
                    "-keyint_min",
                    gop_size,
                    "-sc_threshold",
                    "0",
                    *_audio_encode_args(source_profile),
                    "-movflags",
                    "+faststart",
                    str(final_output),
                ]
            )
        except Exception:
            raise
        else:
            shutil.rmtree(temp_dir, ignore_errors=True)
        return final_output
