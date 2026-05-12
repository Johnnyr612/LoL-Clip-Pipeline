from __future__ import annotations

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
        trimmed = temp_dir / "trimmed.mp4"
        final_output = output_dir / f"{job_id}_final.mp4"

        try:
            _run_ffmpeg([ffmpeg, "-y", "-ss", str(clip_start), "-to", str(clip_end), "-i", str(source), "-c:v", "copy", "-c:a", "copy", str(trimmed)])
            segments = quantize_crop_trajectory(crops, crop_timestamps) or [CropSegment(0.0, clip_end - clip_start, 555)]
            segment_paths: list[Path] = []
            clip_duration = max(0.0, clip_end - clip_start)
            for idx, segment in enumerate(segments):
                segment_start = max(0.0, min(clip_duration, segment.start - clip_start))
                segment_end = max(0.0, min(clip_duration, segment.end - clip_start))
                duration = segment_end - segment_start
                if duration <= 0.05:
                    continue
                segment_path = temp_dir / f"segment_{idx:04d}.mp4"
                vf = f"crop={config.CROP_W}:{config.CROP_H}:{segment.crop_x}:0,scale={config.OUTPUT_WIDTH}:{config.OUTPUT_HEIGHT}:flags=lanczos"
                _run_ffmpeg(
                    [
                        ffmpeg,
                        "-y",
                        "-ss",
                        str(segment_start),
                        "-t",
                        str(duration),
                        "-i",
                        str(trimmed),
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
                        str(segment_path),
                    ]
                )
                segment_paths.append(segment_path)
            if not segment_paths:
                segment_path = temp_dir / "segment_fallback.mp4"
                vf = f"crop={config.CROP_W}:{config.CROP_H}:555:0,scale={config.OUTPUT_WIDTH}:{config.OUTPUT_HEIGHT}:flags=lanczos"
                _run_ffmpeg(
                    [
                        ffmpeg,
                        "-y",
                        "-i",
                        str(trimmed),
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
                        str(segment_path),
                    ]
                )
                segment_paths.append(segment_path)
            concat_list = temp_dir / "concat.txt"
            concat_list.write_text("".join(f"file '{path.as_posix()}'\n" for path in segment_paths), encoding="utf-8")
            _run_ffmpeg([ffmpeg, "-y", "-f", "concat", "-safe", "0", "-i", str(concat_list), "-c", "copy", str(final_output)])
        except Exception:
            raise
        else:
            shutil.rmtree(temp_dir, ignore_errors=True)
        return final_output
