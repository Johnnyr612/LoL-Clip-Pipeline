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


def _audio_stream_count(path: Path) -> int:
    ffprobe = shutil.which("ffprobe")
    if ffprobe is None:
        return 0
    result = subprocess.run(
        [
            ffprobe,
            "-v",
            "error",
            "-select_streams",
            "a",
            "-show_entries",
            "stream=index",
            "-of",
            "csv=p=0",
            str(path),
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return 0
    return len([line for line in result.stdout.splitlines() if line.strip()])


def _trim_with_audio(ffmpeg: str, source: Path, output: Path, clip_start: float, clip_end: float) -> None:
    audio_streams = _audio_stream_count(source)
    if audio_streams <= 0:
        _run_ffmpeg(
            [
                ffmpeg,
                "-y",
                "-ss",
                str(clip_start),
                "-to",
                str(clip_end),
                "-i",
                str(source),
                "-map",
                "0:v:0",
                "-c:v",
                "copy",
                str(output),
            ]
        )
        return

    if audio_streams == 1:
        _run_ffmpeg(
            [
                ffmpeg,
                "-y",
                "-ss",
                str(clip_start),
                "-to",
                str(clip_end),
                "-i",
                str(source),
                "-map",
                "0:v:0",
                "-map",
                "0:a:0",
                "-c:v",
                "copy",
                "-c:a",
                "aac",
                "-b:a",
                "192k",
                str(output),
            ]
        )
        return

    inputs = "".join(f"[0:a:{index}]" for index in range(audio_streams))
    _run_ffmpeg(
        [
            ffmpeg,
            "-y",
            "-ss",
            str(clip_start),
            "-to",
            str(clip_end),
            "-i",
            str(source),
            "-filter_complex",
            f"{inputs}amix=inputs={audio_streams}:duration=longest:dropout_transition=0[aout]",
            "-map",
            "0:v:0",
            "-map",
            "[aout]",
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            str(output),
        ]
    )


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
            _trim_with_audio(ffmpeg, source, trimmed, clip_start, clip_end)
            segments = quantize_crop_trajectory(crops, crop_timestamps) or [CropSegment(0.0, clip_end - clip_start, 555)]
            segment_paths: list[Path] = []
            for idx, segment in enumerate(segments):
                clip_duration = max(0.0, clip_end - clip_start)
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
                        "-map",
                        "0:v:0",
                        "-map",
                        "0:a:0?",
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
                        "-map",
                        "0:v:0",
                        "-map",
                        "0:a:0?",
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
