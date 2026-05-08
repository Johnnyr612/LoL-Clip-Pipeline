from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from . import config

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DialogSegment:
    text: str
    start: float
    end: float


@dataclass(frozen=True)
class TrimResult:
    clip_start: float
    clip_end: float
    fight_start: float
    fight_end: float
    fight_duration: float
    dialog_segments: list[DialogSegment]
    flags: list[str]


def merge_highlights(windows: Sequence[tuple[float, float]], gap: float = config.FIGHT_MERGE_GAP_SEC) -> list[tuple[float, float]]:
    if not windows:
        return []
    merged = [tuple(windows[0])]
    for start, end in sorted(windows[1:]):
        prev_start, prev_end = merged[-1]
        if start - prev_end < gap:
            merged[-1] = (prev_start, max(prev_end, end))
        else:
            merged.append((start, end))
    return merged


def boundaries_from_scores(scores: Sequence[float], source_duration: float) -> tuple[float, float, list[str]]:
    flags: list[str] = []
    if not scores or max(scores) < config.FIGHT_CONFIDENCE_THRESHOLD:
        center = source_duration / 2
        start = max(0.0, center - 5.0)
        end = min(source_duration, center + 5.0)
        return start, end, ["low_confidence"]

    peak_idx = int(np.argmax(np.array(scores)))
    left = peak_idx
    right = peak_idx
    while left > 0 and scores[left - 1] >= config.FIGHT_CONFIDENCE_THRESHOLD:
        left -= 1
    while right + 1 < len(scores) and scores[right + 1] >= config.FIGHT_CONFIDENCE_THRESHOLD:
        right += 1
    fight_start = float(left)
    fight_end = float(right + 8.0)
    duration = fight_end - fight_start
    if duration < config.FIGHT_MIN_DURATION:
        pad = (config.FIGHT_MIN_DURATION - duration) / 2
        fight_start -= pad
        fight_end += pad
    if fight_end - fight_start > config.FIGHT_MAX_DURATION:
        center = (fight_start + fight_end) / 2
        fight_start = center - config.FIGHT_MAX_DURATION / 2
        fight_end = center + config.FIGHT_MAX_DURATION / 2
    return max(0.0, fight_start), min(source_duration, fight_end), flags


def apply_dialog_extension(
    fight_start: float,
    fight_end: float,
    source_duration: float,
    dialog_segments: Sequence[DialogSegment],
) -> TrimResult:
    flags: list[str] = []
    if not dialog_segments:
        flags.append("no_dialog_detected")
    pre = [
        segment
        for segment in dialog_segments
        if segment.end <= fight_start and fight_start - segment.end <= config.DIALOG_EXTENSION_WINDOW
    ]
    post = [
        segment
        for segment in dialog_segments
        if segment.start >= fight_end and segment.start - fight_end <= config.DIALOG_EXTENSION_WINDOW
    ]
    clip_start = fight_start
    clip_end = fight_end
    if pre:
        clip_start = min(clip_start, min(s.start for s in pre) - config.DIALOG_PADDING)
    if post:
        clip_end = max(clip_end, max(s.end for s in post) + config.DIALOG_PADDING)
    clip_start = max(0.0, clip_start)
    clip_end = min(source_duration, clip_end)

    if clip_end - clip_start > config.MAX_CLIP_DURATION:
        fight_center = (fight_start + fight_end) / 2
        clip_start = max(0.0, fight_center - config.MAX_CLIP_DURATION / 2)
        clip_end = min(source_duration, clip_start + config.MAX_CLIP_DURATION)
        clip_start = max(0.0, clip_end - config.MAX_CLIP_DURATION)

    return TrimResult(
        clip_start=round(clip_start, 3),
        clip_end=round(clip_end, 3),
        fight_start=round(fight_start, 3),
        fight_end=round(fight_end, 3),
        fight_duration=round(fight_end - fight_start, 3),
        dialog_segments=list(dialog_segments),
        flags=flags,
    )


class FightDetector:
    def __init__(self) -> None:
        self.videomae_loaded = False
        self.whisper_loaded = False
        if not config.VIDEOMAE_CHECKPOINT.exists():
            logger.warning("VideoMAE checkpoint missing; pretrained/fallback scoring will be used")

    def score_windows(self, full_frames: np.ndarray, timestamps: np.ndarray) -> list[float]:
        if len(full_frames) == 0:
            return []
        # Lightweight fallback: red/intense motion heuristic for synthetic and dev runs.
        scores: list[float] = []
        max_second = int(float(timestamps[-1])) if len(timestamps) else max(0, len(full_frames) - 1)
        for second in range(max(0, max_second - 7)):
            frame_indices = np.where((timestamps >= second) & (timestamps < second + 8))[0]
            if len(frame_indices) == 0:
                scores.append(0.0)
                continue
            sample = full_frames[frame_indices]
            red_dominance = (sample[:, :, :, 0].astype(np.float32) - sample[:, :, :, 1].astype(np.float32)).mean()
            scores.append(float(np.clip((red_dominance + 20) / 80, 0, 1)))
        return scores

    def transcribe(self, audio_path: Path | None) -> list[DialogSegment]:
        if audio_path is None or not audio_path.exists():
            return []
        try:
            import whisper

            model = whisper.load_model("base", device="cpu")
            result = model.transcribe(str(audio_path))
            return [
                DialogSegment(text=str(s.get("text", "")).strip(), start=float(s["start"]), end=float(s["end"]))
                for s in result.get("segments", [])
                if str(s.get("text", "")).strip()
            ]
        except Exception as exc:  # noqa: BLE001 - Whisper absence should not kill the pipeline.
            logger.warning("Whisper transcription unavailable: %s", exc)
            return []

    def detect(self, full_frames: np.ndarray, timestamps: np.ndarray, source_duration: float, audio_path: Path | None) -> TrimResult:
        fight_start, fight_end, flags = boundaries_from_scores(self.score_windows(full_frames, timestamps), source_duration)
        dialog = self.transcribe(audio_path)
        result = apply_dialog_extension(fight_start, fight_end, source_duration, dialog)
        return TrimResult(
            result.clip_start,
            result.clip_end,
            result.fight_start,
            result.fight_end,
            result.fight_duration,
            result.dialog_segments,
            flags + result.flags,
        )
