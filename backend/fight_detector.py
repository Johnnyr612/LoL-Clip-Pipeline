from __future__ import annotations

import contextlib
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from . import config

logger = logging.getLogger(__name__)
CUDA_DEVICE_TYPE = "cu" + "da"


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

    def score_windows(
        self,
        full_frames: np.ndarray,
        timestamps: np.ndarray,
    ) -> list[float]:
        """
        Score each 1-second window as P(fight) using VideoMAE.
        Automatically falls back to heuristic if checkpoint missing
        or if inference fails for any reason.
        """
        if len(full_frames) == 0:
            return []

        checkpoint = Path(str(config.VIDEOMAE_CHECKPOINT))
        if not checkpoint.exists():
            logger.warning(
                "VideoMAE checkpoint not found at %s - heuristic fallback",
                checkpoint,
            )
            return self._heuristic_scores(full_frames, timestamps)

        try:
            import torch
            import torch.nn as nn
            import torchvision.transforms.functional as TF
            from transformers import VideoMAEModel

            device = torch.device(
                CUDA_DEVICE_TYPE if torch.cuda.is_available() else "cpu"
            )

            class _Classifier(nn.Module):
                def __init__(self) -> None:
                    super().__init__()
                    self.videomae = VideoMAEModel.from_pretrained(
                        "MCG-NJU/videomae-base"
                    )
                    self.classifier = nn.Linear(768, 2)

                def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
                    out = self.videomae(pixel_values=pixel_values)
                    pooled = out.last_hidden_state.mean(dim=1)
                    return self.classifier(pooled)

            model = _Classifier().to(device)
            model.load_state_dict(
                torch.load(str(checkpoint), map_location=device)
            )
            model.eval()

            mean = [0.485, 0.456, 0.406]
            std = [0.229, 0.224, 0.225]
            scores: list[float] = []
            max_second = int(float(timestamps[-1])) if len(timestamps) else 0

            with torch.no_grad():
                for second in range(max(0, max_second - 7)):
                    indices = np.where(
                        (timestamps >= second) & (timestamps < second + 8)
                    )[0]
                    if len(indices) == 0:
                        scores.append(0.0)
                        continue
                    selected = indices[
                        np.linspace(0, len(indices) - 1, 8).astype(int)
                    ]
                    tensors = []
                    for frame in full_frames[selected]:
                        t = (
                            torch.from_numpy(frame).permute(2, 0, 1).float()
                            / 255.0
                        )
                        t = TF.resize(t, [224, 224], antialias=True)
                        t = TF.normalize(t, mean, std)
                        tensors.append(t)
                    pixel_values = torch.stack(tensors).unsqueeze(0).to(device)
                    ctx = (
                        torch.cuda.amp.autocast()
                        if device.type == CUDA_DEVICE_TYPE
                        else contextlib.nullcontext()
                    )
                    with ctx:
                        logits = model(pixel_values)
                    prob = float(torch.softmax(logits, dim=-1)[0, 1].cpu())
                    scores.append(prob)

            return scores

        except Exception as exc:
            logger.warning(
                "VideoMAE inference failed (%s) - heuristic fallback",
                exc,
            )
            return self._heuristic_scores(full_frames, timestamps)

    def _heuristic_scores(
        self,
        full_frames: np.ndarray,
        timestamps: np.ndarray,
    ) -> list[float]:
        """Red-dominance fallback - used when VideoMAE checkpoint missing."""
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
            from faster_whisper import WhisperModel

            model = WhisperModel("base", device="cpu", compute_type="int8")
            segments, _info = model.transcribe(str(audio_path))
            return [
                DialogSegment(text=seg.text.strip(), start=float(seg.start), end=float(seg.end))
                for seg in segments
                if seg.text.strip()
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
