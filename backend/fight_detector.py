from __future__ import annotations

import contextlib
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import cv2
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
        fallback_duration = config.COMBAT_EVENT_TARGET_CLIP_DURATION_SEC if config.CONSERVATIVE_FULL_FIGHT_TRIM else 10.0
        start = max(0.0, center - fallback_duration / 2)
        end = min(source_duration, start + fallback_duration)
        start = max(0.0, end - fallback_duration)
        return start, end, ["low_confidence"]

    peak_idx = int(np.argmax(np.array(scores)))
    left = peak_idx
    right = peak_idx
    while left > 0 and scores[left - 1] >= config.FIGHT_CONFIDENCE_THRESHOLD:
        left -= 1
    while right + 1 < len(scores) and scores[right + 1] >= config.FIGHT_CONFIDENCE_THRESHOLD:
        right += 1
    fight_start = float(left)
    fight_end = float(right + 16.0)
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


def finish_on_kill_or_death(
    trim: TrimResult,
    full_frames: np.ndarray,
    timestamps: np.ndarray,
    source_duration: float,
) -> TrimResult:
    min_clip_end = min(source_duration, trim.clip_start + config.COMBAT_EVENT_MIN_CLIP_DURATION_SEC)
    target_clip_end = min(source_duration, trim.clip_start + config.COMBAT_EVENT_TARGET_CLIP_DURATION_SEC)
    max_clip_end = min(source_duration, trim.clip_start + config.MAX_CLIP_DURATION)
    event_time, event_flag = _detect_combat_event_time(
        full_frames,
        timestamps,
        trim.fight_start,
        max(trim.fight_end, max_clip_end - config.COMBAT_EVENT_SEARCH_AFTER_FIGHT_SEC),
        min_event_time=min_clip_end,
    )
    if event_time is None:
        clip_end = max(trim.clip_end, target_clip_end)
        clip_end = _preserve_overlapping_dialog(trim.clip_start, clip_end, trim.dialog_segments, source_duration)
        clip_end = min(clip_end, max_clip_end)
        return TrimResult(
            clip_start=trim.clip_start,
            clip_end=round(clip_end, 3),
            fight_start=trim.fight_start,
            fight_end=round(max(trim.fight_end, clip_end), 3),
            fight_duration=round(max(0.0, max(trim.fight_end, clip_end) - trim.fight_start), 3),
            dialog_segments=trim.dialog_segments,
            flags=[*trim.flags, "combat_event_not_confirmed_extended_to_target"],
        )

    clip_end = min(max_clip_end, event_time + config.COMBAT_EVENT_END_PADDING_SEC)
    flags = [*trim.flags, event_flag, "clip_end_on_kill_or_death"]
    if config.CONSERVATIVE_FULL_FIGHT_TRIM:
        clip_end = max(clip_end, target_clip_end, trim.clip_end)
        flags.append("conservative_full_fight_trim")
    clip_end = _preserve_overlapping_dialog(trim.clip_start, clip_end, trim.dialog_segments, source_duration)
    clip_end = min(max(clip_end, min_clip_end), max_clip_end)
    fight_end = max(trim.fight_end, event_time)
    return TrimResult(
        clip_start=trim.clip_start,
        clip_end=round(clip_end, 3),
        fight_start=trim.fight_start,
        fight_end=round(fight_end, 3),
        fight_duration=round(max(0.0, fight_end - trim.fight_start), 3),
        dialog_segments=trim.dialog_segments,
        flags=flags,
    )


def add_output_context(trim: TrimResult, source_duration: float) -> TrimResult:
    clip_start = max(0.0, trim.clip_start - config.OUTPUT_CONTEXT_PADDING_SEC)
    clip_end = min(source_duration, trim.clip_end + config.OUTPUT_CONTEXT_PADDING_SEC)
    if clip_end - clip_start > config.MAX_CLIP_DURATION:
        overflow = (clip_end - clip_start) - config.MAX_CLIP_DURATION
        front_room = trim.clip_start - clip_start
        back_room = clip_end - trim.clip_end
        trim_front = min(front_room, overflow / 2)
        trim_back = min(back_room, overflow - trim_front)
        remaining = overflow - trim_front - trim_back
        if remaining > 0:
            if front_room - trim_front > back_room - trim_back:
                trim_front += remaining
            else:
                trim_back += remaining
        clip_start += trim_front
        clip_end -= trim_back
    return TrimResult(
        clip_start=round(clip_start, 3),
        clip_end=round(clip_end, 3),
        fight_start=trim.fight_start,
        fight_end=trim.fight_end,
        fight_duration=trim.fight_duration,
        dialog_segments=trim.dialog_segments,
        flags=[*trim.flags, "output_context_padding_applied"],
    )


def estimate_visible_enemy_count(
    full_frames: np.ndarray,
    timestamps: np.ndarray,
    fight_start: float,
    fight_end: float,
) -> int | None:
    if len(full_frames) == 0 or len(timestamps) == 0:
        return None
    indexes = np.flatnonzero((timestamps >= fight_start) & (timestamps <= fight_end))
    if len(indexes) == 0:
        return None
    if len(indexes) > 8:
        indexes = indexes[np.linspace(0, len(indexes) - 1, 8, dtype=int)]
    counts: list[int] = []
    for index in indexes:
        red_bars, green_bars = _combat_health_bars(full_frames[int(index)])
        player_bar = _select_player_health_bar(green_bars)
        count = len(_enemy_bars_near_player(red_bars, player_bar))
        if count:
            counts.append(count)
    if not counts:
        return None
    return int(max(1, min(5, round(float(np.median(counts))))))


def estimate_combat_screen_x_positions(full_frames: np.ndarray) -> tuple[list[float | None], list[float | None]]:
    player_positions: list[float | None] = []
    threat_positions: list[float | None] = []
    for frame in full_frames:
        red_bars, green_bars = _combat_health_bars(frame)
        player_bar = _select_player_health_bar(green_bars)
        if player_bar is None:
            player_positions.append(None)
            threat_positions.append(_mean_bar_center_x(red_bars))
            continue
        x, _, width, _ = player_bar
        player_positions.append(float(x + (width - 1) / 2))
        threat_positions.append(_nearest_bar_center_x(_enemy_bars_near_player(red_bars, player_bar), player_bar))
    return player_positions, threat_positions


def _nearest_bar_center_x(
    bars: Sequence[tuple[int, int, int, int]],
    player_bar: tuple[int, int, int, int],
) -> float | None:
    if not bars:
        return None
    px, py = _bar_center(player_bar)
    nearest = min(bars, key=lambda bar: float(np.linalg.norm(np.array(_bar_center(bar)) - np.array((px, py)))))
    return float(_bar_center(nearest)[0])


def _combine_scores(primary_scores: Sequence[float], healthbar_scores: Sequence[float]) -> list[float]:
    length = max(len(primary_scores), len(healthbar_scores))
    combined: list[float] = []
    for idx in range(length):
        primary = float(primary_scores[idx]) if idx < len(primary_scores) else 0.0
        healthbar = float(healthbar_scores[idx]) if idx < len(healthbar_scores) else 0.0
        combined.append(max(primary, healthbar))
    return combined


def _healthbar_engagement_scores(full_frames: np.ndarray, timestamps: np.ndarray) -> list[float]:
    if len(full_frames) == 0:
        return []
    max_second = int(float(timestamps[-1])) if len(timestamps) else max(0, len(full_frames) - 1)
    scores: list[float] = []
    for second in range(max(0, max_second - 15)):
        frame_indices = np.where((timestamps >= second) & (timestamps < second + 16))[0]
        if len(frame_indices) == 0:
            scores.append(0.0)
            continue
        if len(frame_indices) > 12:
            frame_indices = frame_indices[np.linspace(0, len(frame_indices) - 1, 12, dtype=int)]
        engaged_scores: list[float] = []
        for index in frame_indices:
            red_bars, green_bars = _combat_health_bars(full_frames[int(index)])
            player_bar = _select_player_health_bar(green_bars)
            nearby_enemies = _enemy_bars_near_player(red_bars, player_bar)
            if player_bar is None or not nearby_enemies:
                engaged_scores.append(0.0)
                continue
            enemy_count_bonus = min(len(nearby_enemies), 3) * 0.05
            engaged_scores.append(min(0.95, 0.72 + enemy_count_bonus))
        scores.append(float(np.median(engaged_scores)) if engaged_scores else 0.0)
    return scores


def _mean_bar_center_x(bars: Sequence[tuple[int, int, int, int]]) -> float | None:
    if not bars:
        return None
    centers = [x + (width - 1) / 2 for x, _, width, _ in bars]
    return float(np.mean(centers))


def _bar_center(bar: tuple[int, int, int, int]) -> tuple[float, float]:
    x, y, width, height = bar
    return (float(x + (width - 1) / 2), float(y + (height - 1) / 2))


def _preserve_overlapping_dialog(
    clip_start: float,
    clip_end: float,
    dialog_segments: Sequence[DialogSegment],
    source_duration: float,
) -> float:
    adjusted_end = clip_end
    for segment in dialog_segments:
        if segment.start < adjusted_end < segment.end:
            adjusted_end = segment.end + config.DIALOG_PADDING
        elif clip_start <= segment.start <= adjusted_end and segment.end > adjusted_end:
            adjusted_end = segment.end + config.DIALOG_PADDING
    return min(source_duration, adjusted_end)


def _detect_combat_event_time(
    full_frames: np.ndarray,
    timestamps: np.ndarray,
    fight_start: float,
    fight_end: float,
    min_event_time: float | None = None,
) -> tuple[float | None, str]:
    if len(full_frames) == 0 or len(timestamps) == 0:
        return None, ""
    search_end = min(float(timestamps[-1]), fight_end + config.COMBAT_EVENT_SEARCH_AFTER_FIGHT_SEC)
    indexes = np.flatnonzero((timestamps >= fight_start) & (timestamps <= search_end))
    if len(indexes) < 3:
        return None, ""

    enemy_counts: list[int] = []
    player_present: list[bool] = []
    times: list[float] = []
    for index in indexes:
        frame = full_frames[int(index)]
        red_bars, green_bars = _combat_health_bars(frame)
        player_bar = _select_player_health_bar(green_bars)
        visible_enemies = _enemy_bars_near_player(red_bars, player_bar)
        enemy_counts.append(len(visible_enemies))
        player_present.append(player_bar is not None)
        times.append(float(timestamps[int(index)]))

    visible_enemy_samples = 0
    missing_enemy_run = 0
    missing_player_run = 0
    for idx, enemy_count in enumerate(enemy_counts):
        engaged = visible_enemy_samples >= config.COMBAT_EVENT_MIN_VISIBLE_ENEMY_FRAMES
        if enemy_count > 0:
            visible_enemy_samples += 1
            missing_enemy_run = 0
        elif visible_enemy_samples >= config.COMBAT_EVENT_MIN_VISIBLE_ENEMY_FRAMES:
            missing_enemy_run += 1

        engaged = visible_enemy_samples >= config.COMBAT_EVENT_MIN_VISIBLE_ENEMY_FRAMES
        if engaged and not player_present[idx]:
            missing_player_run += 1
        else:
            missing_player_run = 0

        if engaged and missing_player_run >= config.COMBAT_EVENT_MISSING_FRAMES:
            event_time = times[max(0, idx - missing_player_run + 1)]
            if min_event_time is None or event_time >= min_event_time:
                return event_time, "death_event_detected"
        if engaged and missing_enemy_run >= config.COMBAT_EVENT_MISSING_FRAMES:
            event_time = times[max(0, idx - missing_enemy_run + 1)]
            if min_event_time is None or event_time >= min_event_time:
                return event_time, "kill_event_detected"
    return None, ""


def _combat_health_bars(frame: np.ndarray) -> tuple[list[tuple[int, int, int, int]], list[tuple[int, int, int, int]]]:
    h, w = frame.shape[:2]
    roi_y1, roi_y2 = int(h * 0.08), int(h * 0.82)
    roi_x1, roi_x2 = int(w * 0.02), int(w * 0.96)
    roi = frame[roi_y1:roi_y2, roi_x1:roi_x2]
    red_mask = _mask_color(roi, "red")
    green_mask = _mask_color(roi, "green")
    red_bars = _filter_side_hud_bars(_health_bar_boxes(red_mask, roi_x1, roi_y1), w, h)
    green_bars = _filter_side_hud_bars(_health_bar_boxes(green_mask, roi_x1, roi_y1), w, h)
    return red_bars, green_bars


def _mask_color(roi: np.ndarray, color: str) -> np.ndarray:
    hsv = cv2.cvtColor(roi, cv2.COLOR_RGB2HSV)
    if color == "red":
        mask1 = cv2.inRange(hsv, (0, 80, 90), (12, 255, 255))
        mask2 = cv2.inRange(hsv, (168, 80, 90), (180, 255, 255))
        return cv2.bitwise_or(mask1, mask2)
    return cv2.inRange(hsv, (35, 70, 80), (90, 255, 255))


def _health_bar_boxes(mask: np.ndarray, offset_x: int, offset_y: int) -> list[tuple[int, int, int, int]]:
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 2))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes: list[tuple[int, int, int, int]] = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if not config.COMBAT_HEALTHBAR_MIN_WIDTH <= w <= config.COMBAT_HEALTHBAR_MAX_WIDTH:
            continue
        if not 3 <= h <= 18:
            continue
        boxes.append((x + offset_x, y + offset_y, w, h))
    return boxes


def _filter_side_hud_bars(
    boxes: list[tuple[int, int, int, int]],
    frame_w: int,
    frame_h: int,
) -> list[tuple[int, int, int, int]]:
    max_x = frame_w * config.COMBAT_HEALTHBAR_IGNORE_LEFT_X_PCT
    max_y = frame_h * config.COMBAT_HEALTHBAR_IGNORE_LEFT_Y_MAX_PCT
    return [box for box in boxes if not (box[0] < max_x and box[1] < max_y)]


def _select_player_health_bar(green_bars: list[tuple[int, int, int, int]]) -> tuple[int, int, int, int] | None:
    if not green_bars:
        return None
    return max(green_bars, key=lambda box: (box[2], -box[1]))


def _enemy_bars_near_player(
    red_bars: list[tuple[int, int, int, int]],
    player_bar: tuple[int, int, int, int] | None,
) -> list[tuple[int, int, int, int]]:
    if player_bar is None:
        return red_bars
    px = player_bar[0] + player_bar[2] / 2
    py = player_bar[1] + player_bar[3] / 2
    nearby = []
    for bar in red_bars:
        bx = bar[0] + bar[2] / 2
        by = bar[1] + bar[3] / 2
        if abs(bx - px) <= 420 and abs(by - py) <= 260:
            nearby.append(bar)
    return nearby


class FightDetector:
    def __init__(self) -> None:
        self.videomae_loaded = False
        self.whisper_loaded = False
        self._videomae_model = None
        self._videomae_device = None
        self._videomae_load_error: str | None = None
        self._whisper_model = None
        self._whisper_load_error: str | None = None
        if not config.VIDEOMAE_CHECKPOINT.exists():
            logger.warning("VideoMAE checkpoint missing; pretrained/fallback scoring will be used")

    def _load_videomae(self):
        if self._videomae_model is not None and self._videomae_device is not None:
            return self._videomae_model, self._videomae_device
        if self._videomae_load_error:
            raise RuntimeError(self._videomae_load_error)

        try:
            import torch
            import torch.nn as nn
            from transformers import VideoMAEModel

            checkpoint = Path(str(config.VIDEOMAE_CHECKPOINT))
            if not checkpoint.exists():
                raise FileNotFoundError(f"VideoMAE checkpoint not found at {checkpoint}")

            device = torch.device(CUDA_DEVICE_TYPE if torch.cuda.is_available() else "cpu")

            class _Classifier(nn.Module):
                def __init__(self) -> None:
                    super().__init__()
                    self.videomae = VideoMAEModel.from_pretrained("MCG-NJU/videomae-base")
                    self.classifier = nn.Linear(768, 2)

                def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
                    out = self.videomae(pixel_values=pixel_values)
                    pooled = out.last_hidden_state.mean(dim=1)
                    return self.classifier(pooled)

            model = _Classifier().to(device)
            model.load_state_dict(torch.load(str(checkpoint), map_location=device))
            model.eval()
            self._videomae_model = model
            self._videomae_device = device
            self.videomae_loaded = True
            return model, device
        except Exception as exc:  # noqa: BLE001 - inference can fall back to heuristics.
            self._videomae_load_error = str(exc)
            raise

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

        try:
            import torch
            import torchvision.transforms.functional as TF
            model, device = self._load_videomae()

            mean = [0.485, 0.456, 0.406]
            std = [0.229, 0.224, 0.225]
            scores: list[float] = []
            max_second = int(float(timestamps[-1])) if len(timestamps) else 0

            with torch.no_grad():
                for second in range(max(0, max_second - 15)):
                    indices = np.where(
                        (timestamps >= second) & (timestamps < second + 16)
                    )[0]
                    if len(indices) == 0:
                        scores.append(0.0)
                        continue
                    selected = indices[
                        np.linspace(0, len(indices) - 1, 16).astype(int)
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

            return _combine_scores(scores, _healthbar_engagement_scores(full_frames, timestamps))

        except Exception as exc:
            logger.warning(
                "VideoMAE inference failed (%s) - heuristic fallback",
                exc,
            )
            heuristic_scores = self._heuristic_scores(full_frames, timestamps)
            return _combine_scores(heuristic_scores, _healthbar_engagement_scores(full_frames, timestamps))

    def _heuristic_scores(
        self,
        full_frames: np.ndarray,
        timestamps: np.ndarray,
    ) -> list[float]:
        """Red-dominance fallback - used when VideoMAE checkpoint missing."""
        scores: list[float] = []
        max_second = int(float(timestamps[-1])) if len(timestamps) else max(0, len(full_frames) - 1)
        for second in range(max(0, max_second - 15)):
            frame_indices = np.where((timestamps >= second) & (timestamps < second + 16))[0]
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
            if self._whisper_load_error:
                raise RuntimeError(self._whisper_load_error)
            if self._whisper_model is None:
                from faster_whisper import WhisperModel

                self._whisper_model = WhisperModel("base", device="cpu", compute_type="int8")
                self.whisper_loaded = True
            model = self._whisper_model
            segments, _info = model.transcribe(str(audio_path))
            return [
                DialogSegment(text=seg.text.strip(), start=float(seg.start), end=float(seg.end))
                for seg in segments
                if seg.text.strip()
            ]
        except Exception as exc:  # noqa: BLE001 - Whisper absence should not kill the pipeline.
            self._whisper_load_error = str(exc)
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
