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
class TrimSettings:
    fight_start_preroll_sec: float = config.FIGHT_START_PREROLL_SEC
    output_context_padding_sec: float = config.OUTPUT_CONTEXT_PADDING_SEC
    combat_event_end_padding_sec: float = config.COMBAT_EVENT_END_PADDING_SEC
    max_pre_fight_lead_sec: float = config.MAX_PRE_FIGHT_LEAD_SEC
    min_clip_duration_sec: float = config.COMBAT_EVENT_MIN_CLIP_DURATION_SEC
    target_clip_duration_sec: float = config.COMBAT_EVENT_TARGET_CLIP_DURATION_SEC
    max_clip_duration_sec: float = config.MAX_CLIP_DURATION
    conservative_full_fight_trim: bool = config.CONSERVATIVE_FULL_FIGHT_TRIM
    model_only: bool = False


@dataclass(frozen=True)
class TrimResult:
    clip_start: float
    clip_end: float
    fight_start: float
    fight_end: float
    fight_duration: float
    dialog_segments: list[DialogSegment]
    flags: list[str]


class HighlightEditorError(RuntimeError):
    """Raised when the required highlight editor cannot produce a trim."""


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


def _walk_boundary(
    scores: Sequence[float],
    peak_idx: int,
    direction: int,
    threshold: float,
    gap_tolerance: int,
) -> int:
    """Walk outward from the score peak, tolerating short sub-threshold dips
    so a mid-fight lull doesn't truncate the boundary."""
    last_good = peak_idx
    gap = 0
    idx = peak_idx
    while True:
        idx += direction
        if idx < 0 or idx >= len(scores):
            break
        if scores[idx] >= threshold:
            last_good = idx
            gap = 0
        else:
            gap += 1
            if gap > gap_tolerance:
                break
    return last_good


def boundaries_from_scores(
    scores: Sequence[float],
    source_duration: float,
    trim_settings: TrimSettings | None = None,
) -> tuple[float, float, list[str]]:
    settings = trim_settings or TrimSettings()
    flags: list[str] = []
    if not scores or max(scores) < config.FIGHT_CONFIDENCE_THRESHOLD:
        center = source_duration / 2
        fallback_duration = settings.target_clip_duration_sec if settings.conservative_full_fight_trim else 10.0
        start = max(0.0, center - fallback_duration / 2)
        end = min(source_duration, start + fallback_duration)
        start = max(0.0, end - fallback_duration)
        return start, end, ["low_confidence"]

    peak_idx = int(np.argmax(np.array(scores)))
    # Walk left with a lower "onset" threshold: window scores ramp up as the
    # fight fills the 16s window, so the opening seconds sit below the main
    # confidence threshold even when the fight has clearly begun.
    left = _walk_boundary(
        scores, peak_idx, -1, config.FIGHT_ONSET_THRESHOLD, config.FIGHT_BOUNDARY_GAP_TOLERANCE_SEC
    )
    right = _walk_boundary(
        scores, peak_idx, 1, config.FIGHT_CONFIDENCE_THRESHOLD, config.FIGHT_BOUNDARY_GAP_TOLERANCE_SEC
    )
    # Window index s scores the interval [s, s+16), so detection lags the true
    # engagement. Pull the start back to capture the approach/poke phase.
    fight_start = max(0.0, float(left) - settings.fight_start_preroll_sec)
    fight_end = float(right + 16.0)
    if fight_end - fight_start < config.FIGHT_MIN_DURATION:
        fight_end = fight_start + config.FIGHT_MIN_DURATION
    if fight_end - fight_start > config.FIGHT_MAX_DURATION:
        # Never sacrifice the fight opening: cap by trimming the tail. The
        # kill/death detector extends the end again later if needed.
        fight_end = fight_start + config.FIGHT_MAX_DURATION
        flags.append("fight_capped_at_max_duration")
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
        # Keep the fight opening intact and trim the tail instead of
        # recentering, which used to push the start past the fight onset.
        clip_end = min(source_duration, clip_start + config.MAX_CLIP_DURATION)

    return TrimResult(
        clip_start=round(clip_start, 3),
        clip_end=round(clip_end, 3),
        fight_start=round(fight_start, 3),
        fight_end=round(fight_end, 3),
        fight_duration=round(fight_end - fight_start, 3),
        dialog_segments=list(dialog_segments),
        flags=flags,
    )


def _first_sustained_combat_healthbar_time(
    full_frames: np.ndarray,
    timestamps: np.ndarray,
    search_start: float,
    search_end: float,
) -> float | None:
    indexes = np.flatnonzero((timestamps >= search_start) & (timestamps <= search_end))
    if len(indexes) == 0:
        return None

    run_start: float | None = None
    run_length = 0
    required = max(1, config.COMBAT_EVENT_MIN_VISIBLE_ENEMY_FRAMES)
    for index in indexes:
        frame = full_frames[int(index)]
        red_bars, green_bars = _combat_health_bars(frame)
        player_bar = _select_player_health_bar(green_bars)
        visible_enemies = _enemy_bars_near_player(red_bars, player_bar)
        if player_bar is not None and visible_enemies:
            if run_length == 0:
                run_start = float(timestamps[int(index)])
            run_length += 1
            if run_length >= required:
                return run_start
        else:
            run_start = None
            run_length = 0
    return None


def _snap_trim_start_to_healthbar_onset(
    trim: TrimResult,
    full_frames: np.ndarray,
    timestamps: np.ndarray,
    source_duration: float,
    trim_settings: TrimSettings | None = None,
) -> TrimResult:
    if len(full_frames) == 0 or len(timestamps) == 0:
        return trim

    settings = trim_settings or TrimSettings()
    search_start = max(0.0, min(trim.clip_start, trim.fight_start))
    search_end = min(source_duration, trim.fight_end, search_start + config.HEALTHBAR_START_SNAP_SEARCH_SEC)
    onset = _first_sustained_combat_healthbar_time(full_frames, timestamps, search_start, search_end)
    if onset is None:
        return trim

    snapped_clip_start = max(0.0, onset - settings.fight_start_preroll_sec)
    if snapped_clip_start <= trim.clip_start + 0.001:
        return trim

    fight_start = float(onset)
    fight_end = trim.fight_end
    if fight_end - fight_start < config.FIGHT_MIN_DURATION:
        fight_end = min(source_duration, fight_start + config.FIGHT_MIN_DURATION)
    clip_end = max(trim.clip_end, fight_end)
    return TrimResult(
        clip_start=round(snapped_clip_start, 3),
        clip_end=round(min(source_duration, clip_end), 3),
        fight_start=round(fight_start, 3),
        fight_end=round(min(source_duration, fight_end), 3),
        fight_duration=round(max(0.0, min(source_duration, fight_end) - fight_start), 3),
        dialog_segments=trim.dialog_segments,
        flags=[*trim.flags, "fight_start_snapped_to_healthbar_onset"],
    )


def finish_on_kill_or_death(
    trim: TrimResult,
    full_frames: np.ndarray,
    timestamps: np.ndarray,
    source_duration: float,
    trim_settings: TrimSettings | None = None,
) -> TrimResult:
    settings = trim_settings or TrimSettings()
    min_clip_end = min(source_duration, trim.clip_start + settings.min_clip_duration_sec)
    target_clip_end = min(source_duration, trim.clip_start + settings.target_clip_duration_sec)
    max_clip_end = min(source_duration, trim.clip_start + settings.max_clip_duration_sec)
    event_time, event_flag = _detect_combat_event_time(
        full_frames,
        timestamps,
        trim.fight_start,
        max(trim.fight_end, max_clip_end - config.COMBAT_EVENT_SEARCH_AFTER_FIGHT_SEC),
        min_event_time=min_clip_end,
    )
    if event_time is None:
        # Without a confirmed kill/death, trust the detected fight end instead
        # of stretching every clip to a fixed target length. Conservative mode
        # restores the old always-extend behavior.
        fallback_end = target_clip_end if settings.conservative_full_fight_trim else min_clip_end
        clip_end = max(trim.clip_end, fallback_end)
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

    event_clip_end = min(max_clip_end, event_time + settings.combat_event_end_padding_sec)
    clip_end = max(trim.clip_end, event_clip_end)
    flags = [*trim.flags, event_flag]
    if event_clip_end > trim.clip_end:
        flags.append("clip_end_extended_to_combat_event")
    else:
        flags.append("combat_event_before_model_end_ignored_for_trim")
    if settings.conservative_full_fight_trim:
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


def add_output_context(
    trim: TrimResult,
    source_duration: float,
    trim_settings: TrimSettings | None = None,
) -> TrimResult:
    settings = trim_settings or TrimSettings()
    clip_start = max(0.0, trim.clip_start - settings.output_context_padding_sec)
    clip_end = min(source_duration, trim.clip_end + settings.output_context_padding_sec)
    flags = [*trim.flags, "output_context_padding_applied"]
    # Hard cap on dead air before the fight: dialog extension, preroll, and
    # padding combined may not push the start further back than this.
    earliest_start = max(0.0, trim.fight_start - settings.max_pre_fight_lead_sec)
    if clip_start < earliest_start:
        clip_start = earliest_start
        flags.append("pre_fight_lead_capped")
    if clip_end - clip_start > settings.max_clip_duration_sec:
        overflow = (clip_end - clip_start) - settings.max_clip_duration_sec
        front_room = trim.clip_start - clip_start
        back_room = clip_end - trim.clip_end
        # Trim the tail padding first; only eat into the front lead-in if
        # unavoidable, so the fight opening keeps its context.
        trim_back = min(back_room, overflow)
        trim_front = min(front_room, overflow - trim_back)
        remaining = overflow - trim_front - trim_back
        if remaining > 0:
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
        flags=flags,
    )


def apply_highlight_trim_settings(
    trim: TrimResult,
    full_frames: np.ndarray,
    timestamps: np.ndarray,
    source_duration: float,
    trim_settings: TrimSettings | None = None,
) -> TrimResult:
    settings = trim_settings or TrimSettings()
    if settings.model_only:
        return TrimResult(
            clip_start=trim.clip_start,
            clip_end=trim.clip_end,
            fight_start=trim.fight_start,
            fight_end=trim.fight_end,
            fight_duration=trim.fight_duration,
            dialog_segments=trim.dialog_segments,
            flags=[*trim.flags, "model_only_trim"],
        )
    trim = _snap_trim_start_to_healthbar_onset(trim, full_frames, timestamps, source_duration, settings)
    desired_start = max(0.0, trim.fight_start - settings.fight_start_preroll_sec)
    clip_start = min(trim.clip_start, desired_start)
    flags = list(trim.flags)
    if clip_start < trim.clip_start:
        flags.append("highlight_preroll_applied")
    adjusted = TrimResult(
        clip_start=round(clip_start, 3),
        clip_end=trim.clip_end,
        fight_start=trim.fight_start,
        fight_end=trim.fight_end,
        fight_duration=trim.fight_duration,
        dialog_segments=trim.dialog_segments,
        flags=flags,
    )
    adjusted = finish_on_kill_or_death(adjusted, full_frames, timestamps, source_duration, settings)
    return add_output_context(adjusted, source_duration, settings)


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


def _champion_bars(bars: list[tuple[int, int, int, int]]) -> list[tuple[int, int, int, int]]:
    """Keep thick champion health bars; width is only a low noise floor."""
    return [
        bar
        for bar in bars
        if bar[2] >= config.COMBAT_CHAMPION_HEALTHBAR_MIN_WIDTH
        and bar[3] >= config.COMBAT_CHAMPION_HEALTHBAR_MIN_HEIGHT
    ]


def _camera_threat_bars(bars: list[tuple[int, int, int, int]]) -> list[tuple[int, int, int, int]]:
    """Stricter crop-steering filter so camera motion only trusts thick red bars."""
    return [
        bar
        for bar in bars
        if bar[2] >= config.COMBAT_CAMERA_THREAT_HEALTHBAR_MIN_WIDTH
        and bar[3] >= config.COMBAT_CAMERA_THREAT_HEALTHBAR_MIN_HEIGHT
        and bar[2] * bar[3] >= config.COMBAT_CAMERA_THREAT_HEALTHBAR_MIN_AREA
    ]


def _exclude_objective_health_bars(
    frame: np.ndarray,
    bars: list[tuple[int, int, int, int]],
) -> list[tuple[int, int, int, int]]:
    return [bar for bar in bars if not _has_objective_health_text(frame, bar)]


def _has_objective_health_text(frame: np.ndarray, bar: tuple[int, int, int, int]) -> bool:
    if _has_champion_level_badge(frame, bar):
        return False

    x, y, width, height = bar
    frame_h, frame_w = frame.shape[:2]
    text_x1 = max(0, x + int(width * 0.12))
    text_x2 = min(frame_w, x + int(width * 0.88))
    text_y1 = max(0, y - max(32, height * 4))
    text_y2 = min(frame_h, y + max(2, height // 2))
    if text_x2 <= text_x1 or text_y2 <= text_y1:
        return False

    roi = frame[text_y1:text_y2, text_x1:text_x2]
    hsv = cv2.cvtColor(roi, cv2.COLOR_RGB2HSV)
    light_text = cv2.inRange(hsv, (0, 0, 145), (180, 95, 255))
    yellow_text = cv2.inRange(hsv, (18, 45, 120), (45, 255, 255))
    mask = cv2.bitwise_or(light_text, yellow_text)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (2, 1)))

    text_pixels = int(np.count_nonzero(mask))
    if text_pixels < max(8, int(width * 0.08)):
        return False

    component_count, _labels, stats, _centroids = cv2.connectedComponentsWithStats(mask, 8)
    glyphs = 0
    for index in range(1, component_count):
        _gx, _gy, glyph_w, glyph_h, area = stats[index]
        if 3 <= area <= 90 and 2 <= glyph_w <= 18 and 4 <= glyph_h <= 18:
            glyphs += 1
    return glyphs >= 2 or text_pixels >= max(18, int(width * 0.16))


def _has_champion_level_badge(frame: np.ndarray, bar: tuple[int, int, int, int]) -> bool:
    x, y, _width, height = bar
    frame_h, frame_w = frame.shape[:2]
    badge_x1 = max(0, x - 42)
    badge_x2 = max(0, min(frame_w, x + 4))
    badge_y1 = max(0, y - max(12, height + 2))
    badge_y2 = min(frame_h, y + height + max(12, height + 2))
    if badge_x2 <= badge_x1 or badge_y2 <= badge_y1:
        return False

    roi = frame[badge_y1:badge_y2, badge_x1:badge_x2]
    hsv = cv2.cvtColor(roi, cv2.COLOR_RGB2HSV)
    dark = cv2.inRange(hsv, (0, 0, 0), (180, 110, 85))
    light = cv2.inRange(hsv, (0, 0, 135), (180, 120, 255))
    yellow = cv2.inRange(hsv, (18, 45, 110), (45, 255, 255))
    glyph_mask = cv2.bitwise_or(light, yellow)
    dark_fraction = float(np.count_nonzero(dark)) / float(dark.size)
    glyph_pixels = int(np.count_nonzero(glyph_mask))
    badge_core_x2 = max(1, int(glyph_mask.shape[1] * 0.72))
    core_glyph_pixels = int(np.count_nonzero(glyph_mask[:, :badge_core_x2]))
    return (
        dark_fraction >= 0.18
        and glyph_pixels >= 2
        and core_glyph_pixels >= config.COMBAT_CHAMPION_BADGE_CORE_GLYPH_MIN_PIXELS
    )


def _champion_ui_bars(
    frame: np.ndarray,
    bars: list[tuple[int, int, int, int]],
) -> list[tuple[int, int, int, int]]:
    return [bar for bar in bars if _has_champion_level_badge(frame, bar)]


def estimate_combat_screen_x_positions(full_frames: np.ndarray) -> tuple[list[float | None], list[float | None]]:
    player_positions: list[float | None] = []
    threat_positions: list[float | None] = []
    for frame in full_frames:
        red_bars, green_bars = _combat_health_bars(frame)
        red_bars = _exclude_objective_health_bars(frame, _camera_threat_bars(red_bars))
        player_bar = _select_player_health_bar(green_bars)
        if player_bar is None:
            # Without a confirmed player bar there is no reliable anchor, so
            # do not let stray red bars (minion waves, jungle camps) pull the
            # crop toward them. Report no threat instead.
            player_positions.append(None)
            threat_positions.append(None)
            continue
        x, _, width, _ = player_bar
        player_positions.append(float(x + (width - 1) / 2))
        crop_enemy_bars = _enemy_bars_for_crop(red_bars)
        champion_ui_bars = _champion_ui_bars(frame, crop_enemy_bars)
        threat_positions.append(_threat_bar_center_x(champion_ui_bars, player_bar))
    return player_positions, _stabilize_sparse_threat_positions(threat_positions)


def _stabilize_sparse_threat_positions(values: list[float | None]) -> list[float | None]:
    if not values:
        return []
    stabilized: list[float | None] = []
    radius = max(1, config.COMBAT_CAMERA_THREAT_SUPPORT_RADIUS_FRAMES)
    required = max(2, config.COMBAT_CAMERA_THREAT_MIN_SUPPORT_SAMPLES)
    for index, value in enumerate(values):
        if value is None:
            stabilized.append(None)
            continue
        support = 1
        for neighbor_index in range(max(0, index - radius), min(len(values), index + radius + 1)):
            if neighbor_index == index:
                continue
            neighbor = values[neighbor_index]
            if neighbor is None:
                continue
            if abs(float(neighbor) - float(value)) <= config.COMBAT_CAMERA_THREAT_SUPPORT_TOLERANCE_PX:
                support += 1
        stabilized.append(float(value) if support >= required else None)
    return stabilized


def _nearest_bar_center_x(
    bars: Sequence[tuple[int, int, int, int]],
    player_bar: tuple[int, int, int, int],
) -> float | None:
    if not bars:
        return None
    px, py = _bar_center(player_bar)
    nearest = min(bars, key=lambda bar: float(np.linalg.norm(np.array(_bar_center(bar)) - np.array((px, py)))))
    return float(_bar_center(nearest)[0])


def _threat_bar_center_x(
    bars: Sequence[tuple[int, int, int, int]],
    player_bar: tuple[int, int, int, int],
) -> float | None:
    if not bars:
        return None

    crop_left = float(config.STATIC_CROP_X + config.THREAT_FRAME_MARGIN_PX)
    crop_right = float(config.STATIC_CROP_X + config.CROP_W - config.THREAT_FRAME_MARGIN_PX)
    leftmost = min(bars, key=lambda bar: _bar_center(bar)[0])
    rightmost = max(bars, key=lambda bar: _bar_center(bar)[0])
    left_pressure = max(0.0, crop_left - float(_bar_center(leftmost)[0]))
    right_pressure = max(0.0, float(_bar_center(rightmost)[0]) - crop_right)

    if left_pressure > right_pressure:
        return float(_bar_center(leftmost)[0])
    if right_pressure > left_pressure:
        return float(_bar_center(rightmost)[0])
    if left_pressure > 0:
        px, _py = _bar_center(player_bar)
        edge_bar = max((leftmost, rightmost), key=lambda bar: abs(float(_bar_center(bar)[0]) - float(px)))
        return float(_bar_center(edge_bar)[0])

    return _nearest_bar_center_x(bars, player_bar)


def _enemy_bars_for_crop(red_bars: list[tuple[int, int, int, int]]) -> list[tuple[int, int, int, int]]:
    champion_reds = _champion_bars(red_bars)
    return champion_reds


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
    # Crop steering intentionally only trusts the local player (green) and
    # enemies (red). Ally-blue bars are left unmasked so they cannot move view.
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
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 2))
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
    return [box for box in boxes if not _is_non_gameplay_healthbar_region(box, frame_w, frame_h)]


def _is_non_gameplay_healthbar_region(
    box: tuple[int, int, int, int],
    frame_w: int,
    frame_h: int,
) -> bool:
    x, y, _width, _height = box
    left_hud_max_x = frame_w * config.COMBAT_HEALTHBAR_IGNORE_LEFT_X_PCT
    left_hud_max_y = frame_h * config.COMBAT_HEALTHBAR_IGNORE_LEFT_Y_MAX_PCT
    if x < left_hud_max_x and y < left_hud_max_y:
        return True

    chat_max_x = frame_w * config.COMBAT_HEALTHBAR_IGNORE_CHAT_X_PCT
    chat_min_y = frame_h * config.COMBAT_HEALTHBAR_IGNORE_CHAT_Y_MIN_PCT
    if x < chat_max_x and y > chat_min_y:
        return True

    minimap_min_x = frame_w * config.COMBAT_HEALTHBAR_IGNORE_MINIMAP_X_PCT
    minimap_min_y = frame_h * config.COMBAT_HEALTHBAR_IGNORE_MINIMAP_Y_MIN_PCT
    return x > minimap_min_x and y > minimap_min_y


def _select_player_health_bar(green_bars: list[tuple[int, int, int, int]]) -> tuple[int, int, int, int] | None:
    if not green_bars:
        return None
    # The camera follows the recording player, so their health bar sits near
    # the middle of the screen (slightly above center, floating over the
    # champion). Prefer central bars; a wide green bar at the screen edge is
    # almost always allied minions, a ward, or a plant - not the player.
    frame_w, frame_h = 1920.0, 1080.0
    champion_greens = _champion_bars(green_bars)
    pool_source = champion_greens or green_bars
    central = [
        bar
        for bar in pool_source
        if abs((bar[0] + bar[2] / 2) - frame_w / 2) <= frame_w * 0.30
    ]
    pool = central or pool_source

    def selection_key(bar: tuple[int, int, int, int]) -> float:
        cx = bar[0] + bar[2] / 2
        cy = bar[1] + bar[3] / 2
        distance = ((cx - frame_w * 0.5) ** 2 + (cy - frame_h * 0.42) ** 2) ** 0.5
        return distance - bar[2] * 1.5  # closer to center wins; width breaks ties

    return min(pool, key=selection_key)


def _enemy_bars_near_player(
    red_bars: list[tuple[int, int, int, int]],
    player_bar: tuple[int, int, int, int] | None,
) -> list[tuple[int, int, int, int]]:
    champion_reds = _champion_bars(red_bars)
    if player_bar is None:
        return champion_reds
    px = player_bar[0] + player_bar[2] / 2
    py = player_bar[1] + player_bar[3] / 2
    nearby = []
    for bar in champion_reds:
        bx = bar[0] + bar[2] / 2
        by = bar[1] + bar[3] / 2
        if abs(bx - px) <= 420 and abs(by - py) <= 260:
            nearby.append(bar)
    return nearby


def _smooth_series(values: np.ndarray, radius: int = 1) -> np.ndarray:
    if len(values) == 0 or radius <= 0:
        return values
    smoothed = np.zeros_like(values, dtype=np.float32)
    for idx in range(len(values)):
        start = max(0, idx - radius)
        end = min(len(values), idx + radius + 1)
        smoothed[idx] = float(np.mean(values[start:end]))
    return smoothed


def _best_include_span(values: np.ndarray, threshold: float) -> tuple[float, float] | None:
    best: tuple[int, int, float] | None = None
    start: int | None = None
    score = 0.0
    for idx, value in enumerate(values):
        if float(value) >= threshold:
            if start is None:
                start = idx
                score = 0.0
            score += float(value)
        elif start is not None:
            candidate = (start, idx, score)
            if best is None or candidate[2] > best[2]:
                best = candidate
            start = None
    if start is not None:
        candidate = (start, len(values), score)
        if best is None or candidate[2] > best[2]:
            best = candidate
    return None if best is None else (float(best[0]), float(best[1]))


def _enforce_highlight_duration(start: float, end: float, valid_seconds: int) -> tuple[float, float]:
    min_duration = max(0.0, float(config.HIGHLIGHT_MIN_CLIP_DURATION_SEC))
    max_duration = max(min_duration, float(config.HIGHLIGHT_MAX_CLIP_DURATION_SEC))
    duration = end - start
    if duration < min_duration:
        center = (start + end) / 2.0
        start = center - min_duration / 2.0
        end = center + min_duration / 2.0
    if end - start > max_duration:
        center = (start + end) / 2.0
        start = center - max_duration / 2.0
        end = center + max_duration / 2.0
    if start < 0.0:
        end -= start
        start = 0.0
    if end > valid_seconds:
        start = max(0.0, start - (end - valid_seconds))
        end = float(valid_seconds)
    return max(0.0, start), max(start, end)


class FightDetector:
    def __init__(self) -> None:
        self.videomae_loaded = False
        self.whisper_loaded = False
        self._videomae_model = None
        self._videomae_device = None
        self._videomae_load_error: str | None = None
        self._highlight_model = None
        self._highlight_device = None
        self._highlight_checkpoint_path: Path | None = None
        self._highlight_load_error: str | None = None
        self._highlight_load_error_path: Path | None = None
        self._whisper_model = None
        self._whisper_load_error: str | None = None

    def _load_highlight_editor(self, checkpoint_path: Path | None = None):
        resolved_checkpoint = Path(str(checkpoint_path or config.VIDEOMAE_HIGHLIGHT_CHECKPOINT)).resolve()
        if (
            self._highlight_model is not None
            and self._highlight_device is not None
            and self._highlight_checkpoint_path == resolved_checkpoint
        ):
            return self._highlight_model, self._highlight_device
        if self._highlight_load_error and self._highlight_load_error_path == resolved_checkpoint:
            raise RuntimeError(self._highlight_load_error)

        try:
            import torch
            import torch.nn as nn
            import torch.nn.functional as F
            from transformers import VideoMAEModel

            if not resolved_checkpoint.exists():
                raise FileNotFoundError(f"VideoMAE highlight checkpoint not found at {resolved_checkpoint}")

            checkpoint = torch.load(str(resolved_checkpoint), map_location="cpu")
            context_seconds = int(checkpoint.get("context_seconds", config.HIGHLIGHT_CONTEXT_SECONDS))
            phase_count = len(checkpoint.get("phase_names", ("exclude", "buildup", "fight", "payoff")))
            state_dict = checkpoint.get("model_state", checkpoint)
            architecture = str(checkpoint.get("architecture") or "").strip()
            include_weight = state_dict.get("include_head.weight") if isinstance(state_dict, dict) else None
            if not architecture and include_weight is not None:
                architecture = (
                    "pooled_context_head"
                    if int(include_weight.shape[0]) == context_seconds * 2
                    else "temporal_token_head"
                )
            device = torch.device(CUDA_DEVICE_TYPE if torch.cuda.is_available() else "cpu")

            def temporal_video_features(
                hidden_states: torch.Tensor,
                pixel_values: torch.Tensor,
                videomae_config: object,
            ) -> torch.Tensor:
                batch_size, sequence_length, hidden_size = hidden_states.shape
                input_frames = int(pixel_values.shape[1])
                tubelet_size = int(getattr(videomae_config, "tubelet_size", 2) or 2)
                temporal_tokens = max(1, input_frames // max(1, tubelet_size))
                if sequence_length % temporal_tokens != 0:
                    temporal_tokens = min(input_frames, sequence_length)
                    while temporal_tokens > 1 and sequence_length % temporal_tokens != 0:
                        temporal_tokens -= 1
                spatial_tokens = max(1, sequence_length // max(1, temporal_tokens))
                usable_tokens = temporal_tokens * spatial_tokens
                temporal = hidden_states[:, :usable_tokens, :].reshape(
                    batch_size,
                    temporal_tokens,
                    spatial_tokens,
                    hidden_size,
                ).mean(dim=2)
                if temporal_tokens != context_seconds:
                    temporal = F.interpolate(
                        temporal.transpose(1, 2),
                        size=context_seconds,
                        mode="linear",
                        align_corners=False,
                    ).transpose(1, 2)
                return temporal

            class _HighlightEditor(nn.Module):
                def __init__(self) -> None:
                    super().__init__()
                    self.context_seconds = context_seconds
                    self.architecture = architecture or "temporal_token_head"
                    self.videomae = VideoMAEModel.from_pretrained("MCG-NJU/videomae-base")
                    if self.architecture == "pooled_context_head":
                        self.include_head = nn.Linear(768, context_seconds * 2)
                        self.phase_head = nn.Linear(768, context_seconds * phase_count)
                    else:
                        self.include_head = nn.Linear(768, 2)
                        self.phase_head = nn.Linear(768, phase_count)

                def forward(self, pixel_values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
                    out = self.videomae(pixel_values=pixel_values)
                    if self.architecture == "pooled_context_head":
                        pooled = out.last_hidden_state.mean(dim=1)
                        include_logits = self.include_head(pooled).view(-1, context_seconds, 2)
                        phase_logits = self.phase_head(pooled).view(-1, context_seconds, phase_count)
                        return include_logits, phase_logits
                    temporal = temporal_video_features(out.last_hidden_state, pixel_values, self.videomae.config)
                    include_logits = self.include_head(temporal)
                    phase_logits = self.phase_head(temporal)
                    return include_logits, phase_logits

            model = _HighlightEditor().to(device)
            model.load_state_dict(state_dict)
            model.eval()
            self._highlight_model = model
            self._highlight_device = device
            self._highlight_checkpoint_path = resolved_checkpoint
            self._highlight_load_error = None
            self._highlight_load_error_path = None
            return model, device
        except Exception as exc:  # noqa: BLE001 - surface the loader error on the next inference attempt.
            self._highlight_load_error = str(exc)
            self._highlight_load_error_path = resolved_checkpoint
            raise

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
        except Exception as exc:  # noqa: BLE001 - cache the loader error for consistent retries.
            self._videomae_load_error = str(exc)
            raise

    def predict_highlight_trim(
        self,
        full_frames: np.ndarray,
        timestamps: np.ndarray,
        source_duration: float,
        checkpoint_path: Path | None = None,
    ) -> TrimResult:
        if len(full_frames) == 0 or len(timestamps) == 0:
            raise HighlightEditorError("VideoMAE highlight editor received no decoded frames")
        try:
            import torch
            import torchvision.transforms.functional as TF

            try:
                model, device = self._load_highlight_editor(checkpoint_path)
            except Exception as exc:  # noqa: BLE001 - preserve the original loader failure as the cause.
                raise HighlightEditorError(f"VideoMAE highlight editor failed to load: {exc}") from exc

            context_seconds = int(getattr(model, "context_seconds", config.HIGHLIGHT_CONTEXT_SECONDS))
            valid_seconds = max(1, min(context_seconds, int(np.ceil(source_duration))))
            sample_count = max(1, config.HIGHLIGHT_INPUT_FRAMES)
            sample_times = np.linspace(0.0, max(float(valid_seconds) - 1.0, 0.0), sample_count)
            mean = [0.485, 0.456, 0.406]
            std = [0.229, 0.224, 0.225]
            tensors = []
            for sample_time in sample_times:
                index = int(np.argmin(np.abs(timestamps - float(sample_time))))
                frame = full_frames[min(index, len(full_frames) - 1)]
                tensor = torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0
                tensor = TF.resize(tensor, [224, 224], antialias=True)
                tensor = TF.normalize(tensor, mean, std)
                tensors.append(tensor)

            pixel_values = torch.stack(tensors).unsqueeze(0).to(device)
            with torch.no_grad():
                ctx = torch.cuda.amp.autocast() if device.type == CUDA_DEVICE_TYPE else contextlib.nullcontext()
                with ctx:
                    include_logits, phase_logits = model(pixel_values)
                include_probs = torch.softmax(include_logits[0], dim=-1)[:, 1].detach().cpu().numpy()
                phase_probs = torch.softmax(phase_logits[0], dim=-1).detach().cpu().numpy()

            include_probs = include_probs[:valid_seconds]
            phase_probs = phase_probs[:valid_seconds]
            smoothed = _smooth_series(include_probs, radius=1)
            span = _best_include_span(smoothed, config.HIGHLIGHT_INCLUDE_THRESHOLD)
            if span is None:
                peak = float(np.max(include_probs)) if len(include_probs) else 0.0
                raise HighlightEditorError(
                    "VideoMAE highlight editor did not select an include span "
                    f"(peak={peak:.3f}, threshold={config.HIGHLIGHT_INCLUDE_THRESHOLD:.3f})"
                )
            clip_start, clip_end = _enforce_highlight_duration(span[0], span[1], valid_seconds)
            if clip_end <= clip_start:
                raise HighlightEditorError(
                    f"VideoMAE highlight editor produced an invalid trim: {clip_start:.3f}s to {clip_end:.3f}s"
                )
            phase_pred = np.argmax(phase_probs, axis=1)
            fight_indexes = [
                idx
                for idx in range(int(np.floor(clip_start)), int(np.ceil(clip_end)))
                if 0 <= idx < len(phase_pred) and int(phase_pred[idx]) == 2
            ]
            fight_start = float(fight_indexes[0]) if fight_indexes else clip_start
            fight_end = float(fight_indexes[-1] + 1) if fight_indexes else clip_end
            flags = [
                "highlight_editor_model",
                f"highlight_include_peak={float(np.max(include_probs)):.3f}",
            ]
            return TrimResult(
                clip_start=round(float(clip_start), 3),
                clip_end=round(float(min(source_duration, clip_end)), 3),
                fight_start=round(float(fight_start), 3),
                fight_end=round(float(min(source_duration, fight_end)), 3),
                fight_duration=round(float(max(0.0, fight_end - fight_start)), 3),
                dialog_segments=[],
                flags=flags,
            )
        except HighlightEditorError:
            raise
        except Exception as exc:  # noqa: BLE001 - make editor inference failures visible to the job.
            raise HighlightEditorError(f"VideoMAE highlight editor inference failed: {exc}") from exc

    def score_windows(
        self,
        full_frames: np.ndarray,
        timestamps: np.ndarray,
    ) -> list[float]:
        """
        Score each 1-second window as P(fight) using VideoMAE.
        """
        if len(full_frames) == 0:
            return []

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

    def detect(
        self,
        full_frames: np.ndarray,
        timestamps: np.ndarray,
        source_duration: float,
        audio_path: Path | None,
        trim_settings: TrimSettings | None = None,
    ) -> TrimResult:
        fight_start, fight_end, flags = boundaries_from_scores(
            self.score_windows(full_frames, timestamps),
            source_duration,
            trim_settings,
        )
        dialog = self.transcribe(audio_path)
        result = apply_dialog_extension(fight_start, fight_end, source_duration, dialog)
        result = _snap_trim_start_to_healthbar_onset(
            result,
            full_frames,
            timestamps,
            source_duration,
            trim_settings,
        )
        return TrimResult(
            result.clip_start,
            result.clip_end,
            result.fight_start,
            result.fight_end,
            result.fight_duration,
            result.dialog_segments,
            flags + result.flags,
        )
