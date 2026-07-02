from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from . import config
from .minimap_detector import ChampionResult, map_pos_to_screen_hint


@dataclass(frozen=True)
class CropKeyframe:
    timestamp: float
    crop_x: int
    crop_y: int = config.CROP_Y
    crop_w: int = config.CROP_W
    crop_h: int = config.CROP_H


def blend_target(fight_type: str, player_sx: float, threat_sx: float, flow_sx: float) -> float:
    if fight_type == "1v1":
        weights = config.BLEND_1V1
    elif fight_type.startswith("1v"):
        weights = config.BLEND_1VN
    else:
        weights = config.BLEND_NVN
    return weights[0] * player_sx + weights[1] * threat_sx + weights[2] * flow_sx


def enforce_safe_zone(crop_x: float, player_sx: float) -> float:
    player_x_in_crop = player_sx - crop_x
    if player_x_in_crop < config.PLAYER_SAFE_LEFT_PX:
        crop_x = player_sx - config.PLAYER_SAFE_LEFT_PX
    elif player_x_in_crop > config.PLAYER_SAFE_RIGHT_PX:
        crop_x = player_sx - config.PLAYER_SAFE_RIGHT_PX
    return crop_x


def enforce_center_preference(crop_x: float, player_sx: float) -> float:
    crop_center = crop_x + config.CROP_W / 2
    offset = player_sx - crop_center
    if offset < -config.PLAYER_CENTER_DEADZONE_PX:
        crop_x = player_sx + config.PLAYER_CENTER_DEADZONE_PX - config.CROP_W / 2
    elif offset > config.PLAYER_CENTER_DEADZONE_PX:
        crop_x = player_sx - config.PLAYER_CENTER_DEADZONE_PX - config.CROP_W / 2
    return crop_x


def include_threat_in_crop(crop_x: float, player_sx: float, threat_sx: float | None) -> float:
    if threat_sx is None:
        return crop_x
    threat_x_in_crop = threat_sx - crop_x
    if threat_x_in_crop < config.THREAT_FRAME_MARGIN_PX:
        crop_x = threat_sx - config.THREAT_FRAME_MARGIN_PX
    elif threat_x_in_crop > config.CROP_W - config.THREAT_FRAME_MARGIN_PX:
        crop_x = threat_sx - (config.CROP_W - config.THREAT_FRAME_MARGIN_PX)
    return enforce_player_framing(crop_x, player_sx)


def avoid_minimap_ui(crop_x: float, player_sx: float, frame_w: int = 1920) -> float:
    minimap_left = frame_w * config.MINIMAP_CROP_X_PCT
    max_without_minimap = minimap_left - config.CROP_W - config.MINIMAP_UI_AVOID_MARGIN_PX
    if crop_x <= max_without_minimap:
        return crop_x

    capped = max(0.0, max_without_minimap)
    player_x_in_capped_crop = player_sx - capped
    if config.PLAYER_SAFE_LEFT_PX <= player_x_in_capped_crop <= config.PLAYER_SAFE_RIGHT_PX:
        return capped
    return crop_x


def clamp_crop_x(crop_x: float, frame_w: int = 1920) -> int:
    return int(np.clip(round(crop_x), 0, frame_w - config.CROP_W))


def compute_threat_sx(
    player_map_pos: tuple[float, float],
    enemies: Sequence[ChampionResult],
    frame_size: tuple[int, int],
) -> float:
    if not enemies:
        return map_pos_to_screen_hint(player_map_pos, frame_size)[0]
    player = np.array(player_map_pos, dtype=np.float32)
    weighted = 0.0
    total = 0.0
    for enemy in enemies:
        enemy_pos = np.array(enemy.mean_pos, dtype=np.float32)
        distance = max(float(np.linalg.norm(enemy_pos - player)), 1.0)
        weight = enemy.confidence / distance
        weighted += map_pos_to_screen_hint(tuple(enemy_pos), frame_size)[0] * weight
        total += weight
    return weighted / total if total else map_pos_to_screen_hint(player_map_pos, frame_size)[0]


def stabilize_screen_positions(
    values: Sequence[float | None] | None,
    max_outlier_px: float = 320.0,
    half_window: int = 2,
) -> list[float | None] | None:
    """Discard single-frame detection jumps so one bad health-bar match
    (a minion wave, a ward, a plant) can't yank the camera sideways."""
    if not values:
        return None
    series = [None if value is None else float(value) for value in values]
    stabilized: list[float | None] = []
    for idx, value in enumerate(series):
        if value is None:
            stabilized.append(None)
            continue
        neighbors = [
            series[j]
            for j in range(max(0, idx - half_window), min(len(series), idx + half_window + 1))
            if series[j] is not None
        ]
        if len(neighbors) >= 3:
            median = float(np.median(neighbors))
            if abs(value - median) > max_outlier_px:
                stabilized.append(None)
                continue
        stabilized.append(value)
    return stabilized


def windowed_median(
    series: Sequence[float | None] | None,
    timestamps: np.ndarray,
    center_time: float,
    half_window_sec: float,
) -> float | None:
    """Median of valid detections in a small time window around a keyframe,
    instead of trusting whatever single frame happens to be nearest."""
    if not series or len(timestamps) == 0:
        return None
    ts = np.asarray(timestamps, dtype=np.float32)
    indexes = np.flatnonzero(np.abs(ts - center_time) <= half_window_sec)
    if len(indexes) == 0:
        indexes = np.array([int(np.argmin(np.abs(ts - center_time)))])
    valid = [series[int(i)] for i in indexes if int(i) < len(series) and series[int(i)] is not None]
    if not valid:
        return None
    return float(np.median(valid))


def limit_pan_speed(values: Sequence[float], times: Sequence[float]) -> list[float]:
    """Turn raw per-keyframe targets into a camera-like pan: seed the opening
    shot from the first few keyframes, ignore micro-adjustments, and cap the
    pan velocity so the view never snaps."""
    if not values:
        return []
    seed_count = min(max(1, config.CROP_START_SEED_KEYFRAMES), len(values))
    current = float(np.median(np.asarray(values[:seed_count], dtype=np.float32)))
    smoothed = [current]
    for idx in range(1, len(values)):
        dt = max(float(times[idx]) - float(times[idx - 1]), 1e-3)
        delta = float(values[idx]) - current
        if abs(delta) < config.PAN_DEADBAND_PX:
            smoothed.append(current)
            continue
        max_step = config.MAX_PAN_SPEED_PX_PER_SEC * dt
        current += float(np.clip(delta, -max_step, max_step))
        smoothed.append(current)
    return smoothed


def smooth_crop_values(
    values: Sequence[float],
    player_sx_values: Sequence[float],
    frame_w: int = 1920,
    threat_sx_values: Sequence[float | None] | None = None,
) -> np.ndarray:
    """Final safety pass after pan smoothing. Uses only the loose safe zone
    (not the tight center deadzone) so it corrects real framing violations
    without re-introducing jitter."""
    x = np.asarray(values, dtype=np.float32)
    if len(x) == 0:
        return x
    for idx, player_sx in enumerate(player_sx_values):
        value = enforce_safe_zone(float(x[idx]), float(player_sx))
        if threat_sx_values is not None:
            value = include_threat_in_crop(value, float(player_sx), threat_sx_values[idx])
        value = avoid_minimap_ui(value, float(player_sx), frame_w)
        x[idx] = clamp_crop_x(value, frame_w)
    return x


def enforce_player_framing(crop_x: float, player_sx: float) -> float:
    crop_x = enforce_safe_zone(crop_x, player_sx)
    return enforce_center_preference(crop_x, player_sx)


class AdaptiveCropper:
    def compute_keyframes(
        self,
        frames: np.ndarray,
        timestamps: np.ndarray,
        clip_start: float,
        clip_end: float,
        player_positions: Sequence[tuple[float, float] | None],
        enemies: Sequence[ChampionResult],
        fight_type: str,
        player_screen_x_positions: Sequence[float | None] | None = None,
        threat_screen_x_positions: Sequence[float | None] | None = None,
    ) -> list[CropKeyframe]:
        frame_h, frame_w = frames.shape[1:3]
        if config.CROP_MODE == "static":
            # Locked in-game camera already keeps the champion framed, so a
            # single fixed crop produces the steadiest, most natural output.
            crop_x = clamp_crop_x(float(config.STATIC_CROP_X), frame_w)
            return [CropKeyframe(float(clip_start), crop_x)]

        key_times = _trajectory_times(clip_start, clip_end)

        if config.CROP_MODE == "hybrid":
            return self._hybrid_keyframes(
                key_times,
                timestamps,
                stabilize_screen_positions(player_screen_x_positions),
                stabilize_screen_positions(threat_screen_x_positions),
                frame_w,
            )

        player_series = stabilize_screen_positions(player_screen_x_positions)
        threat_series = stabilize_screen_positions(threat_screen_x_positions)

        raw_x: list[float] = []
        player_sx_values: list[float] = []
        threat_sx_values: list[float | None] = []
        for timestamp in key_times:
            player_sx = windowed_median(
                player_series, timestamps, float(timestamp), config.PLAYER_SX_MEDIAN_WINDOW_SEC
            )
            if player_sx is None:
                # The recorded view already follows the camera. Without a
                # confirmed player health bar, stay centered instead of
                # guessing from unrelated bars.
                player_sx = frame_w / 2
            threat_sx = windowed_median(
                threat_series, timestamps, float(timestamp), config.PLAYER_SX_MEDIAN_WINDOW_SEC
            )
            blend_threat_sx = threat_sx if threat_sx is not None else player_sx
            target = blend_target(fight_type, player_sx, blend_threat_sx, player_sx)
            crop_x = enforce_player_framing(target - config.CROP_W / 2, player_sx)
            crop_x = include_threat_in_crop(crop_x, player_sx, threat_sx)
            crop_x = avoid_minimap_ui(crop_x, player_sx, frame_w)
            raw_x.append(float(np.clip(crop_x, 0, frame_w - config.CROP_W)))
            player_sx_values.append(player_sx)
            threat_sx_values.append(threat_sx)

        panned = limit_pan_speed(raw_x, [float(t) for t in key_times])
        smoothed = smooth_crop_values(panned, player_sx_values, frame_w, threat_sx_values)
        return [CropKeyframe(float(t), int(x)) for t, x in zip(key_times, smoothed)]

    def _hybrid_keyframes(
        self,
        key_times: np.ndarray,
        timestamps: np.ndarray,
        player_series: list[float | None] | None,
        threat_series: list[float | None] | None,
        frame_w: int,
    ) -> list[CropKeyframe]:
        """Camera style for locked-cam recordings: hold a steady centered shot,
        and only reposition toward a flank when threats persist on that side.
        The result reads as deliberate reframing, not continuous sliding."""
        base_x = float(clamp_crop_x(float(config.STATIC_CROP_X), frame_w))
        interval = float(key_times[1] - key_times[0]) if len(key_times) > 1 else 1.0
        hold_samples = max(1, int(round(config.HYBRID_HOLD_SEC / max(interval, 1e-3))))

        # Which side of the champion are the threats on, per keyframe?
        sides: list[int] = []
        player_sx_values: list[float] = []
        for timestamp in key_times:
            player_sx = windowed_median(
                player_series, timestamps, float(timestamp), config.PLAYER_SX_MEDIAN_WINDOW_SEC
            )
            if player_sx is None:
                player_sx = frame_w / 2
            threat_sx = windowed_median(
                threat_series, timestamps, float(timestamp), config.PLAYER_SX_MEDIAN_WINDOW_SEC
            )
            if threat_sx is None:
                sides.append(0)
            else:
                delta = threat_sx - player_sx
                if delta <= -config.HYBRID_SIDE_TRIGGER_PX:
                    sides.append(-1)
                elif delta >= config.HYBRID_SIDE_TRIGGER_PX:
                    sides.append(1)
                else:
                    sides.append(0)
            player_sx_values.append(float(player_sx))

        # Hysteresis: commit to a side (or back to center) only after it
        # persists for HYBRID_HOLD_SEC worth of keyframes.
        committed = 0
        candidate = 0
        streak = 0
        desired: list[int] = []
        for side in sides:
            if side == candidate:
                streak += 1
            else:
                candidate = side
                streak = 1
            if candidate != committed and streak >= hold_samples:
                committed = candidate
            desired.append(committed)

        # Budget: at most HYBRID_MAX_VIEW_CHANGES cuts per clip, spaced at
        # least HYBRID_MIN_CUT_SPACING_SEC apart. Once the budget is spent the
        # camera holds for the rest of the clip.
        changes_used = 0
        last_change_time = -1e9
        current_side = 0
        targets: list[float] = []
        for key_time, side in zip(key_times, desired):
            t = float(key_time)
            if side != current_side:
                can_cut = (
                    changes_used < config.HYBRID_MAX_VIEW_CHANGES
                    and t - last_change_time >= config.HYBRID_MIN_CUT_SPACING_SEC
                )
                if can_cut:
                    current_side = side
                    changes_used += 1
                    last_change_time = t
            targets.append(base_x + current_side * config.HYBRID_OFFSET_PX)

        if config.CROP_TRANSITION == "cut":
            # Hold each position and snap to the next one, like a camera cut.
            positioned = targets
        else:
            positioned = limit_pan_speed(targets, [float(t) for t in key_times])
        smoothed: list[int] = []
        for value, player_sx in zip(positioned, player_sx_values):
            value = enforce_safe_zone(float(value), player_sx)
            value = avoid_minimap_ui(value, player_sx, frame_w)
            smoothed.append(clamp_crop_x(value, frame_w))
        return [CropKeyframe(float(t), int(x)) for t, x in zip(key_times, smoothed)]

    def interpolate_to_frames(
        self, keyframes: Sequence[CropKeyframe], source_timestamps: np.ndarray
    ) -> list[tuple[int, int, int, int]]:
        if not keyframes:
            return []
        key_times = np.array([k.timestamp for k in keyframes], dtype=np.float32)
        key_x = np.array([k.crop_x for k in keyframes], dtype=np.float32)
        ts = np.asarray(source_timestamps, dtype=np.float32)
        if config.CROP_TRANSITION == "cut":
            # Hold each keyframe's position until the next keyframe: hard cut.
            indexes = np.clip(np.searchsorted(key_times, ts, side="right") - 1, 0, len(key_x) - 1)
            x_values = key_x[indexes]
        else:
            x_values = np.interp(ts, key_times, key_x)
        return [(int(round(float(x))), config.CROP_Y, config.CROP_W, config.CROP_H) for x in x_values]


def _trajectory_times(clip_start: float, clip_end: float) -> np.ndarray:
    duration = max(0.0, clip_end - clip_start)
    if duration <= 1e-6:
        return np.array([clip_start], dtype=np.float32)
    count = min(config.MAX_CROP_KEYFRAMES, max(1, int(np.ceil(duration / max(config.KEYFRAME_INTERVAL_SEC, 1e-6))) + 1))
    return np.linspace(clip_start, clip_end, count, dtype=np.float32)
