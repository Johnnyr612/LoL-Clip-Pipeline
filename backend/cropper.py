from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from . import config


@dataclass(frozen=True)
class CropKeyframe:
    timestamp: float
    crop_x: int
    crop_y: int = config.CROP_Y
    crop_w: int = config.CROP_W
    crop_h: int = config.CROP_H


@dataclass(frozen=True)
class CropSettings:
    mode: str = config.CROP_MODE
    transition: str = config.CROP_TRANSITION


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


def _player_composition() -> str:
    composition = config.PLAYER_COMPOSITION.strip().lower()
    return composition if composition in {"center", "thirds"} else "thirds"


def _crop_mode(value: str | None = None) -> str:
    mode = (value or config.CROP_MODE).strip().lower()
    return mode if mode in {"static", "dynamic"} else "dynamic"


def _crop_transition(value: str | None = None) -> str:
    transition = (value or config.CROP_TRANSITION).strip().lower()
    return transition if transition in {"cut", "pan"} else "cut"


def _threat_side(player_sx: float, threat_sx: float | None) -> int:
    if threat_sx is None:
        return 0
    delta = threat_sx - player_sx
    if delta <= -config.DYNAMIC_THREAT_SIDE_TRIGGER_PX:
        return -1
    if delta >= config.DYNAMIC_THREAT_SIDE_TRIGGER_PX:
        return 1
    return 0


def player_anchor_x(player_sx: float, threat_sx: float | None) -> float:
    if _player_composition() != "thirds":
        return config.CROP_W / 2
    side = _threat_side(player_sx, threat_sx)
    look_room = max(0, int(config.PLAYER_THIRDS_LOOK_ROOM_PX))
    if side > 0:
        return max(config.PLAYER_SAFE_LEFT_PX, config.CROP_W / 3 - look_room)
    if side < 0:
        return min(config.PLAYER_SAFE_RIGHT_PX, config.CROP_W * 2 / 3 + look_room)
    return config.CROP_W / 2


def rule_of_thirds_crop_x(player_sx: float, threat_sx: float | None) -> float:
    return player_sx - player_anchor_x(player_sx, threat_sx)


def enforce_thirds_preference(crop_x: float, player_sx: float, threat_sx: float | None) -> float:
    anchor_x = player_anchor_x(player_sx, threat_sx)
    if anchor_x == config.CROP_W / 2:
        return enforce_center_preference(crop_x, player_sx)
    player_x_in_crop = player_sx - crop_x
    offset = player_x_in_crop - anchor_x
    if abs(offset) > config.PLAYER_THIRDS_DEADZONE_PX:
        crop_x = player_sx - anchor_x
    return enforce_safe_zone(crop_x, player_sx)


def include_threat_in_crop(crop_x: float, player_sx: float, threat_sx: float | None) -> float:
    if threat_sx is None:
        return crop_x
    threat_x_in_crop = threat_sx - crop_x
    if threat_x_in_crop < config.THREAT_FRAME_MARGIN_PX:
        crop_x = threat_sx - config.THREAT_FRAME_MARGIN_PX
    elif threat_x_in_crop > config.CROP_W - config.THREAT_FRAME_MARGIN_PX:
        crop_x = threat_sx - (config.CROP_W - config.THREAT_FRAME_MARGIN_PX)
    if _player_composition() == "thirds" and _threat_side(player_sx, threat_sx) != 0:
        preferred = enforce_thirds_preference(crop_x, player_sx, threat_sx)
        threat_x_in_preferred_crop = threat_sx - preferred
        if config.THREAT_FRAME_MARGIN_PX <= threat_x_in_preferred_crop <= config.CROP_W - config.THREAT_FRAME_MARGIN_PX:
            return preferred
        return enforce_safe_zone(crop_x, player_sx)
    return enforce_player_framing(crop_x, player_sx, threat_sx)


def combat_focus_crop_x(
    player_sx: float,
    threat_sx: float | None,
    frame_w: int = 1920,
    preferred_crop_x: float | None = None,
) -> float:
    """Prefer the player centered, then shift only enough to keep the threat
    visible. If both cannot fit, keep the player in the safe zone."""
    frame_low = 0.0
    frame_high = max(0.0, float(frame_w - config.CROP_W))
    player_low = max(frame_low, float(player_sx) - config.PLAYER_SAFE_RIGHT_PX)
    player_high = min(frame_high, float(player_sx) - config.PLAYER_SAFE_LEFT_PX)
    if player_low > player_high:
        player_low, player_high = frame_low, frame_high

    preferred = float(player_sx) - config.CROP_W / 2 if preferred_crop_x is None else float(preferred_crop_x)
    preferred = float(np.clip(preferred, player_low, player_high))
    if threat_sx is None:
        return preferred

    threat_low = float(threat_sx) - (config.CROP_W - config.THREAT_FRAME_MARGIN_PX)
    threat_high = float(threat_sx) - config.THREAT_FRAME_MARGIN_PX
    both_low = max(player_low, threat_low, frame_low)
    both_high = min(player_high, threat_high, frame_high)
    if both_low <= both_high:
        return float(np.clip(preferred, both_low, both_high))

    # The player and threat are too far apart for an 810px crop. Move toward
    # the threat as far as the player's safe zone allows.
    if float(threat_sx) < float(player_sx):
        return player_low
    return player_high


def _player_safe_crop_range(player_sx: float, frame_w: int = 1920) -> tuple[float, float]:
    frame_low = 0.0
    frame_high = max(0.0, float(frame_w - config.CROP_W))
    player_low = max(frame_low, float(player_sx) - config.PLAYER_SAFE_RIGHT_PX)
    player_high = min(frame_high, float(player_sx) - config.PLAYER_SAFE_LEFT_PX)
    if player_low > player_high:
        return frame_low, frame_high
    return player_low, player_high


def _visible_threat_crop_range(
    player_sx: float,
    threat_sx: float | None,
    frame_w: int = 1920,
) -> tuple[float, float] | None:
    if threat_sx is None:
        return None
    player_low, player_high = _player_safe_crop_range(player_sx, frame_w)
    frame_low = 0.0
    frame_high = max(0.0, float(frame_w - config.CROP_W))
    threat_low = float(threat_sx) - (config.CROP_W - config.THREAT_FRAME_MARGIN_PX)
    threat_high = float(threat_sx) - config.THREAT_FRAME_MARGIN_PX
    low = max(player_low, threat_low, frame_low)
    high = min(player_high, threat_high, frame_high)
    if low > high:
        return None
    return low, high


def dynamic_rule_of_thirds_crop_x(
    player_sx: float,
    threat_sx: float | None,
    frame_w: int = 1920,
    preferred_crop_x: float | None = None,
) -> float:
    """Dynamic mode contract: stay centered unless a threat can fit, then use
    the player-on-thirds composition while keeping the enemy visible."""
    center = float(clamp_crop_x(float(config.STATIC_CROP_X), frame_w))
    fit_range = _visible_threat_crop_range(player_sx, threat_sx, frame_w)
    if fit_range is None:
        if preferred_crop_x is not None and threat_sx is not None and _threat_is_visible(float(preferred_crop_x), threat_sx):
            player_x_in_preferred_crop = float(player_sx) - float(preferred_crop_x)
            if config.PLAYER_SAFE_LEFT_PX <= player_x_in_preferred_crop <= config.PLAYER_SAFE_RIGHT_PX:
                return float(preferred_crop_x)
        return center
    low, high = fit_range
    thirds_target = rule_of_thirds_crop_x(player_sx, threat_sx)
    preferred = thirds_target if preferred_crop_x is None else float(preferred_crop_x)
    return float(np.clip(preferred, low, high))


def _threat_is_visible(crop_x: float, threat_sx: float | None) -> bool:
    if threat_sx is None:
        return True
    threat_x_in_crop = float(threat_sx) - float(crop_x)
    return config.THREAT_FRAME_MARGIN_PX <= threat_x_in_crop <= config.CROP_W - config.THREAT_FRAME_MARGIN_PX


def avoid_minimap_ui(
    crop_x: float,
    player_sx: float,
    frame_w: int = 1920,
    threat_sx: float | None = None,
) -> float:
    minimap_left = frame_w * config.MINIMAP_CROP_X_PCT
    max_without_minimap = minimap_left - config.CROP_W - config.MINIMAP_UI_AVOID_MARGIN_PX
    if crop_x <= max_without_minimap:
        return crop_x

    capped = max(0.0, max_without_minimap)
    if threat_sx is not None and _threat_is_visible(crop_x, threat_sx) and not _threat_is_visible(capped, threat_sx):
        return crop_x
    player_x_in_capped_crop = player_sx - capped
    if config.PLAYER_SAFE_LEFT_PX <= player_x_in_capped_crop <= config.PLAYER_SAFE_RIGHT_PX:
        return capped
    return crop_x


def clamp_crop_x(crop_x: float, frame_w: int = 1920) -> int:
    return int(np.clip(round(crop_x), 0, frame_w - config.CROP_W))


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
    return _limit_pan_speed(values, times, config.CROP_START_SEED_KEYFRAMES)


def _limit_pan_speed(values: Sequence[float], times: Sequence[float], seed_keyframes: int) -> list[float]:
    if not values:
        return []
    seed_count = min(max(1, seed_keyframes), len(values))
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
    include_threat: bool = True,
) -> np.ndarray:
    """Final safety pass after pan smoothing. Uses only the loose safe zone
    (not the tight center deadzone) so it corrects real framing violations
    without re-introducing jitter."""
    x = np.asarray(values, dtype=np.float32)
    if len(x) == 0:
        return x
    for idx, player_sx in enumerate(player_sx_values):
        if include_threat and threat_sx_values is not None:
            value = combat_focus_crop_x(float(player_sx), threat_sx_values[idx], frame_w, float(x[idx]))
        else:
            value = enforce_safe_zone(float(x[idx]), float(player_sx))
        threat_sx = threat_sx_values[idx] if threat_sx_values is not None else None
        value = avoid_minimap_ui(value, float(player_sx), frame_w, threat_sx)
        x[idx] = clamp_crop_x(value, frame_w)
    return x


def enforce_player_framing(crop_x: float, player_sx: float, threat_sx: float | None = None) -> float:
    crop_x = enforce_safe_zone(crop_x, player_sx)
    if _player_composition() == "thirds":
        return enforce_thirds_preference(crop_x, player_sx, threat_sx)
    return enforce_center_preference(crop_x, player_sx)


class AdaptiveCropper:
    def compute_keyframes(
        self,
        frames: np.ndarray,
        timestamps: np.ndarray,
        clip_start: float,
        clip_end: float,
        player_positions: Sequence[tuple[float, float] | None],
        enemies: Sequence[object],
        fight_type: str,
        player_screen_x_positions: Sequence[float | None] | None = None,
        threat_screen_x_positions: Sequence[float | None] | None = None,
        crop_settings: CropSettings | None = None,
    ) -> list[CropKeyframe]:
        frame_h, frame_w = frames.shape[1:3]
        settings = crop_settings or CropSettings()
        mode = _crop_mode(settings.mode)
        transition = _crop_transition(settings.transition)
        if mode == "static":
            # Locked in-game camera already keeps the champion framed, so a
            # single fixed crop produces the steadiest, most natural output.
            crop_x = clamp_crop_x(float(config.STATIC_CROP_X), frame_w)
            return [CropKeyframe(float(clip_start), crop_x)]

        key_times = _trajectory_times(clip_start, clip_end)

        if mode == "dynamic":
            return self._dynamic_keyframes(
                key_times,
                timestamps,
                stabilize_screen_positions(player_screen_x_positions),
                stabilize_screen_positions(threat_screen_x_positions),
                frame_w,
                transition,
            )

    def _dynamic_keyframes(
        self,
        key_times: np.ndarray,
        timestamps: np.ndarray,
        player_series: list[float | None] | None,
        threat_series: list[float | None] | None,
        frame_w: int,
        transition: str,
    ) -> list[CropKeyframe]:
        """Locked-camera dynamic crop: begin centered, shift to thirds only
        after a visible enemy side persists, then return to center when it
        cannot fit or disappears."""
        base_x = float(clamp_crop_x(float(config.STATIC_CROP_X), frame_w))
        locked_player_sx = base_x + config.CROP_W / 2
        interval = float(key_times[1] - key_times[0]) if len(key_times) > 1 else 1.0
        hold_samples = max(1, int(round(config.DYNAMIC_THREAT_HOLD_SEC / max(interval, 1e-3))))

        candidate_side = 0
        side_streak = 0
        committed_side = 0
        view_changes = 0
        raw_targets: list[float] = []
        player_targets: list[float] = []
        threat_targets: list[float | None] = []
        for index, timestamp in enumerate(key_times):
            player_sx = _trusted_player_sx(
                windowed_median(player_series, timestamps, float(timestamp), config.PLAYER_SX_MEDIAN_WINDOW_SEC),
                locked_player_sx,
            )
            threat_sx = windowed_median(
                threat_series, timestamps, float(timestamp), config.PLAYER_SX_MEDIAN_WINDOW_SEC
            )
            previous_crop_x = raw_targets[-1] if raw_targets else base_x
            side = _threat_side(locked_player_sx, threat_sx)
            if _visible_threat_crop_range(player_sx, threat_sx, frame_w) is None:
                side = 0

            keep_committed_view = (
                index > 0
                and side == 0
                and committed_side != 0
                and threat_sx is not None
                and _threat_is_visible(previous_crop_x, threat_sx)
            )
            if index == 0 or side == 0:
                if keep_committed_view:
                    side = committed_side
                else:
                    committed_side = 0
                    candidate_side = 0
                    side_streak = 0
            else:
                if side == candidate_side:
                    side_streak += 1
                else:
                    candidate_side = side
                    side_streak = 1
                if side != committed_side and side_streak >= hold_samples and view_changes < config.DYNAMIC_MAX_VIEW_CHANGES:
                    committed_side = side
                    view_changes += 1

            active_threat_sx = threat_sx if committed_side != 0 and side == committed_side else None
            player_targets.append(player_sx)
            threat_targets.append(active_threat_sx)
            preferred_crop_x = previous_crop_x if keep_committed_view else None
            target = dynamic_rule_of_thirds_crop_x(player_sx, active_threat_sx, frame_w, preferred_crop_x)
            target = avoid_minimap_ui(target, player_sx, frame_w, active_threat_sx)
            raw_targets.append(float(np.clip(target, 0, frame_w - config.CROP_W)))

        if transition == "pan":
            positioned = _limit_pan_speed(raw_targets, [float(t) for t in key_times], 1)
        else:
            positioned = _hold_small_crop_changes(raw_targets)
        finalized: list[int] = []
        for value, player_sx, threat_sx in zip(positioned, player_targets, threat_targets):
            value = dynamic_rule_of_thirds_crop_x(player_sx, threat_sx, frame_w, value)
            value = avoid_minimap_ui(value, player_sx, frame_w, threat_sx)
            finalized.append(clamp_crop_x(value, frame_w))
        return [CropKeyframe(float(t), int(x)) for t, x in zip(key_times, finalized)]

    def interpolate_to_frames(
        self,
        keyframes: Sequence[CropKeyframe],
        source_timestamps: np.ndarray,
        crop_settings: CropSettings | None = None,
    ) -> list[tuple[int, int, int, int]]:
        if not keyframes:
            return []
        transition = _crop_transition((crop_settings or CropSettings()).transition)
        key_times = np.array([k.timestamp for k in keyframes], dtype=np.float32)
        key_x = np.array([k.crop_x for k in keyframes], dtype=np.float32)
        ts = np.asarray(source_timestamps, dtype=np.float32)
        if transition == "cut":
            # Hold each keyframe's position until the next keyframe: hard cut.
            indexes = np.clip(np.searchsorted(key_times, ts, side="right") - 1, 0, len(key_x) - 1)
            x_values = key_x[indexes]
        else:
            x_values = np.interp(ts, key_times, key_x)
        return [(int(round(float(x))), config.CROP_Y, config.CROP_W, config.CROP_H) for x in x_values]


def _hold_small_crop_changes(values: Sequence[float]) -> list[float]:
    if not values:
        return []
    threshold = max(1.0, float(config.PAN_DEADBAND_PX))
    held = [float(values[0])]
    current = float(values[0])
    for value in values[1:]:
        if abs(float(value) - current) >= threshold:
            current = float(value)
        held.append(current)
    return held


def _trusted_player_sx(player_sx: float | None, locked_player_sx: float) -> float:
    if player_sx is None:
        return float(locked_player_sx)
    if abs(float(player_sx) - float(locked_player_sx)) > config.CROP_W * 0.45:
        return float(locked_player_sx)
    return float(player_sx)


def _trajectory_times(clip_start: float, clip_end: float) -> np.ndarray:
    duration = max(0.0, clip_end - clip_start)
    if duration <= 1e-6:
        return np.array([clip_start], dtype=np.float32)
    count = min(config.MAX_CROP_KEYFRAMES, max(1, int(np.ceil(duration / max(config.KEYFRAME_INTERVAL_SEC, 1e-6))) + 1))
    return np.linspace(clip_start, clip_end, count, dtype=np.float32)
