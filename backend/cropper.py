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


def smooth_crop_values(
    values: Sequence[float],
    player_sx_values: Sequence[float],
    frame_w: int = 1920,
    threat_sx_values: Sequence[float | None] | None = None,
) -> np.ndarray:
    x = np.asarray(values, dtype=np.float32)
    if len(x) == 0:
        return x
    for idx, player_sx in enumerate(player_sx_values):
        x[idx] = enforce_player_framing(float(x[idx]), float(player_sx))
        if threat_sx_values is not None:
            x[idx] = include_threat_in_crop(float(x[idx]), float(player_sx), threat_sx_values[idx])
        x[idx] = avoid_minimap_ui(float(x[idx]), float(player_sx), frame_w)
        x[idx] = clamp_crop_x(float(x[idx]), frame_w)
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
        key_times = _trajectory_times(clip_start, clip_end)
        raw_x: list[float] = []
        player_sx_values: list[float] = []
        threat_sx_values: list[float | None] = []
        previous_player_pos = (0.5, 0.5)
        for timestamp in key_times:
            frame_idx = int(np.argmin(np.abs(timestamps - timestamp)))
            player_pos = player_positions[min(frame_idx, len(player_positions) - 1)] if player_positions else None
            if player_pos is None:
                player_pos = previous_player_pos
            previous_player_pos = player_pos
            detected_player_sx = (
                player_screen_x_positions[min(frame_idx, len(player_screen_x_positions) - 1)]
                if player_screen_x_positions
                else None
            )
            # The minimap white box is the camera viewport already being recorded.
            # If no in-world green health bar is visible, center the crop in that recorded view.
            player_sx = float(detected_player_sx) if detected_player_sx is not None else frame_w / 2
            detected_threat_sx = (
                threat_screen_x_positions[min(frame_idx, len(threat_screen_x_positions) - 1)]
                if threat_screen_x_positions
                else None
            )
            threat_sx = float(detected_threat_sx) if detected_threat_sx is not None else None
            blend_threat_sx = threat_sx if threat_sx is not None else player_sx
            target = blend_target(fight_type, player_sx, blend_threat_sx, player_sx)
            crop_x = enforce_player_framing(target - config.CROP_W / 2, player_sx)
            crop_x = include_threat_in_crop(crop_x, player_sx, threat_sx)
            crop_x = avoid_minimap_ui(crop_x, player_sx, frame_w)
            raw_x.append(clamp_crop_x(crop_x, frame_w))
            player_sx_values.append(player_sx)
            threat_sx_values.append(threat_sx)
        smoothed = smooth_crop_values(raw_x, player_sx_values, frame_w, threat_sx_values)
        return [CropKeyframe(float(t), int(x)) for t, x in zip(key_times, smoothed)]

    def interpolate_to_frames(self, keyframes: Sequence[CropKeyframe], source_timestamps: np.ndarray) -> list[tuple[int, int, int, int]]:
        if not keyframes:
            return []
        key_times = np.array([k.timestamp for k in keyframes], dtype=np.float32)
        key_x = np.array([k.crop_x for k in keyframes], dtype=np.float32)
        indexes = np.searchsorted(key_times, source_timestamps, side="right") - 1
        indexes = np.clip(indexes, 0, len(key_x) - 1)
        x_values = key_x[indexes]
        return [(int(round(x)), config.CROP_Y, config.CROP_W, config.CROP_H) for x in x_values]


def _trajectory_times(clip_start: float, clip_end: float) -> np.ndarray:
    duration = max(0.0, clip_end - clip_start)
    if duration <= 1e-6:
        return np.array([clip_start], dtype=np.float32)
    count = min(config.MAX_CROP_KEYFRAMES, max(1, int(np.ceil(duration / max(config.KEYFRAME_INTERVAL_SEC, 1e-6))) + 1))
    return np.linspace(clip_start, clip_end, count, dtype=np.float32)
