from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter1d

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


def optical_flow_centroid(prev_frame: np.ndarray | None, frame: np.ndarray, player_sx: float) -> float:
    if prev_frame is None:
        return player_sx
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_RGB2GRAY)
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    magnitude = np.sqrt(flow[:, :, 0] ** 2 + flow[:, :, 1] ** 2)
    if float(magnitude.sum()) < config.LOW_FLOW_THRESHOLD:
        return player_sx
    threshold = np.percentile(magnitude, 85)
    mask = magnitude >= threshold
    if not np.any(mask):
        return player_sx
    xs = np.nonzero(mask)[1]
    weights = magnitude[mask]
    return float(np.average(xs, weights=weights))


def smooth_crop_values(values: Sequence[float], player_sx_values: Sequence[float], frame_w: int = 1920) -> np.ndarray:
    x = np.asarray(values, dtype=np.float32)
    if len(x) == 0:
        return x
    x = gaussian_filter1d(x, sigma=config.GAUSSIAN_SIGMA, mode="nearest")
    for idx, player_sx in enumerate(player_sx_values):
        x[idx] = enforce_player_framing(float(x[idx]), float(player_sx))
        x[idx] = clamp_crop_x(float(x[idx]), frame_w)
    for idx in range(len(x) - 1):
        delta = x[idx + 1] - x[idx]
        if abs(delta) > config.MAX_PAN_SPEED_PX:
            x[idx + 1] = x[idx] + np.sign(delta) * config.MAX_PAN_SPEED_PX
        x[idx + 1] = enforce_player_framing(float(x[idx + 1]), float(player_sx_values[idx + 1]))
        x[idx + 1] = clamp_crop_x(float(x[idx + 1]), frame_w)
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
        key_times = np.arange(clip_start, clip_end + 1e-6, config.KEYFRAME_INTERVAL_SEC)
        raw_x: list[float] = []
        player_sx_values: list[float] = []
        previous_frame: np.ndarray | None = None
        previous_player_pos = (0.5, 0.5)
        for timestamp in key_times:
            frame_idx = int(np.argmin(np.abs(timestamps - timestamp)))
            frame = frames[frame_idx]
            player_pos = player_positions[min(frame_idx, len(player_positions) - 1)] if player_positions else None
            if player_pos is None:
                player_pos = previous_player_pos
            previous_player_pos = player_pos
            detected_player_sx = (
                player_screen_x_positions[min(frame_idx, len(player_screen_x_positions) - 1)]
                if player_screen_x_positions
                else None
            )
            player_sx = float(detected_player_sx) if detected_player_sx is not None else map_pos_to_screen_hint(player_pos, (frame_w, frame_h))[0]
            detected_threat_sx = (
                threat_screen_x_positions[min(frame_idx, len(threat_screen_x_positions) - 1)]
                if threat_screen_x_positions
                else None
            )
            threat_sx = float(detected_threat_sx) if detected_threat_sx is not None else compute_threat_sx(player_pos, enemies, (frame_w, frame_h))
            flow_sx = optical_flow_centroid(previous_frame, frame, player_sx)
            target = blend_target(fight_type, player_sx, threat_sx, flow_sx)
            crop_x = enforce_player_framing(target - config.CROP_W / 2, player_sx)
            raw_x.append(clamp_crop_x(crop_x, frame_w))
            player_sx_values.append(player_sx)
            previous_frame = frame
        smoothed = smooth_crop_values(raw_x, player_sx_values, frame_w)
        return [CropKeyframe(float(t), int(x)) for t, x in zip(key_times, smoothed)]

    def interpolate_to_frames(self, keyframes: Sequence[CropKeyframe], source_timestamps: np.ndarray) -> list[tuple[int, int, int, int]]:
        if not keyframes:
            return []
        key_times = np.array([k.timestamp for k in keyframes], dtype=np.float32)
        key_x = np.array([k.crop_x for k in keyframes], dtype=np.float32)
        x_values = np.interp(source_timestamps, key_times, key_x)
        return [(int(round(x)), config.CROP_Y, config.CROP_W, config.CROP_H) for x in x_values]
