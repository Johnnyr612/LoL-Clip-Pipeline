from __future__ import annotations

import numpy as np

from backend import config
from backend.cropper import (
    AdaptiveCropper,
    blend_target,
    clamp_crop_x,
    compute_threat_sx,
    enforce_center_preference,
    enforce_safe_zone,
    smooth_crop_values,
)
from backend.minimap_detector import ChampionResult


def test_blend_1v1():
    player, threat, flow = config.BLEND_1V1
    assert blend_target("1v1", 400, 700, 600) == player * 400 + threat * 700 + flow * 600


def test_blend_1vn():
    player, threat, flow = config.BLEND_1VN
    assert blend_target("1v3", 400, 700, 600) == player * 400 + threat * 700 + flow * 600


def test_safe_zone_left():
    crop_x = enforce_safe_zone(0, 50)
    assert 50 - crop_x >= config.PLAYER_SAFE_LEFT_PX


def test_safe_zone_right():
    crop_x = enforce_safe_zone(1110, 1870)
    assert 1870 - crop_x <= config.PLAYER_SAFE_RIGHT_PX


def test_center_preference_keeps_player_near_middle():
    crop_x = enforce_center_preference(0, 700)
    player_offset = abs(700 - (crop_x + config.CROP_W / 2))
    assert player_offset <= config.PLAYER_CENTER_DEADZONE_PX


def test_clamp_frame_boundary():
    assert clamp_crop_x(-20) == 0


def test_velocity_clamp():
    values = smooth_crop_values([0, 200, 400], [400, 400, 400])
    assert np.all(np.abs(np.diff(values)) <= config.MAX_PAN_SPEED_PX)


def test_no_enemy_fallback():
    assert compute_threat_sx((0.5, 0.5), [], (1920, 1080)) == 960


def test_gaussian_smooth_shape():
    values = smooth_crop_values(np.arange(20) * 10, [500] * 20)
    assert values.shape == (20,)
    assert not np.isnan(values).any()


def test_cropper_uses_green_healthbar_screen_position_over_minimap_hint():
    frames = np.zeros((3, 1080, 1920, 3), dtype=np.uint8)
    timestamps = np.array([0.0, 0.5, 1.0], dtype=np.float32)
    keyframes = AdaptiveCropper().compute_keyframes(
        frames,
        timestamps,
        0.0,
        1.0,
        [(0.1, 0.5)] * 3,
        [],
        "1v1",
        [960.0, 960.0, 960.0],
    )

    for keyframe in keyframes:
        assert abs(960 - (keyframe.crop_x + config.CROP_W / 2)) <= config.PLAYER_CENTER_DEADZONE_PX


def test_enemy_can_still_pull_crop_while_player_stays_centered():
    frames = np.zeros((3, 1080, 1920, 3), dtype=np.uint8)
    timestamps = np.array([0.0, 0.5, 1.0], dtype=np.float32)
    enemy = ChampionResult("Enemy", 1.0, "enemy", (0.85, 0.5))
    keyframes = AdaptiveCropper().compute_keyframes(
        frames,
        timestamps,
        0.0,
        1.0,
        [(0.5, 0.5)] * 3,
        [enemy],
        "1v1",
        [960.0, 960.0, 960.0],
    )

    centered_crop_x = 960 - config.CROP_W / 2
    assert any(keyframe.crop_x > centered_crop_x for keyframe in keyframes)
    assert all(abs(960 - (keyframe.crop_x + config.CROP_W / 2)) <= config.PLAYER_CENTER_DEADZONE_PX for keyframe in keyframes)


def test_cropper_uses_visible_enemy_healthbar_as_threat_pull():
    frames = np.zeros((3, 1080, 1920, 3), dtype=np.uint8)
    timestamps = np.array([0.0, 0.5, 1.0], dtype=np.float32)
    keyframes = AdaptiveCropper().compute_keyframes(
        frames,
        timestamps,
        0.0,
        1.0,
        [(0.5, 0.5)] * 3,
        [],
        "1v1",
        [960.0, 960.0, 960.0],
        [1250.0, 1250.0, 1250.0],
    )

    centered_crop_x = 960 - config.CROP_W / 2
    assert any(keyframe.crop_x > centered_crop_x for keyframe in keyframes)
