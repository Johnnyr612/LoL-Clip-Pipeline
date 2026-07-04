from __future__ import annotations

import numpy as np

from backend import config
from backend.cropper import (
    AdaptiveCropper,
    CropKeyframe,
    avoid_minimap_ui,
    blend_target,
    clamp_crop_x,
    compute_threat_sx,
    enforce_center_preference,
    enforce_safe_zone,
    include_threat_in_crop,
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


def test_no_enemy_fallback():
    assert compute_threat_sx((0.5, 0.5), [], (1920, 1080)) == 960


def test_crop_values_are_not_smoothed():
    values = smooth_crop_values(np.arange(20) * 10, [500] * 20)
    assert values.shape == (20,)
    assert not np.isnan(values).any()
    assert len(set(values.tolist())) > 1


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


def test_minimap_white_box_fallback_centers_recorded_camera_view():
    frames = np.zeros((3, 1080, 1920, 3), dtype=np.uint8)
    timestamps = np.array([0.0, 0.5, 1.0], dtype=np.float32)
    keyframes = AdaptiveCropper().compute_keyframes(
        frames,
        timestamps,
        0.0,
        1.0,
        [(0.05, 0.5)] * 3,
        [],
        "1v1",
        [None, None, None],
        [None, None, None],
    )

    assert all(abs(960 - (keyframe.crop_x + config.CROP_W / 2)) <= config.PLAYER_CENTER_DEADZONE_PX for keyframe in keyframes)


def test_cropper_limits_trajectory_to_configured_keyframe_budget():
    frames = np.zeros((40, 1080, 1920, 3), dtype=np.uint8)
    timestamps = np.arange(40, dtype=np.float32)
    keyframes = AdaptiveCropper().compute_keyframes(
        frames,
        timestamps,
        0.0,
        38.0,
        [(0.5, 0.5)] * 40,
        [],
        "1v1",
        [960.0] * 40,
    )

    assert 1 <= len(keyframes) <= config.MAX_CROP_KEYFRAMES


def test_interpolate_to_frames_holds_stepwise_crops():
    cropper = AdaptiveCropper()
    keyframes = [CropKeyframe(0.0, 300), CropKeyframe(2.0, 700)]
    crops = cropper.interpolate_to_frames(keyframes, np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float32))

    assert crops[0][0] == crops[1][0] == 300
    assert crops[2][0] == crops[3][0] == 700


def test_minimap_only_enemy_does_not_pull_crop_off_player():
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
    assert all(abs(keyframe.crop_x - centered_crop_x) <= 1 for keyframe in keyframes)
    assert all(abs(960 - (keyframe.crop_x + config.CROP_W / 2)) <= config.PLAYER_CENTER_DEADZONE_PX for keyframe in keyframes)


def test_cropper_uses_persistent_visible_enemy_healthbar_as_threat_pull():
    frames = np.zeros((8, 1080, 1920, 3), dtype=np.uint8)
    timestamps = np.arange(8, dtype=np.float32)
    keyframes = AdaptiveCropper().compute_keyframes(
        frames,
        timestamps,
        0.0,
        7.0,
        [(0.5, 0.5)] * 8,
        [],
        "1v1",
        [960.0] * 8,
        [1250.0] * 8,
    )

    assert any(keyframe.crop_x > config.STATIC_CROP_X for keyframe in keyframes)



def test_hybrid_crop_ignores_bad_off_center_green_healthbar_matches():
    frames = np.zeros((8, 1080, 1920, 3), dtype=np.uint8)
    timestamps = np.arange(8, dtype=np.float32)
    bad_player_positions = [960.0, 1445.5, 481.0, 383.5, 1520.5, 606.0, 805.5, 1189.0]
    keyframes = AdaptiveCropper().compute_keyframes(
        frames,
        timestamps,
        0.0,
        7.0,
        [(0.5, 0.5)] * 8,
        [],
        "1v1",
        bad_player_positions,
        [None] * 8,
    )

    assert all(keyframe.crop_x == config.STATIC_CROP_X for keyframe in keyframes)


def test_hybrid_crop_uses_threat_side_relative_to_locked_center():
    frames = np.zeros((8, 1080, 1920, 3), dtype=np.uint8)
    timestamps = np.arange(8, dtype=np.float32)
    bad_player_positions = [960.0, 1445.5, 481.0, 383.5, 1520.5, 606.0, 805.5, 1189.0]
    keyframes = AdaptiveCropper().compute_keyframes(
        frames,
        timestamps,
        0.0,
        7.0,
        [(0.5, 0.5)] * 8,
        [],
        "1v1",
        bad_player_positions,
        [1250.0] * 8,
    )

    assert any(keyframe.crop_x > config.STATIC_CROP_X for keyframe in keyframes)
    assert all(keyframe.crop_x >= config.STATIC_CROP_X for keyframe in keyframes)

def test_threat_inclusion_keeps_green_health_player_center_priority():
    crop_x = include_threat_in_crop(960 - config.CROP_W / 2, 960, 1390)

    assert 0 <= 1390 - crop_x <= config.CROP_W
    assert abs(960 - (crop_x + config.CROP_W / 2)) <= config.PLAYER_CENTER_DEADZONE_PX


def test_far_enemy_cannot_pull_green_health_player_off_center():
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
        [1500.0, 1500.0, 1500.0],
    )

    assert all(abs(960 - (keyframe.crop_x + config.CROP_W / 2)) <= config.PLAYER_CENTER_DEADZONE_PX for keyframe in keyframes)


def test_minimap_ui_is_avoided_when_player_stays_safe():
    crop_x = avoid_minimap_ui(900, 960)

    assert crop_x + config.CROP_W <= 1920 * config.MINIMAP_CROP_X_PCT - config.MINIMAP_UI_AVOID_MARGIN_PX + 0.01
    assert config.PLAYER_SAFE_LEFT_PX <= 960 - crop_x <= config.PLAYER_SAFE_RIGHT_PX
