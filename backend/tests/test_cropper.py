from __future__ import annotations

import numpy as np

from backend import config
from backend.cropper import (
    AdaptiveCropper,
    CropKeyframe,
    CropSettings,
    avoid_minimap_ui,
    blend_target,
    clamp_crop_x,
    combat_focus_crop_x,
    dynamic_rule_of_thirds_crop_x,
    enforce_center_preference,
    enforce_safe_zone,
    include_threat_in_crop,
    player_anchor_x,
    rule_of_thirds_crop_x,
    smooth_crop_values,
)
from backend.minimap_detector import ChampionResult


def _right_thirds_crop_x(player_sx: float = 960.0) -> int:
    threat_sx = player_sx + max(float(config.DYNAMIC_THREAT_SIDE_TRIGGER_PX + 1), float(config.CROP_W) * 0.35)
    return clamp_crop_x(rule_of_thirds_crop_x(player_sx, threat_sx))


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


def test_rule_of_thirds_places_player_opposite_clear_threat_side():
    right_threat_anchor = config.CROP_W / 3 - config.PLAYER_THIRDS_LOOK_ROOM_PX
    left_threat_anchor = config.CROP_W * 2 / 3 + config.PLAYER_THIRDS_LOOK_ROOM_PX

    assert player_anchor_x(960, 1250) == right_threat_anchor
    assert rule_of_thirds_crop_x(960, 1250) == 960 - right_threat_anchor
    assert player_anchor_x(960, 650) == left_threat_anchor
    assert rule_of_thirds_crop_x(960, 650) == 960 - left_threat_anchor


def test_rule_of_thirds_falls_back_to_center_without_clear_threat():
    assert player_anchor_x(960, None) == config.CROP_W / 2
    assert player_anchor_x(960, 1000) == config.CROP_W / 2


def test_clamp_frame_boundary():
    assert clamp_crop_x(-20) == 0


def test_crop_values_are_not_smoothed():
    values = smooth_crop_values(np.arange(20) * 10, [500] * 20)
    assert values.shape == (20,)
    assert not np.isnan(values).any()
    assert len(set(values.tolist())) > 1


def test_cropper_starts_dynamic_view_on_rule_of_thirds_without_enemy_threat():
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

    assert all(keyframe.crop_x == _right_thirds_crop_x(960.0) for keyframe in keyframes)
    assert all(abs((960 - keyframe.crop_x) - player_anchor_x(960, 1250)) <= config.PLAYER_THIRDS_DEADZONE_PX for keyframe in keyframes)


def test_cropper_ignores_minimap_hint_while_using_default_thirds_view():
    frames = np.zeros((3, 1080, 1920, 3), dtype=np.uint8)
    timestamps = np.array([0.0, 0.5, 1.0], dtype=np.float32)
    keyframes = AdaptiveCropper().compute_keyframes(
        frames,
        timestamps,
        0.0,
        1.0,
        [(0.05, 0.5), (0.95, 0.5), (0.05, 0.5)],
        [],
        "1v1",
        [None, None, None],
        [None, None, None],
    )

    assert all(keyframe.crop_x == _right_thirds_crop_x(960.0) for keyframe in keyframes)


def test_dynamic_crop_does_not_follow_green_bar_jitter_without_enemy():
    frames = np.zeros((6, 1080, 1920, 3), dtype=np.uint8)
    timestamps = np.arange(6, dtype=np.float32)
    keyframes = AdaptiveCropper().compute_keyframes(
        frames,
        timestamps,
        0.0,
        5.0,
        [(0.5, 0.5)] * 6,
        [],
        "1v1",
        [910.0, 960.0, 900.0, 980.0, 915.0, 965.0],
        [None] * 6,
        CropSettings(mode="dynamic", transition="cut"),
    )

    assert all(keyframe.crop_x != config.STATIC_CROP_X for keyframe in keyframes)
    assert max(keyframe.crop_x for keyframe in keyframes) - min(keyframe.crop_x for keyframe in keyframes) <= 90


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


def test_interpolate_to_frames_can_slide_between_crops():
    cropper = AdaptiveCropper()
    keyframes = [CropKeyframe(0.0, 300), CropKeyframe(2.0, 700)]
    crops = cropper.interpolate_to_frames(
        keyframes,
        np.array([0.0, 1.0, 2.0], dtype=np.float32),
        CropSettings(transition="pan"),
    )

    assert crops[0][0] == 300
    assert crops[1][0] == 500
    assert crops[2][0] == 700


def test_minimap_only_enemy_does_not_move_dynamic_crop():
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

    assert all(keyframe.crop_x == _right_thirds_crop_x(960.0) for keyframe in keyframes)
    assert all(abs((960 - keyframe.crop_x) - player_anchor_x(960, 1250)) <= config.PLAYER_THIRDS_DEADZONE_PX for keyframe in keyframes)


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
        [1390.0] * 8,
    )

    assert any(keyframe.crop_x > config.STATIC_CROP_X for keyframe in keyframes)



def test_dynamic_crop_ignores_bad_off_center_green_healthbar_matches():
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

    assert all(keyframe.crop_x != config.STATIC_CROP_X for keyframe in keyframes)
    assert max(keyframe.crop_x for keyframe in keyframes) - min(keyframe.crop_x for keyframe in keyframes) <= 100


def test_dynamic_crop_uses_threat_side_relative_to_locked_center():
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


def test_dynamic_crop_extends_farther_to_keep_visible_enemy_in_frame():
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
        [1390.0] * 8,
    )

    assert any(keyframe.crop_x > config.STATIC_CROP_X for keyframe in keyframes)
    assert all(config.PLAYER_SAFE_LEFT_PX <= 960 - keyframe.crop_x <= config.PLAYER_SAFE_RIGHT_PX for keyframe in keyframes)


def test_dynamic_crop_waits_for_persistent_left_enemy():
    frames = np.zeros((4, 1080, 1920, 3), dtype=np.uint8)
    timestamps = np.arange(4, dtype=np.float32)
    keyframes = AdaptiveCropper().compute_keyframes(
        frames,
        timestamps,
        0.0,
        3.0,
        [(0.5, 0.5)] * 4,
        [],
        "1v1",
        [960.0] * 4,
        [500.0] * 4,
        CropSettings(mode="dynamic", transition="cut"),
    )

    assert keyframes[0].crop_x < config.STATIC_CROP_X
    assert keyframes[1].crop_x < config.STATIC_CROP_X
    assert keyframes[2].crop_x < config.STATIC_CROP_X
    assert keyframes[-1].crop_x < config.STATIC_CROP_X
    assert 500 - keyframes[-1].crop_x >= config.THREAT_FRAME_MARGIN_PX
    assert all(config.PLAYER_SAFE_LEFT_PX <= 960 - keyframe.crop_x <= config.PLAYER_SAFE_RIGHT_PX for keyframe in keyframes)


def test_dynamic_crop_can_smooth_slide_toward_enemy():
    frames = np.zeros((4, 1080, 1920, 3), dtype=np.uint8)
    timestamps = np.arange(4, dtype=np.float32)
    keyframes = AdaptiveCropper().compute_keyframes(
        frames,
        timestamps,
        0.0,
        3.0,
        [(0.5, 0.5)] * 4,
        [],
        "1v1",
        [960.0] * 4,
        [None, 500.0, 500.0, 500.0],
        CropSettings(mode="dynamic", transition="pan"),
    )

    assert keyframes[0].crop_x < config.STATIC_CROP_X
    assert keyframes[-1].crop_x < config.STATIC_CROP_X
    assert keyframes[-1].crop_x >= keyframes[0].crop_x - config.MAX_PAN_SPEED_PX_PER_SEC * 3


def test_threat_inclusion_keeps_green_health_player_near_thirds_anchor():
    crop_x = include_threat_in_crop(960 - config.CROP_W / 2, 960, 1390)

    assert 0 <= 1390 - crop_x <= config.CROP_W
    assert abs((960 - crop_x) - player_anchor_x(960, 1390)) <= config.PLAYER_THIRDS_DEADZONE_PX


def test_center_composition_keeps_green_health_player_center_priority(monkeypatch):
    monkeypatch.setattr(config, "PLAYER_COMPOSITION", "center")

    crop_x = include_threat_in_crop(960 - config.CROP_W / 2, 960, 1390)

    assert 0 <= 1390 - crop_x <= config.CROP_W
    assert abs(960 - (crop_x + config.CROP_W / 2)) <= config.PLAYER_CENTER_DEADZONE_PX


def test_dynamic_edge_enemy_pulls_view_while_player_stays_safe():
    frames = np.zeros((4, 1080, 1920, 3), dtype=np.uint8)
    timestamps = np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float32)
    keyframes = AdaptiveCropper().compute_keyframes(
        frames,
        timestamps,
        0.0,
        1.0,
        [(0.5, 0.5)] * 4,
        [],
        "1v1",
        [960.0] * 4,
        [1390.0] * 4,
        CropSettings(mode="dynamic"),
    )

    assert keyframes[0].crop_x > config.STATIC_CROP_X
    assert keyframes[-1].crop_x > config.STATIC_CROP_X
    assert 1390 - keyframes[-1].crop_x <= config.CROP_W - config.THREAT_FRAME_MARGIN_PX
    assert all(config.PLAYER_SAFE_LEFT_PX <= 960 - keyframe.crop_x <= config.PLAYER_SAFE_RIGHT_PX for keyframe in keyframes)


def test_dynamic_crop_reframes_for_modest_visible_enemy_offset():
    frames = np.zeros((4, 1080, 1920, 3), dtype=np.uint8)
    timestamps = np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float32)
    keyframes = AdaptiveCropper().compute_keyframes(
        frames,
        timestamps,
        0.0,
        3.0,
        [(0.5, 0.5)] * 4,
        [],
        "1v1",
        [960.0] * 4,
        [1120.0] * 4,
        CropSettings(mode="dynamic", transition="cut"),
    )

    assert keyframes[0].crop_x > config.STATIC_CROP_X
    assert keyframes[-1].crop_x > config.STATIC_CROP_X
    assert config.THREAT_FRAME_MARGIN_PX <= 1120 - keyframes[-1].crop_x <= config.CROP_W - config.THREAT_FRAME_MARGIN_PX
    assert all(config.PLAYER_SAFE_LEFT_PX <= 960 - keyframe.crop_x <= config.PLAYER_SAFE_RIGHT_PX for keyframe in keyframes)


def test_dynamic_crop_uses_locked_center_for_threat_side_when_green_bar_drifts():
    frames = np.zeros((4, 1080, 1920, 3), dtype=np.uint8)
    timestamps = np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float32)
    keyframes = AdaptiveCropper().compute_keyframes(
        frames,
        timestamps,
        0.0,
        3.0,
        [(0.5, 0.5)] * 4,
        [],
        "1v1",
        [1029.0] * 4,
        [1071.0] * 4,
        CropSettings(mode="dynamic", transition="cut"),
    )

    assert keyframes[0].crop_x > config.STATIC_CROP_X
    assert keyframes[-1].crop_x > config.STATIC_CROP_X
    assert config.THREAT_FRAME_MARGIN_PX <= 1071 - keyframes[-1].crop_x <= config.CROP_W - config.THREAT_FRAME_MARGIN_PX


def test_dynamic_crop_switches_right_when_enemy_is_right_of_off_center_player():
    frames = np.zeros((6, 1080, 1920, 3), dtype=np.uint8)
    timestamps = np.arange(6, dtype=np.float32)
    keyframes = AdaptiveCropper().compute_keyframes(
        frames,
        timestamps,
        0.0,
        5.0,
        [(0.5, 0.5)] * 6,
        [],
        "1v1",
        [880.0] * 6,
        [650.0, 650.0, 1010.0, 1010.0, 1010.0, 1010.0],
        CropSettings(mode="dynamic", transition="cut"),
    )

    assert keyframes[1].crop_x < config.STATIC_CROP_X
    assert keyframes[2].crop_x > config.STATIC_CROP_X
    assert all(keyframe.crop_x > config.STATIC_CROP_X for keyframe in keyframes[2:])
    assert all(config.THREAT_FRAME_MARGIN_PX <= 1010 - keyframe.crop_x <= config.CROP_W - config.THREAT_FRAME_MARGIN_PX for keyframe in keyframes[2:])


def test_dynamic_crop_seeds_opening_from_fight_start_threat():
    frames = np.zeros((6, 1080, 1920, 3), dtype=np.uint8)
    timestamps = np.arange(6, dtype=np.float32)
    keyframes = AdaptiveCropper().compute_keyframes(
        frames,
        timestamps,
        0.0,
        5.0,
        [(0.5, 0.5)] * 6,
        [],
        "1v1",
        [880.0] * 6,
        [650.0, 650.0, 1010.0, 1010.0, 1010.0, 1010.0],
        CropSettings(mode="dynamic", transition="pan"),
        focus_start=2.0,
    )

    assert keyframes[0].crop_x > config.STATIC_CROP_X
    assert all(keyframe.crop_x > config.STATIC_CROP_X for keyframe in keyframes[:3])
    assert config.THREAT_FRAME_MARGIN_PX <= 1010 - keyframes[0].crop_x <= config.CROP_W - config.THREAT_FRAME_MARGIN_PX


def test_dynamic_crop_returns_to_default_thirds_when_enemy_signal_disappears():
    frames = np.zeros((5, 1080, 1920, 3), dtype=np.uint8)
    timestamps = np.arange(5, dtype=np.float32)
    keyframes = AdaptiveCropper().compute_keyframes(
        frames,
        timestamps,
        0.0,
        4.0,
        [(0.5, 0.5)] * 5,
        [],
        "1v1",
        [960.0] * 5,
        [1390.0, 1390.0, 1390.0, 1390.0, None],
        CropSettings(mode="dynamic", transition="cut"),
    )

    assert keyframes[0].crop_x > config.STATIC_CROP_X
    assert keyframes[3].crop_x > config.STATIC_CROP_X
    assert keyframes[-1].crop_x == _right_thirds_crop_x(960.0)


def test_dynamic_crop_holds_shifted_view_while_enemy_remains_visible():
    frames = np.zeros((7, 1080, 1920, 3), dtype=np.uint8)
    timestamps = np.arange(7, dtype=np.float32)
    keyframes = AdaptiveCropper().compute_keyframes(
        frames,
        timestamps,
        0.0,
        6.0,
        [(0.5, 0.5)] * 7,
        [],
        "1v1",
        [960.0] * 7,
        [1390.0, 1390.0, 1390.0, 1390.0, 1120.0, 1120.0, None],
        CropSettings(mode="dynamic", transition="cut"),
    )

    assert keyframes[3].crop_x > config.STATIC_CROP_X
    assert keyframes[4].crop_x == keyframes[3].crop_x
    assert keyframes[5].crop_x == keyframes[3].crop_x
    assert keyframes[-1].crop_x == _right_thirds_crop_x(960.0)


def test_dynamic_crop_limits_enemy_side_changes_to_three():
    frames = np.zeros((18, 1080, 1920, 3), dtype=np.uint8)
    timestamps = np.arange(18, dtype=np.float32)
    threat_positions = [
        1390.0, 1390.0, 1390.0, 1390.0,
        None,
        500.0, 500.0, 500.0,
        None,
        1390.0, 1390.0, 1390.0,
        None,
        500.0, 500.0, 500.0,
        None, None,
    ]
    keyframes = AdaptiveCropper().compute_keyframes(
        frames,
        timestamps,
        0.0,
        17.0,
        [(0.5, 0.5)] * 18,
        [],
        "1v1",
        [960.0] * 18,
        threat_positions,
        CropSettings(mode="dynamic", transition="cut"),
    )

    side_runs = 0
    current_side = 0
    for keyframe in keyframes:
        if keyframe.crop_x < config.STATIC_CROP_X:
            side = -1
        elif keyframe.crop_x > config.STATIC_CROP_X:
            side = 1
        else:
            side = 0
        if side != 0 and side != current_side:
            side_runs += 1
        current_side = side

    assert side_runs <= config.DYNAMIC_MAX_VIEW_CHANGES + 1


def test_dynamic_crop_does_not_shift_for_enemy_that_cannot_fit():
    crop_x = dynamic_rule_of_thirds_crop_x(960, 1800)

    assert crop_x == config.STATIC_CROP_X


def test_dynamic_crop_uses_rule_of_thirds_when_enemy_fits():
    crop_x = dynamic_rule_of_thirds_crop_x(960, 1390)

    assert abs((960 - crop_x) - player_anchor_x(960, 1390)) <= config.PLAYER_THIRDS_DEADZONE_PX
    assert config.THREAT_FRAME_MARGIN_PX <= 1390 - crop_x <= config.CROP_W - config.THREAT_FRAME_MARGIN_PX


def test_combat_focus_centers_player_when_threat_already_fits():
    crop_x = combat_focus_crop_x(960, 1250)

    assert abs(960 - (crop_x + config.CROP_W / 2)) <= config.PLAYER_CENTER_DEADZONE_PX
    assert config.THREAT_FRAME_MARGIN_PX <= 1250 - crop_x <= config.CROP_W - config.THREAT_FRAME_MARGIN_PX


def test_combat_focus_shifts_only_enough_for_edge_threat():
    crop_x = combat_focus_crop_x(960, 1480)

    assert crop_x > config.STATIC_CROP_X
    assert config.PLAYER_SAFE_LEFT_PX <= 960 - crop_x <= config.PLAYER_SAFE_RIGHT_PX
    assert config.THREAT_FRAME_MARGIN_PX <= 1480 - crop_x <= config.CROP_W - config.THREAT_FRAME_MARGIN_PX


def test_minimap_ui_is_avoided_when_player_stays_safe():
    crop_x = avoid_minimap_ui(900, 960)

    assert crop_x + config.CROP_W <= 1920 * config.MINIMAP_CROP_X_PCT - config.MINIMAP_UI_AVOID_MARGIN_PX + 0.01
    assert config.PLAYER_SAFE_LEFT_PX <= 960 - crop_x <= config.PLAYER_SAFE_RIGHT_PX
