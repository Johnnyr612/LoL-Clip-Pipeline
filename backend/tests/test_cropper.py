from __future__ import annotations

import numpy as np

from backend import config
from backend.cropper import blend_target, clamp_crop_x, compute_threat_sx, enforce_safe_zone, smooth_crop_values


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
