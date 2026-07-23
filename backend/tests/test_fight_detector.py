from __future__ import annotations

import cv2
import numpy as np

from backend import config
from backend.cropper import AdaptiveCropper
from backend.fight_detector import (
    DialogSegment,
    FightDetector,
    TrimResult,
    TrimSettings,
    add_output_context,
    apply_dialog_extension,
    boundaries_from_scores,
    estimate_visible_enemy_count,
    estimate_combat_screen_x_positions,
    finish_on_kill_or_death,
    merge_highlights,
)


def test_highlight_merge():
    assert merge_highlights([(5, 10), (11, 14)]) == [(5, 14)]


def test_highlight_no_merge():
    assert merge_highlights([(5, 10), (12, 16)]) == [(5, 10), (12, 16)]


def test_low_confidence_fallback():
    start, end, flags = boundaries_from_scores([0.0] * 60, 60)
    assert (start, end) == (25.0, 35.0)
    assert "low_confidence" in flags


def test_custom_trim_settings_control_fight_start_preroll():
    scores = [0.0] * 60
    scores[10:18] = [0.8] * 8

    default_start, _default_end, _default_flags = boundaries_from_scores(scores, 60)
    custom_start, _custom_end, _custom_flags = boundaries_from_scores(
        scores,
        60,
        TrimSettings(fight_start_preroll_sec=3.0),
    )

    assert default_start == 8.5
    assert custom_start == 7.0


def test_dialog_extension_pre():
    result = apply_dialog_extension(15, 25, 60, [DialogSegment("go", 11.5, 13.8)])
    assert result.clip_start == 11.0


def test_dialog_extension_post():
    result = apply_dialog_extension(15, 25, 60, [DialogSegment("done", 26.0, 28.5)])
    assert result.clip_end == 29.0


def test_max_duration_clamp():
    result = apply_dialog_extension(5, 20, 90, [DialogSegment("long", 20.5, 88.0)])
    assert result.clip_end - result.clip_start == config.MAX_CLIP_DURATION


def test_finish_on_kill_or_death_ends_on_confirmed_event_with_padding():
    frames = np.zeros((38, 1080, 1920, 3), dtype=np.uint8)
    timestamps = np.arange(38, dtype=np.float32)
    for idx in range(34):
        cv2.rectangle(frames[idx], (800, 300), (900, 307), (210, 30, 30), thickness=-1)
        cv2.rectangle(frames[idx], (820, 360), (935, 367), (40, 210, 60), thickness=-1)
    for idx in range(34, 38):
        cv2.rectangle(frames[idx], (820, 360), (935, 367), (40, 210, 60), thickness=-1)

    trim = TrimResult(clip_start=0, clip_end=20, fight_start=1, fight_end=20, fight_duration=19, dialog_segments=[], flags=[])
    result = finish_on_kill_or_death(trim, frames, timestamps, 40)

    assert result.clip_end == 37.0
    assert result.fight_end == 34.0
    assert "kill_event_detected" in result.flags
    assert "clip_end_on_kill_or_death" in result.flags


def test_finish_on_kill_or_death_does_not_force_conservative_target_window():
    frames = np.zeros((60, 1080, 1920, 3), dtype=np.uint8)
    timestamps = np.arange(60, dtype=np.float32)
    for idx in range(46):
        cv2.rectangle(frames[idx], (800, 300), (900, 307), (210, 30, 30), thickness=-1)
        cv2.rectangle(frames[idx], (820, 360), (935, 367), (40, 210, 60), thickness=-1)
    for idx in range(46, 60):
        cv2.rectangle(frames[idx], (820, 360), (935, 367), (40, 210, 60), thickness=-1)

    trim = TrimResult(clip_start=0, clip_end=20, fight_start=1, fight_end=20, fight_duration=19, dialog_segments=[], flags=[])
    result = finish_on_kill_or_death(trim, frames, timestamps, 60)

    assert result.clip_end == 49.0
    assert result.fight_end == 46.0
    assert "kill_event_detected" in result.flags
    assert "clip_end_on_kill_or_death" in result.flags
    assert "conservative_full_fight_trim" not in result.flags


def test_finish_on_kill_or_death_ignores_early_healthbar_flicker():
    frames = np.zeros((40, 1080, 1920, 3), dtype=np.uint8)
    timestamps = np.arange(40, dtype=np.float32)
    cv2.rectangle(frames[1], (800, 300), (900, 307), (210, 30, 30), thickness=-1)
    for idx in range(40):
        cv2.rectangle(frames[idx], (820, 360), (935, 367), (40, 210, 60), thickness=-1)

    trim = TrimResult(clip_start=0, clip_end=20, fight_start=1, fight_end=20, fight_duration=19, dialog_segments=[], flags=[])
    result = finish_on_kill_or_death(trim, frames, timestamps, 40)

    assert result.clip_end == 20.0
    assert "kill_event_detected" not in result.flags
    assert "combat_event_not_confirmed_extended_to_target" in result.flags


def test_finish_on_kill_or_death_preserves_overlapping_dialog():
    frames = np.zeros((38, 1080, 1920, 3), dtype=np.uint8)
    timestamps = np.arange(38, dtype=np.float32)
    for idx in range(34):
        cv2.rectangle(frames[idx], (800, 300), (900, 307), (210, 30, 30), thickness=-1)
        cv2.rectangle(frames[idx], (820, 360), (935, 367), (40, 210, 60), thickness=-1)
    for idx in range(34, 38):
        cv2.rectangle(frames[idx], (820, 360), (935, 367), (40, 210, 60), thickness=-1)

    trim = TrimResult(clip_start=0, clip_end=20, fight_start=1, fight_end=20, fight_duration=19, dialog_segments=[DialogSegment("wait", 34.5, 36.5)], flags=[])
    result = finish_on_kill_or_death(trim, frames, timestamps, 40)

    assert result.clip_end == 37.0
    assert "clip_end_on_kill_or_death" in result.flags


def test_add_output_context_adds_padding_without_moving_fight_markers():
    trim = TrimResult(clip_start=10, clip_end=38, fight_start=12, fight_end=36, fight_duration=24, dialog_segments=[], flags=[])
    result = add_output_context(trim, 60)

    assert result.clip_start == 9.5
    assert result.clip_end == 39.5
    assert result.fight_start == 12
    assert result.fight_end == 36
    assert "output_context_padding_applied" in result.flags
    assert "pre_fight_lead_capped" in result.flags


def test_custom_trim_settings_can_tighten_pre_fight_lead():
    trim = TrimResult(clip_start=10, clip_end=38, fight_start=12, fight_end=36, fight_duration=24, dialog_segments=[], flags=[])
    result = add_output_context(trim, 60, TrimSettings(output_context_padding_sec=1.5, max_pre_fight_lead_sec=1.0))

    assert result.clip_start == 11.0
    assert result.clip_end == 39.5
    assert "pre_fight_lead_capped" in result.flags


def test_estimate_visible_enemy_count():
    frames = np.zeros((4, 1080, 1920, 3), dtype=np.uint8)
    timestamps = np.arange(4, dtype=np.float32)
    for idx in range(4):
        cv2.rectangle(frames[idx], (820, 360), (935, 367), (40, 210, 60), thickness=-1)
        cv2.rectangle(frames[idx], (760, 300), (860, 307), (210, 30, 30), thickness=-1)
        cv2.rectangle(frames[idx], (900, 300), (1000, 307), (210, 30, 30), thickness=-1)

    assert estimate_visible_enemy_count(frames, timestamps, 0, 3) == 2


def test_estimate_combat_screen_x_positions():
    frames = np.zeros((1, 1080, 1920, 3), dtype=np.uint8)
    cv2.rectangle(frames[0], (820, 360), (940, 367), (40, 210, 60), thickness=-1)
    cv2.rectangle(frames[0], (1000, 300), (1100, 307), (210, 30, 30), thickness=-1)

    player_positions, threat_positions = estimate_combat_screen_x_positions(frames)

    assert player_positions == [880.0]
    assert threat_positions == [1050.0]


def test_estimate_combat_screen_x_positions_uses_nearest_red_bar():
    frames = np.zeros((1, 1080, 1920, 3), dtype=np.uint8)
    cv2.rectangle(frames[0], (820, 360), (940, 367), (40, 210, 60), thickness=-1)
    cv2.rectangle(frames[0], (1000, 300), (1100, 307), (210, 30, 30), thickness=-1)
    cv2.rectangle(frames[0], (1280, 300), (1380, 307), (210, 30, 30), thickness=-1)

    _player_positions, threat_positions = estimate_combat_screen_x_positions(frames)

    assert threat_positions == [1050.0]


def test_estimate_combat_screen_x_positions_ignores_left_hud_bars():
    frames = np.zeros((1, 1080, 1920, 3), dtype=np.uint8)
    cv2.rectangle(frames[0], (190, 220), (300, 227), (40, 210, 60), thickness=-1)
    cv2.rectangle(frames[0], (820, 360), (940, 367), (40, 210, 60), thickness=-1)
    cv2.rectangle(frames[0], (1000, 300), (1100, 307), (210, 30, 30), thickness=-1)

    player_positions, threat_positions = estimate_combat_screen_x_positions(frames)

    assert player_positions == [880.0]
    assert threat_positions == [1050.0]

def test_minion_red_health_bars_do_not_count_as_visible_enemies():
    frames = np.zeros((4, 1080, 1920, 3), dtype=np.uint8)
    timestamps = np.arange(4, dtype=np.float32)
    for idx in range(4):
        cv2.rectangle(frames[idx], (820, 360), (940, 367), (40, 210, 60), thickness=-1)
        cv2.rectangle(frames[idx], (1000, 300), (1060, 307), (210, 30, 30), thickness=-1)
        cv2.rectangle(frames[idx], (1120, 330), (1180, 337), (210, 30, 30), thickness=-1)

    assert estimate_visible_enemy_count(frames, timestamps, 0, 3) is None


def test_minion_red_health_bars_do_not_pull_crop_threat_position():
    frames = np.zeros((1, 1080, 1920, 3), dtype=np.uint8)
    cv2.rectangle(frames[0], (820, 360), (940, 367), (40, 210, 60), thickness=-1)
    cv2.rectangle(frames[0], (1000, 300), (1060, 307), (210, 30, 30), thickness=-1)
    cv2.rectangle(frames[0], (1120, 330), (1180, 337), (210, 30, 30), thickness=-1)

    player_positions, threat_positions = estimate_combat_screen_x_positions(frames)

    assert player_positions == [880.0]
    assert threat_positions == [None]


def test_champion_red_health_bar_still_pulls_crop_threat_position():
    frames = np.zeros((1, 1080, 1920, 3), dtype=np.uint8)
    cv2.rectangle(frames[0], (820, 360), (940, 367), (40, 210, 60), thickness=-1)
    cv2.rectangle(frames[0], (1000, 300), (1100, 307), (210, 30, 30), thickness=-1)

    player_positions, threat_positions = estimate_combat_screen_x_positions(frames)

    assert player_positions == [880.0]
    assert threat_positions == [1050.0]

def test_hybrid_crop_holds_center_when_only_minion_red_bars_are_visible():
    frames = np.zeros((8, 1080, 1920, 3), dtype=np.uint8)
    timestamps = np.arange(8, dtype=np.float32)
    for idx in range(8):
        cv2.rectangle(frames[idx], (900, 360), (1020, 367), (40, 210, 60), thickness=-1)
        cv2.rectangle(frames[idx], (1220, 300), (1280, 307), (210, 30, 30), thickness=-1)

    player_positions, threat_positions = estimate_combat_screen_x_positions(frames)
    keyframes = AdaptiveCropper().compute_keyframes(
        frames,
        timestamps,
        0.0,
        7.0,
        [(0.5, 0.5)] * 8,
        [],
        "1v1",
        player_positions,
        threat_positions,
    )

    assert threat_positions == [None] * 8
    assert all(keyframe.crop_x == config.STATIC_CROP_X for keyframe in keyframes)


def test_hybrid_crop_can_shift_after_persistent_champion_red_bar():
    frames = np.zeros((8, 1080, 1920, 3), dtype=np.uint8)
    timestamps = np.arange(8, dtype=np.float32)
    for idx in range(8):
        cv2.rectangle(frames[idx], (900, 360), (1020, 367), (40, 210, 60), thickness=-1)
        cv2.rectangle(frames[idx], (1200, 300), (1320, 307), (210, 30, 30), thickness=-1)

    player_positions, threat_positions = estimate_combat_screen_x_positions(frames)
    keyframes = AdaptiveCropper().compute_keyframes(
        frames,
        timestamps,
        0.0,
        7.0,
        [(0.5, 0.5)] * 8,
        [],
        "1v1",
        player_positions,
        threat_positions,
    )

    assert threat_positions == [1260.0] * 8
    assert any(keyframe.crop_x > config.STATIC_CROP_X for keyframe in keyframes)

def test_score_windows_uses_green_and_red_healthbar_engagement(monkeypatch):
    frames = np.zeros((20, 1080, 1920, 3), dtype=np.uint8)
    timestamps = np.arange(20, dtype=np.float32)
    for idx in range(20):
        cv2.rectangle(frames[idx], (820, 360), (940, 367), (40, 210, 60), thickness=-1)
        cv2.rectangle(frames[idx], (1000, 300), (1100, 307), (210, 30, 30), thickness=-1)

    detector = FightDetector()

    def fail_videomae():
        raise RuntimeError("skip model")

    monkeypatch.setattr(detector, "_load_videomae", fail_videomae)
    scores = detector.score_windows(frames, timestamps)

    assert scores
    assert max(scores) >= config.FIGHT_CONFIDENCE_THRESHOLD
