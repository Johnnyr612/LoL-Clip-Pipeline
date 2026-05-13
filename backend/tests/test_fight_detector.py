from __future__ import annotations

import cv2
import numpy as np

from backend import config
from backend.fight_detector import (
    DialogSegment,
    TrimResult,
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


def test_dialog_extension_pre():
    result = apply_dialog_extension(15, 25, 60, [DialogSegment("go", 11.5, 13.8)])
    assert result.clip_start == 11.0


def test_dialog_extension_post():
    result = apply_dialog_extension(15, 25, 60, [DialogSegment("done", 26.0, 28.5)])
    assert result.clip_end == 29.0


def test_max_duration_clamp():
    result = apply_dialog_extension(5, 20, 60, [DialogSegment("long", 20.5, 50.5)])
    assert result.clip_end - result.clip_start == config.MAX_CLIP_DURATION


def test_finish_on_kill_or_death_ends_after_enemy_healthbar_disappears():
    frames = np.zeros((38, 1080, 1920, 3), dtype=np.uint8)
    timestamps = np.arange(38, dtype=np.float32)
    for idx in range(34):
        cv2.rectangle(frames[idx], (800, 300), (900, 307), (210, 30, 30), thickness=-1)
        cv2.rectangle(frames[idx], (820, 360), (935, 367), (40, 210, 60), thickness=-1)
    for idx in range(34, 38):
        cv2.rectangle(frames[idx], (820, 360), (935, 367), (40, 210, 60), thickness=-1)

    trim = TrimResult(clip_start=0, clip_end=20, fight_start=1, fight_end=20, fight_duration=19, dialog_segments=[], flags=[])
    result = finish_on_kill_or_death(trim, frames, timestamps, 40)

    assert result.clip_end == 35.0
    assert "kill_event_detected" in result.flags


def test_finish_on_kill_or_death_ignores_early_healthbar_flicker():
    frames = np.zeros((40, 1080, 1920, 3), dtype=np.uint8)
    timestamps = np.arange(40, dtype=np.float32)
    cv2.rectangle(frames[1], (800, 300), (900, 307), (210, 30, 30), thickness=-1)
    for idx in range(40):
        cv2.rectangle(frames[idx], (820, 360), (935, 367), (40, 210, 60), thickness=-1)

    trim = TrimResult(clip_start=0, clip_end=20, fight_start=1, fight_end=20, fight_duration=19, dialog_segments=[], flags=[])
    result = finish_on_kill_or_death(trim, frames, timestamps, 40)

    assert result.clip_end == 38.0
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


def test_add_output_context_adds_two_seconds_without_moving_fight_markers():
    trim = TrimResult(clip_start=10, clip_end=38, fight_start=12, fight_end=36, fight_duration=24, dialog_segments=[], flags=[])
    result = add_output_context(trim, 60)

    assert result.clip_start == 8.0
    assert result.clip_end == 40.0
    assert result.fight_start == 12
    assert result.fight_end == 36
    assert "output_context_padding_applied" in result.flags


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
