from __future__ import annotations

import cv2
import numpy as np

from backend.fight_detector import DialogSegment, TrimResult, apply_dialog_extension, boundaries_from_scores, finish_on_kill_or_death, merge_highlights


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
    result = apply_dialog_extension(5, 20, 60, [DialogSegment("long", 20.5, 42.0)])
    assert result.clip_end - result.clip_start == 35.0


def test_finish_on_kill_or_death_ends_after_enemy_healthbar_disappears():
    frames = np.zeros((8, 1080, 1920, 3), dtype=np.uint8)
    timestamps = np.arange(8, dtype=np.float32)
    for idx in range(5):
        cv2.rectangle(frames[idx], (800, 300), (900, 307), (210, 30, 30), thickness=-1)
        cv2.rectangle(frames[idx], (820, 360), (935, 367), (40, 210, 60), thickness=-1)
    for idx in range(5, 8):
        cv2.rectangle(frames[idx], (820, 360), (935, 367), (40, 210, 60), thickness=-1)

    trim = TrimResult(clip_start=0, clip_end=10, fight_start=1, fight_end=8, fight_duration=7, dialog_segments=[], flags=[])
    result = finish_on_kill_or_death(trim, frames, timestamps, 10)

    assert result.clip_end == 6.0
    assert "kill_event_detected" in result.flags


def test_finish_on_kill_or_death_preserves_overlapping_dialog():
    frames = np.zeros((8, 1080, 1920, 3), dtype=np.uint8)
    timestamps = np.arange(8, dtype=np.float32)
    for idx in range(5):
        cv2.rectangle(frames[idx], (800, 300), (900, 307), (210, 30, 30), thickness=-1)
        cv2.rectangle(frames[idx], (820, 360), (935, 367), (40, 210, 60), thickness=-1)
    for idx in range(5, 8):
        cv2.rectangle(frames[idx], (820, 360), (935, 367), (40, 210, 60), thickness=-1)

    trim = TrimResult(clip_start=0, clip_end=10, fight_start=1, fight_end=8, fight_duration=7, dialog_segments=[DialogSegment("wait", 5.5, 7.0)], flags=[])
    result = finish_on_kill_or_death(trim, frames, timestamps, 10)

    assert result.clip_end == 7.5
