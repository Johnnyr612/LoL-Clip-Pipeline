from __future__ import annotations

from backend.fight_detector import DialogSegment, apply_dialog_extension, boundaries_from_scores, merge_highlights


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
