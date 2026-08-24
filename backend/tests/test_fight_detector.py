from __future__ import annotations

import cv2
import numpy as np
import pytest

from backend import config
from backend.cropper import AdaptiveCropper, clamp_crop_x, rule_of_thirds_crop_x
from backend.fight_detector import (
    DialogSegment,
    FightDetector,
    HighlightEditorError,
    TrimResult,
    TrimSettings,
    add_output_context,
    apply_highlight_trim_settings,
    apply_dialog_extension,
    boundaries_from_scores,
    estimate_combat_screen_x_position_tracks,
    estimate_visible_enemy_count,
    estimate_combat_screen_x_positions,
    finish_on_kill_or_death,
    merge_highlights,
    _best_include_span,
    _camera_threat_bars,
    _enforce_highlight_duration,
)

THICK_BAR_H = 10
THIN_BAR_H = 4


def _draw_bar(frame: np.ndarray, x1: int, y1: int, x2: int, height: int, color: tuple[int, int, int]) -> None:
    cv2.rectangle(frame, (x1, y1), (x2, y1 + height - 1), color, thickness=-1)


def _draw_player_enemy_bars(frame: np.ndarray) -> None:
    _draw_bar(frame, 820, 360, 940, THICK_BAR_H, (40, 210, 60))
    _draw_enemy_champion_bar(frame, 1000, 300, 1060)


def _draw_objective_health_number(frame: np.ndarray, x: int, y: int, text: str = "2032") -> None:
    cv2.putText(
        frame,
        text,
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (235, 220, 150),
        1,
        cv2.LINE_AA,
    )


def _draw_champion_level_badge(frame: np.ndarray, x: int, y: int, level: str = "11") -> None:
    cv2.rectangle(frame, (x, y - 12), (x + 24, y + 16), (10, 18, 24), thickness=-1)
    cv2.putText(
        frame,
        level,
        (x + 4, y + 8),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        (225, 230, 235),
        1,
        cv2.LINE_AA,
    )


def _draw_enemy_champion_bar(frame: np.ndarray, x1: int, y1: int, x2: int, height: int = THICK_BAR_H) -> None:
    _draw_champion_level_badge(frame, max(0, x1 - 32), y1 + 5)
    _draw_bar(frame, x1, y1, x2, height, (210, 30, 30))


def _right_thirds_crop_x(player_sx: float = 960.0) -> int:
    threat_sx = player_sx + max(float(config.DYNAMIC_THREAT_SIDE_TRIGGER_PX + 1), float(config.CROP_W) * 0.35)
    return clamp_crop_x(rule_of_thirds_crop_x(player_sx, threat_sx))


def test_highlight_merge():
    assert merge_highlights([(5, 10), (11, 14)]) == [(5, 14)]


def test_highlight_no_merge():
    assert merge_highlights([(5, 10), (12, 16)]) == [(5, 10), (12, 16)]


def test_low_confidence_default_window():
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

    assert default_start == 9.2
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
    assert "clip_end_extended_to_combat_event" in result.flags


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
    assert "clip_end_extended_to_combat_event" in result.flags
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
    assert "clip_end_extended_to_combat_event" in result.flags


def test_finish_on_kill_or_death_does_not_cut_before_model_fight_end():
    frames = np.zeros((50, 1080, 1920, 3), dtype=np.uint8)
    timestamps = np.arange(50, dtype=np.float32)
    for idx in range(24):
        cv2.rectangle(frames[idx], (800, 300), (900, 307), (210, 30, 30), thickness=-1)
        cv2.rectangle(frames[idx], (820, 360), (935, 367), (40, 210, 60), thickness=-1)
    for idx in range(24, 50):
        cv2.rectangle(frames[idx], (820, 360), (935, 367), (40, 210, 60), thickness=-1)

    trim = TrimResult(clip_start=0, clip_end=35, fight_start=0, fight_end=35, fight_duration=35, dialog_segments=[], flags=[])
    result = finish_on_kill_or_death(trim, frames, timestamps, 50)

    assert result.clip_end == 35.0
    assert "kill_event_detected" in result.flags
    assert "combat_event_before_model_end_ignored_for_trim" in result.flags


def test_add_output_context_adds_padding_without_moving_fight_markers():
    trim = TrimResult(clip_start=10, clip_end=38, fight_start=12, fight_end=36, fight_duration=24, dialog_segments=[], flags=[])
    result = add_output_context(trim, 60)

    assert result.clip_start == 10.8
    assert result.clip_end == 38.5
    assert result.fight_start == 12
    assert result.fight_end == 36
    assert "output_context_padding_applied" in result.flags
    assert "pre_fight_lead_capped" in result.flags


def test_highlight_span_helpers_choose_model_include_run():
    values = np.array([0.1, 0.7, 0.8, 0.2, 0.65, 0.9, 0.88, 0.1], dtype=np.float32)

    assert _best_include_span(values, 0.5) == (4.0, 7.0)
    assert _enforce_highlight_duration(4.0, 7.0, 8)[0] <= 4.0


def test_custom_trim_settings_can_tighten_pre_fight_lead():
    trim = TrimResult(clip_start=10, clip_end=38, fight_start=12, fight_end=36, fight_duration=24, dialog_segments=[], flags=[])
    result = add_output_context(trim, 60, TrimSettings(output_context_padding_sec=1.5, max_pre_fight_lead_sec=1.0))

    assert result.clip_start == 11.0
    assert result.clip_end == 39.5
    assert "pre_fight_lead_capped" in result.flags


def test_highlight_trim_settings_use_current_balanced_defaults():
    frames = np.zeros((60, 1080, 1920, 3), dtype=np.uint8)
    timestamps = np.arange(60, dtype=np.float32)
    trim = TrimResult(clip_start=10, clip_end=38, fight_start=12, fight_end=36, fight_duration=24, dialog_segments=[], flags=["highlight_editor_model"])

    tight = apply_highlight_trim_settings(
        trim,
        frames,
        timestamps,
        60,
        TrimSettings(
            fight_start_preroll_sec=0.8,
            output_context_padding_sec=0.5,
            combat_event_end_padding_sec=2.0,
            max_pre_fight_lead_sec=1.2,
            min_clip_duration_sec=14.0,
        ),
    )
    balanced = apply_highlight_trim_settings(trim, frames, timestamps, 60, TrimSettings())

    assert tight.clip_start == 10.8
    assert tight.clip_end == 38.5
    assert balanced.clip_start == 10.8
    assert balanced.clip_end == 38.5


def test_highlight_trim_settings_can_add_missing_incoming_rewind():
    frames = np.zeros((60, 1080, 1920, 3), dtype=np.uint8)
    timestamps = np.arange(60, dtype=np.float32)
    trim = TrimResult(clip_start=12, clip_end=30, fight_start=12, fight_end=28, fight_duration=16, dialog_segments=[], flags=["highlight_editor_model"])

    result = apply_highlight_trim_settings(
        trim,
        frames,
        timestamps,
        60,
        TrimSettings(fight_start_preroll_sec=2.0, output_context_padding_sec=0.0, max_pre_fight_lead_sec=2.0),
    )

    assert result.clip_start == 10.0
    assert "highlight_preroll_applied" in result.flags


def test_model_only_trim_settings_return_raw_model_trim():
    frames = np.zeros((60, 1080, 1920, 3), dtype=np.uint8)
    timestamps = np.arange(60, dtype=np.float32)
    for idx in range(40):
        cv2.rectangle(frames[idx], (800, 300), (900, 307), (210, 30, 30), thickness=-1)
        cv2.rectangle(frames[idx], (820, 360), (935, 367), (40, 210, 60), thickness=-1)
    for idx in range(40, 60):
        cv2.rectangle(frames[idx], (820, 360), (935, 367), (40, 210, 60), thickness=-1)
    trim = TrimResult(clip_start=12, clip_end=30, fight_start=12, fight_end=28, fight_duration=16, dialog_segments=[], flags=["highlight_editor_model"])

    result = apply_highlight_trim_settings(
        trim,
        frames,
        timestamps,
        60,
        TrimSettings(
            fight_start_preroll_sec=4.0,
            output_context_padding_sec=4.0,
            combat_event_end_padding_sec=4.0,
            max_pre_fight_lead_sec=4.0,
            min_clip_duration_sec=20.0,
            model_only=True,
        ),
    )

    assert result.clip_start == trim.clip_start
    assert result.clip_end == trim.clip_end
    assert result.fight_start == trim.fight_start
    assert result.fight_end == trim.fight_end
    assert "model_only_trim" in result.flags
    assert "highlight_preroll_applied" not in result.flags
    assert "output_context_padding_applied" not in result.flags
    assert "clip_end_extended_to_combat_event" not in result.flags


def test_highlight_trim_snaps_early_clip_to_sustained_healthbar_onset():
    frames = np.zeros((40, 1080, 1920, 3), dtype=np.uint8)
    timestamps = np.arange(40, dtype=np.float32)
    for idx in range(12, 25):
        _draw_player_enemy_bars(frames[idx])
    trim = TrimResult(clip_start=2, clip_end=24, fight_start=4, fight_end=24, fight_duration=20, dialog_segments=[], flags=[])

    result = apply_highlight_trim_settings(
        trim,
        frames,
        timestamps,
        40,
        TrimSettings(
            fight_start_preroll_sec=1.0,
            output_context_padding_sec=0.0,
            min_clip_duration_sec=4.0,
        ),
    )

    assert result.clip_start == 11.0
    assert result.fight_start == 12.0
    assert "fight_start_snapped_to_healthbar_onset" in result.flags


def test_highlight_trim_does_not_snap_to_single_healthbar_flicker():
    frames = np.zeros((40, 1080, 1920, 3), dtype=np.uint8)
    timestamps = np.arange(40, dtype=np.float32)
    _draw_player_enemy_bars(frames[12])
    trim = TrimResult(clip_start=2, clip_end=24, fight_start=4, fight_end=24, fight_duration=20, dialog_segments=[], flags=[])

    result = apply_highlight_trim_settings(
        trim,
        frames,
        timestamps,
        40,
        TrimSettings(
            fight_start_preroll_sec=1.0,
            output_context_padding_sec=0.0,
            min_clip_duration_sec=4.0,
        ),
    )

    assert result.clip_start == 2.8
    assert result.fight_start == 4
    assert "fight_start_snapped_to_healthbar_onset" not in result.flags


def test_detect_snaps_early_score_boundary_to_healthbar_onset():
    class StubFightDetector(FightDetector):
        def score_windows(self, full_frames: np.ndarray, timestamps: np.ndarray) -> list[float]:
            scores = [0.0] * 30
            scores[4:18] = [0.8] * 14
            return scores

        def transcribe(self, audio_path):
            return []

    frames = np.zeros((30, 1080, 1920, 3), dtype=np.uint8)
    timestamps = np.arange(30, dtype=np.float32)
    for idx in range(10, 18):
        _draw_player_enemy_bars(frames[idx])

    result = StubFightDetector().detect(
        frames,
        timestamps,
        30,
        None,
        TrimSettings(fight_start_preroll_sec=1.5),
    )

    assert result.clip_start == 8.5
    assert result.fight_start == 10.0
    assert "fight_start_snapped_to_healthbar_onset" in result.flags


def test_estimate_visible_enemy_count():
    frames = np.zeros((4, 1080, 1920, 3), dtype=np.uint8)
    timestamps = np.arange(4, dtype=np.float32)
    for idx in range(4):
        _draw_bar(frames[idx], 820, 360, 935, THICK_BAR_H, (40, 210, 60))
        _draw_bar(frames[idx], 760, 300, 860, THICK_BAR_H, (210, 30, 30))
        _draw_bar(frames[idx], 900, 300, 1000, THICK_BAR_H, (210, 30, 30))

    assert estimate_visible_enemy_count(frames, timestamps, 0, 3) == 2


def test_estimate_combat_screen_x_positions():
    frames = np.zeros((3, 1080, 1920, 3), dtype=np.uint8)
    for frame in frames:
        _draw_bar(frame, 820, 360, 940, THICK_BAR_H, (40, 210, 60))
        _draw_enemy_champion_bar(frame, 1000, 300, 1100)

    player_positions, threat_positions = estimate_combat_screen_x_positions(frames)

    assert player_positions == [880.0] * 3
    assert threat_positions == [1050.0] * 3


def test_estimate_combat_screen_x_positions_uses_nearest_red_bar():
    frames = np.zeros((3, 1080, 1920, 3), dtype=np.uint8)
    for frame in frames:
        _draw_bar(frame, 820, 360, 940, THICK_BAR_H, (40, 210, 60))
        _draw_enemy_champion_bar(frame, 1000, 300, 1100)
        _draw_enemy_champion_bar(frame, 1180, 300, 1260)

    _player_positions, threat_positions = estimate_combat_screen_x_positions(frames)

    assert threat_positions == [1050.0] * 3


def test_estimate_combat_screen_x_positions_uses_edge_enemy_that_needs_framing():
    frames = np.zeros((3, 1080, 1920, 3), dtype=np.uint8)
    for frame in frames:
        _draw_bar(frame, 900, 360, 1020, THICK_BAR_H, (40, 210, 60))
        _draw_enemy_champion_bar(frame, 1000, 300, 1100)
        _draw_enemy_champion_bar(frame, 1280, 300, 1380)

    _player_positions, threat_positions = estimate_combat_screen_x_positions(frames)

    assert threat_positions == [1330.0] * 3


def test_estimate_combat_screen_x_positions_uses_farther_enemy_for_crop_view():
    frames = np.zeros((3, 1080, 1920, 3), dtype=np.uint8)
    for frame in frames:
        _draw_bar(frame, 900, 360, 1020, THICK_BAR_H, (40, 210, 60))
        _draw_enemy_champion_bar(frame, 1000, 300, 1100)
        _draw_enemy_champion_bar(frame, 1450, 300, 1550)

    _player_positions, threat_positions = estimate_combat_screen_x_positions(frames)

    assert threat_positions == [1500.0] * 3


def test_estimate_combat_screen_x_positions_uses_visible_edge_enemy_above_player():
    frames = np.zeros((3, 1080, 1920, 3), dtype=np.uint8)
    for frame in frames:
        _draw_bar(frame, 900, 600, 1020, THICK_BAR_H, (40, 210, 60))
        _draw_enemy_champion_bar(frame, 1600, 150, 1720)

    _player_positions, threat_positions = estimate_combat_screen_x_positions(frames)

    assert threat_positions == [1660.0] * 3


def test_estimate_combat_screen_x_positions_ignores_left_hud_bars():
    frames = np.zeros((3, 1080, 1920, 3), dtype=np.uint8)
    for frame in frames:
        _draw_bar(frame, 190, 220, 300, THICK_BAR_H, (40, 210, 60))
        _draw_bar(frame, 820, 360, 940, THICK_BAR_H, (40, 210, 60))
        _draw_enemy_champion_bar(frame, 1000, 300, 1100)

    player_positions, threat_positions = estimate_combat_screen_x_positions(frames)

    assert player_positions == [880.0] * 3
    assert threat_positions == [1050.0] * 3


def test_estimate_combat_screen_x_positions_ignores_chat_overlay_red_text():
    frames = np.zeros((3, 1080, 1920, 3), dtype=np.uint8)
    for frame in frames:
        _draw_bar(frame, 820, 360, 940, THICK_BAR_H, (40, 210, 60))
        _draw_bar(frame, 332, 743, 405, 16, (210, 30, 30))

    player_positions, threat_positions = estimate_combat_screen_x_positions(frames)

    assert player_positions == [880.0] * 3
    assert threat_positions == [None] * 3


def test_estimate_combat_screen_x_positions_ignores_minimap_green_as_player():
    frames = np.zeros((3, 1080, 1920, 3), dtype=np.uint8)
    for frame in frames:
        _draw_bar(frame, 1746, 841, 1799, 14, (40, 210, 60))
        _draw_enemy_champion_bar(frame, 1000, 300, 1100)

    player_positions, threat_positions = estimate_combat_screen_x_positions(frames)

    assert player_positions == [None] * 3
    assert threat_positions == [None] * 3


def test_minion_red_health_bars_do_not_count_as_visible_enemies():
    frames = np.zeros((4, 1080, 1920, 3), dtype=np.uint8)
    timestamps = np.arange(4, dtype=np.float32)
    for idx in range(4):
        _draw_bar(frames[idx], 820, 360, 940, THICK_BAR_H, (40, 210, 60))
        _draw_bar(frames[idx], 1000, 300, 1060, THIN_BAR_H, (210, 30, 30))
        _draw_bar(frames[idx], 1120, 330, 1180, THIN_BAR_H, (210, 30, 30))

    assert estimate_visible_enemy_count(frames, timestamps, 0, 3) is None


def test_minion_red_health_bars_do_not_pull_crop_threat_position():
    frames = np.zeros((1, 1080, 1920, 3), dtype=np.uint8)
    _draw_bar(frames[0], 820, 360, 940, THICK_BAR_H, (40, 210, 60))
    _draw_bar(frames[0], 1000, 300, 1060, THIN_BAR_H, (210, 30, 30))
    _draw_bar(frames[0], 1120, 330, 1180, THIN_BAR_H, (210, 30, 30))

    player_positions, threat_positions = estimate_combat_screen_x_positions(frames)

    assert player_positions == [880.0]
    assert threat_positions == [None]


def test_champion_red_health_bar_still_pulls_crop_threat_position():
    frames = np.zeros((3, 1080, 1920, 3), dtype=np.uint8)
    for frame in frames:
        _draw_bar(frame, 820, 360, 940, THICK_BAR_H, (40, 210, 60))
        _draw_enemy_champion_bar(frame, 1000, 300, 1100)

    player_positions, threat_positions = estimate_combat_screen_x_positions(frames)

    assert player_positions == [880.0] * 3
    assert threat_positions == [1050.0] * 3


def test_blue_ally_health_bars_do_not_pull_crop_threat_position():
    frames = np.zeros((3, 1080, 1920, 3), dtype=np.uint8)
    for frame in frames:
        _draw_bar(frame, 820, 360, 940, THICK_BAR_H, (40, 210, 60))
        _draw_bar(frame, 1180, 300, 1300, THICK_BAR_H, (30, 120, 230))

    player_positions, threat_positions = estimate_combat_screen_x_positions(frames)

    assert player_positions == [880.0] * 3
    assert threat_positions == [None] * 3


def test_blue_ally_health_bars_do_not_beat_red_enemy_for_crop_threat():
    frames = np.zeros((3, 1080, 1920, 3), dtype=np.uint8)
    for frame in frames:
        _draw_bar(frame, 820, 360, 940, THICK_BAR_H, (40, 210, 60))
        _draw_bar(frame, 960, 300, 1060, THICK_BAR_H, (30, 120, 230))
        _draw_enemy_champion_bar(frame, 1200, 300, 1320)

    player_positions, threat_positions = estimate_combat_screen_x_positions(frames)

    assert player_positions == [880.0] * 3
    assert threat_positions == [1260.0] * 3


def test_estimate_combat_screen_x_tracks_reports_edge_ally_separately():
    frames = np.zeros((3, 1080, 1920, 3), dtype=np.uint8)
    for frame in frames:
        _draw_bar(frame, 820, 360, 940, THICK_BAR_H, (40, 210, 60))
        _draw_bar(frame, 520, 300, 640, THICK_BAR_H, (30, 120, 230))
        _draw_enemy_champion_bar(frame, 1200, 300, 1320)

    player_positions, threat_positions, ally_positions = estimate_combat_screen_x_position_tracks(frames)

    assert player_positions == [880.0] * 3
    assert threat_positions == [1260.0] * 3
    assert ally_positions == [580.0] * 3


def test_objective_health_number_red_bar_does_not_pull_crop_threat_position():
    frames = np.zeros((3, 1080, 1920, 3), dtype=np.uint8)
    for frame in frames:
        _draw_bar(frame, 820, 360, 940, THICK_BAR_H, (40, 210, 60))
        _draw_bar(frame, 1000, 300, 1120, THICK_BAR_H, (210, 30, 30))
        _draw_objective_health_number(frame, 1036, 299)

    player_positions, threat_positions = estimate_combat_screen_x_positions(frames)

    assert player_positions == [880.0] * 3
    assert threat_positions == [None] * 3


def test_objective_health_number_red_bar_does_not_beat_real_enemy_threat():
    frames = np.zeros((3, 1080, 1920, 3), dtype=np.uint8)
    for frame in frames:
        _draw_bar(frame, 820, 360, 940, THICK_BAR_H, (40, 210, 60))
        _draw_bar(frame, 980, 300, 1100, THICK_BAR_H, (210, 30, 30))
        _draw_objective_health_number(frame, 1016, 299)
        _draw_enemy_champion_bar(frame, 1200, 310, 1320)

    player_positions, threat_positions = estimate_combat_screen_x_positions(frames)

    assert player_positions == [880.0] * 3
    assert threat_positions == [1260.0] * 3


def test_objective_health_number_above_bar_does_not_pull_crop_threat_position():
    frames = np.zeros((3, 1080, 1920, 3), dtype=np.uint8)
    for frame in frames:
        _draw_bar(frame, 820, 360, 940, THICK_BAR_H, (40, 210, 60))
        _draw_bar(frame, 1000, 320, 1120, THICK_BAR_H, (210, 30, 30))
        _draw_objective_health_number(frame, 1036, 300)

    player_positions, threat_positions = estimate_combat_screen_x_positions(frames)

    assert player_positions == [880.0] * 3
    assert threat_positions == [None] * 3


def test_enemy_name_above_champion_badge_red_bar_still_pulls_crop_threat_position():
    frames = np.zeros((3, 1080, 1920, 3), dtype=np.uint8)
    for frame in frames:
        _draw_bar(frame, 820, 360, 940, THICK_BAR_H, (40, 210, 60))
        _draw_champion_level_badge(frame, 1168, 305)
        cv2.putText(
            frame,
            "EnemyName",
            (1195, 292),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (235, 235, 235),
            1,
            cv2.LINE_AA,
        )
        _draw_bar(frame, 1200, 300, 1320, THICK_BAR_H, (210, 30, 30))

    player_positions, threat_positions = estimate_combat_screen_x_positions(frames)

    assert player_positions == [880.0] * 3
    assert threat_positions == [1260.0] * 3


def test_champion_badge_enemy_bar_beats_nearer_unbadged_red_noise_for_crop_threat():
    frames = np.zeros((3, 1080, 1920, 3), dtype=np.uint8)
    for frame in frames:
        _draw_bar(frame, 820, 360, 940, THICK_BAR_H, (40, 210, 60))
        _draw_bar(frame, 640, 330, 760, THICK_BAR_H, (210, 30, 30))
        _draw_champion_level_badge(frame, 1168, 305)
        _draw_bar(frame, 1200, 300, 1320, THICK_BAR_H, (210, 30, 30))

    player_positions, threat_positions = estimate_combat_screen_x_positions(frames)

    assert player_positions == [880.0] * 3
    assert threat_positions == [1260.0] * 3


def test_neutral_jungle_monster_bar_does_not_pull_crop_threat_position():
    frames = np.zeros((3, 1080, 1920, 3), dtype=np.uint8)
    for frame in frames:
        _draw_bar(frame, 900, 360, 1020, THICK_BAR_H, (40, 210, 60))
        _draw_bar(frame, 1450, 330, 1580, THICK_BAR_H, (210, 30, 30))

    player_positions, threat_positions = estimate_combat_screen_x_positions(frames)

    assert player_positions == [960.0] * 3
    assert threat_positions == [None] * 3


def test_dynamic_crop_uses_default_thirds_when_only_minion_red_bars_are_visible():
    frames = np.zeros((8, 1080, 1920, 3), dtype=np.uint8)
    timestamps = np.arange(8, dtype=np.float32)
    for idx in range(8):
        _draw_bar(frames[idx], 900, 360, 1020, THICK_BAR_H, (40, 210, 60))
        _draw_bar(frames[idx], 1220, 300, 1280, THIN_BAR_H, (210, 30, 30))

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
    assert all(keyframe.crop_x == _right_thirds_crop_x() for keyframe in keyframes)


def test_dynamic_crop_can_shift_after_persistent_champion_red_bar():
    frames = np.zeros((8, 1080, 1920, 3), dtype=np.uint8)
    timestamps = np.arange(8, dtype=np.float32)
    for idx in range(8):
        _draw_bar(frames[idx], 900, 360, 1020, THICK_BAR_H, (40, 210, 60))
        _draw_enemy_champion_bar(frames[idx], 1200, 300, 1320)

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


def test_narrow_thick_low_health_champion_bar_pulls_crop_threat_position():
    frames = np.zeros((3, 1080, 1920, 3), dtype=np.uint8)
    for frame in frames:
        _draw_bar(frame, 820, 360, 940, THICK_BAR_H, (40, 210, 60))
        _draw_enemy_champion_bar(frame, 1000, 300, 1044)

    player_positions, threat_positions = estimate_combat_screen_x_positions(frames)

    assert player_positions == [880.0] * 3
    assert threat_positions == [1022.0] * 3


def test_wide_thin_merged_minion_wave_is_ignored_as_crop_threat():
    assert _camera_threat_bars([(1000, 300, 160, THIN_BAR_H)]) == []

def test_predict_highlight_trim_requires_decoded_frames():
    detector = FightDetector()

    with pytest.raises(HighlightEditorError, match="no decoded frames"):
        detector.predict_highlight_trim(
            np.empty((0, 224, 224, 3), dtype=np.uint8),
            np.empty((0,), dtype=np.float32),
            60.0,
        )
