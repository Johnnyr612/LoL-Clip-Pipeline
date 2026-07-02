from __future__ import annotations

import cv2
import numpy as np

from backend.team_tracker import TeamTracker, clustered_indices, infer_team_from_border


def _synthetic_icon(border_bgr: tuple[int, int, int], interior_bgr: tuple[int, int, int]) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    image = np.full((80, 80, 3), 45, dtype=np.uint8)
    center = (40, 40)
    radius = 20
    cv2.circle(image, center, radius, border_bgr, thickness=-1)
    cv2.circle(image, center, radius - 4, interior_bgr, thickness=-1)
    return image, (20, 20, 60, 60)


def test_red_interior_with_blue_border_classifies_as_ally():
    image, box = _synthetic_icon((255, 40, 20), (20, 20, 220))

    team, confidence = infer_team_from_border(image, box)

    assert team == "ally"
    assert confidence >= 0.95


def test_blue_interior_with_red_border_classifies_as_enemy():
    image, box = _synthetic_icon((20, 20, 220), (255, 40, 20))

    team, confidence = infer_team_from_border(image, box)

    assert team == "enemy"
    assert confidence >= 0.95


def test_gray_border_returns_none():
    image, box = _synthetic_icon((120, 120, 120), (20, 20, 220))

    team, confidence = infer_team_from_border(image, box)

    assert team is None
    assert confidence == 0.0


def test_clustered_indices_flags_overlapping_boxes():
    boxes = [
        (10, 10, 50, 50),
        (22, 22, 62, 62),
        (90, 90, 120, 120),
    ]

    assert clustered_indices(boxes) == {0, 1}


def test_tracker_finalizes_clean_votes_and_marks_single_vote_for_review():
    tracker = TeamTracker()
    for _ in range(8):
        tracker.update("Ahri", "ally", 0.96, 0.90)
    tracker.update("Aatrox", "enemy", 0.96, 0.90)

    team, confidence = tracker.get_team("Ahri")
    summary = tracker.summary()

    assert team == "ally"
    assert confidence >= 0.95
    assert summary["Ahri"]["team"] == "ally"
    assert summary["Ahri"]["needs_review"] is False
    assert summary["Ahri"]["vote_count"] == 8
    assert summary["Aatrox"]["team"] is None
    assert summary["Aatrox"]["needs_review"] is True
    assert summary["Aatrox"]["vote_count"] == 1
