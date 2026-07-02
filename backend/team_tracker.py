from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import cv2
import numpy as np

Box = tuple[float, float, float, float]


def infer_team_from_border(image_bgr: np.ndarray, box_xyxy: Sequence[float]) -> tuple[str | None, float]:
    x1, y1, x2, y2 = [float(value) for value in box_xyxy[:4]]
    if x2 <= x1 or y2 <= y1 or image_bgr.size == 0:
        return None, 0.0

    height, width = image_bgr.shape[:2]
    center_x = (x1 + x2) / 2.0
    center_y = (y1 + y2) / 2.0
    radius = max((x2 - x1) / 2.0, (y2 - y1) / 2.0)
    if radius <= 0:
        return None, 0.0

    pad = max(2, int(round(radius * 0.10)))
    crop_x1 = max(0, int(np.floor(x1 - pad)))
    crop_y1 = max(0, int(np.floor(y1 - pad)))
    crop_x2 = min(width, int(np.ceil(x2 + pad)))
    crop_y2 = min(height, int(np.ceil(y2 + pad)))
    if crop_x2 <= crop_x1 or crop_y2 <= crop_y1:
        return None, 0.0

    crop = image_bgr[crop_y1:crop_y2, crop_x1:crop_x2]
    local_center_x = center_x - crop_x1
    local_center_y = center_y - crop_y1
    yy, xx = np.ogrid[: crop.shape[0], : crop.shape[1]]
    distance = np.sqrt((xx - local_center_x) ** 2 + (yy - local_center_y) ** 2)
    annulus = (distance >= radius * 0.85) & (distance <= radius * 1.05)
    ring_pixels = int(np.count_nonzero(annulus))
    if ring_pixels < 30:
        return None, 0.0

    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    hue = hsv[:, :, 0]
    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]
    saturated = (saturation >= 80) & (value >= 80)
    blue_mask = annulus & saturated & (hue >= 95) & (hue <= 130)
    red_mask = annulus & saturated & (((hue >= 0) & (hue <= 10)) | ((hue >= 165) & (hue <= 180)))

    blue_count = int(np.count_nonzero(blue_mask))
    red_count = int(np.count_nonzero(red_mask))
    colored_count = blue_count + red_count
    if colored_count / ring_pixels < 0.10:
        return None, 0.0

    winning_count = max(blue_count, red_count)
    confidence = winning_count / colored_count
    if confidence < 0.60:
        return None, 0.0
    return ("ally" if blue_count > red_count else "enemy"), float(confidence)


def clustered_indices(boxes: Sequence[Sequence[float]], iou_threshold: float = 0.30) -> set[int]:
    clustered: set[int] = set()
    for left_idx in range(len(boxes)):
        for right_idx in range(left_idx + 1, len(boxes)):
            if _iou(boxes[left_idx], boxes[right_idx]) > iou_threshold:
                clustered.add(left_idx)
                clustered.add(right_idx)
    return clustered


@dataclass
class _TeamRecord:
    team: str | None = None
    confidence: float = 0.0
    weight_total: float = 0.0
    vote_count: int = 0
    votes: dict[str, int] = field(default_factory=lambda: {"ally": 0, "enemy": 0})


class TeamTracker:
    def __init__(self, min_votes: int = 2, min_confidence: float = 0.85) -> None:
        self.min_votes = min_votes
        self.min_confidence = min_confidence
        self.registry: dict[str, _TeamRecord] = {}

    def update(
        self,
        champion_class: str,
        team_vote: str | None,
        border_confidence: float,
        yolo_confidence: float,
    ) -> None:
        if team_vote is None:
            return
        if team_vote not in {"ally", "enemy"}:
            return
        champion_class = str(champion_class or "").strip()
        if not champion_class:
            return

        weight = max(0.0, float(border_confidence)) * max(0.0, float(yolo_confidence))
        if weight <= 0:
            return

        record = self.registry.setdefault(champion_class, _TeamRecord())
        next_weight_total = record.weight_total + weight
        record.confidence = (
            (record.confidence * record.weight_total + float(border_confidence) * weight)
            / next_weight_total
        )
        record.weight_total = next_weight_total
        record.vote_count += 1
        record.votes[team_vote] = record.votes.get(team_vote, 0) + 1
        record.team = _majority_team(record.votes)

    def get_team(self, champion_class: str) -> tuple[str | None, float]:
        record = self.registry.get(champion_class)
        if record is None:
            return None, 0.0
        if record.vote_count < self.min_votes or record.confidence < self.min_confidence:
            return None, 0.0
        return record.team, float(record.confidence)

    def summary(self) -> dict:
        return {
            champion: {
                "team": self.get_team(champion)[0],
                "confidence": round(float(record.confidence), 4),
                "vote_count": int(record.vote_count),
                "votes": dict(record.votes),
                "needs_review": self.get_team(champion)[0] is None,
            }
            for champion, record in sorted(self.registry.items())
        }


def _majority_team(votes: dict[str, int]) -> str | None:
    ally_votes = int(votes.get("ally", 0))
    enemy_votes = int(votes.get("enemy", 0))
    if ally_votes == enemy_votes:
        return None
    return "ally" if ally_votes > enemy_votes else "enemy"


def _iou(left: Sequence[float], right: Sequence[float]) -> float:
    lx1, ly1, lx2, ly2 = [float(value) for value in left[:4]]
    rx1, ry1, rx2, ry2 = [float(value) for value in right[:4]]
    ix1 = max(lx1, rx1)
    iy1 = max(ly1, ry1)
    ix2 = min(lx2, rx2)
    iy2 = min(ly2, ry2)
    intersection = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    left_area = max(0.0, lx2 - lx1) * max(0.0, ly2 - ly1)
    right_area = max(0.0, rx2 - rx1) * max(0.0, ry2 - ry1)
    union = left_area + right_area - intersection
    return intersection / union if union > 0 else 0.0
