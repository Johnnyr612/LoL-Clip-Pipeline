from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Sequence

import numpy as np

from . import config
from .fight_detector import _combat_health_bars, _enemy_bars_near_player, _select_player_health_bar

logger = logging.getLogger(__name__)

TeamHint = Literal["player", "ally", "enemy", "unknown"]

_YOLO_MODEL = None
_YOLO_LOAD_ERROR: str | None = None


@dataclass(frozen=True)
class VisionFightResult:
    player_champion: str
    enemy_champions: list[str]
    fight_type: str
    confidence: float


@dataclass(frozen=True)
class _Detection:
    champion_name: str
    confidence: float
    team: TeamHint
    box: tuple[float, float, float, float]

    @property
    def center(self) -> tuple[float, float]:
        x1, y1, x2, y2 = self.box
        return ((x1 + x2) / 2, (y1 + y2) / 2)


def classify_fight_participants(
    frames: np.ndarray,
    timestamps: np.ndarray,
    clip_start: float,
    clip_end: float,
) -> VisionFightResult | None:
    if len(frames) == 0 or len(timestamps) == 0:
        return None

    model = _load_yolo_model()
    if model is None:
        return None

    sampled = _sample_frames(frames, timestamps, clip_start, clip_end)
    if not sampled:
        return None

    player_votes: dict[str, list[float]] = {}
    enemy_votes: dict[str, list[float]] = {}

    for frame in sampled:
        detections = _detect_frame(model, frame)
        if not detections:
            continue

        player = _select_player_detection(frame, detections)
        if player is not None:
            player_votes.setdefault(player.champion_name, []).append(player.confidence)

        for enemy in _select_enemy_detections(frame, detections, player):
            enemy_votes.setdefault(enemy.champion_name, []).append(enemy.confidence)

    if not player_votes:
        return None

    player_name, player_confidence = _best_vote(player_votes)
    enemies = [
        name
        for name, _confidence in sorted(
            (_best_vote({name: scores}) for name, scores in enemy_votes.items() if name != player_name),
            key=lambda item: item[1],
            reverse=True,
        )
    ][:5]
    confidence = min(1.0, max(player_confidence, max((max(scores) for scores in enemy_votes.values()), default=0.0)))
    if confidence < config.VISION_CLASSIFIER_MIN_CONFIDENCE:
        return None
    return VisionFightResult(player_name, enemies, f"1v{max(1, len(enemies))}", confidence)


def _load_yolo_model():
    global _YOLO_MODEL, _YOLO_LOAD_ERROR
    if _YOLO_MODEL is not None:
        return _YOLO_MODEL
    if _YOLO_LOAD_ERROR:
        return None

    weights = config.YOLO_DETECTOR_WEIGHTS
    if weights is None:
        return None
    weights = Path(weights)
    if not weights.exists():
        _YOLO_LOAD_ERROR = f"YOLO weights not found: {weights}"
        logger.warning(_YOLO_LOAD_ERROR)
        return None

    try:
        from ultralytics import YOLO

        _YOLO_MODEL = YOLO(str(weights))
        return _YOLO_MODEL
    except Exception as exc:  # noqa: BLE001 - local detector is optional.
        _YOLO_LOAD_ERROR = str(exc)
        logger.warning("YOLO participant classifier unavailable: %s", exc)
        return None


def _sample_frames(frames: np.ndarray, timestamps: np.ndarray, clip_start: float, clip_end: float) -> list[np.ndarray]:
    indexes = np.flatnonzero((timestamps >= clip_start) & (timestamps <= clip_end))
    if len(indexes) == 0:
        indexes = np.array([int(np.argmin(np.abs(timestamps - clip_start)))])
    max_frames = max(1, config.YOLO_DETECTOR_MAX_FRAMES)
    if len(indexes) > max_frames:
        indexes = indexes[np.linspace(0, len(indexes) - 1, max_frames, dtype=int)]
    return [frames[int(index)] for index in indexes]


def _detect_frame(model, frame: np.ndarray) -> list[_Detection]:
    kwargs = {"conf": config.YOLO_DETECTOR_CONFIDENCE, "verbose": False}
    if config.YOLO_DETECTOR_DEVICE:
        kwargs["device"] = config.YOLO_DETECTOR_DEVICE
    try:
        results = model.predict(frame, **kwargs)
    except Exception as exc:  # noqa: BLE001 - failed YOLO inference should not kill the pipeline.
        logger.warning("YOLO participant inference failed: %s", exc)
        return []

    detections: list[_Detection] = []
    for result in results:
        names = getattr(result, "names", None) or getattr(model, "names", {})
        boxes = getattr(result, "boxes", None)
        if boxes is None:
            continue
        for box in boxes:
            cls = int(_tensor_scalar(box.cls))
            raw_name = str(names.get(cls, cls) if isinstance(names, dict) else names[cls])
            champion_name, team = _parse_class_name(raw_name)
            if not champion_name:
                continue
            confidence = float(_tensor_scalar(box.conf))
            xyxy = _tensor_array(box.xyxy).reshape(-1)[:4]
            if len(xyxy) != 4:
                continue
            detections.append(_Detection(champion_name, confidence, team, tuple(float(v) for v in xyxy)))
    return detections


def _select_player_detection(frame: np.ndarray, detections: Sequence[_Detection]) -> _Detection | None:
    explicit = [item for item in detections if item.team == "player"]
    if explicit:
        return max(explicit, key=lambda item: item.confidence)
    allies = [item for item in detections if item.team == "ally"]
    if allies:
        return max(allies, key=lambda item: item.confidence)

    _red_bars, green_bars = _combat_health_bars(frame)
    player_bar = _select_player_health_bar(green_bars)
    if player_bar is None:
        return max(detections, key=lambda item: item.confidence, default=None)
    target = _box_center(player_bar)
    return min(detections, key=lambda item: _distance(item.center, target), default=None)


def _select_enemy_detections(frame: np.ndarray, detections: Sequence[_Detection], player: _Detection | None) -> list[_Detection]:
    explicit = [item for item in detections if item.team == "enemy" and item is not player]
    if explicit:
        return _dedupe_by_champion(explicit)

    red_bars, green_bars = _combat_health_bars(frame)
    player_bar = _select_player_health_bar(green_bars)
    enemy_bars = _enemy_bars_near_player(red_bars, player_bar)
    if enemy_bars:
        enemy_centers = [_box_center(box) for box in enemy_bars]
        nearby = [
            item
            for item in detections
            if item is not player and min(_distance(item.center, center) for center in enemy_centers) <= 260
        ]
        if nearby:
            return _dedupe_by_champion(nearby)

    return _dedupe_by_champion([item for item in detections if item is not player])


def _dedupe_by_champion(detections: Sequence[_Detection]) -> list[_Detection]:
    best: dict[str, _Detection] = {}
    for detection in detections:
        current = best.get(detection.champion_name)
        if current is None or detection.confidence > current.confidence:
            best[detection.champion_name] = detection
    return list(best.values())


def _parse_class_name(raw_name: str) -> tuple[str, TeamHint]:
    normalized = raw_name.strip()
    if not normalized:
        return "", "unknown"

    team: TeamHint = "unknown"
    lowered = normalized.lower().replace("-", "_").replace(" ", "_")
    if re.match(r"^(player|self|me)(_|:)", lowered):
        team = "player"
    elif re.match(r"^(ally|blue|green)(_|:)", lowered):
        team = "ally"
    elif re.match(r"^(enemy|red)(_|:)", lowered):
        team = "enemy"

    name = re.sub(r"^(player|self|me|ally|blue|green|enemy|red)[_:\-\s]+", "", normalized, flags=re.IGNORECASE)
    name = name.replace("_", " ").strip()
    if name.lower() in {"player", "ally", "enemy", "champion", "unknown"}:
        return "", team
    return name, team


def _best_vote(votes: dict[str, list[float]]) -> tuple[str, float]:
    name, scores = max(votes.items(), key=lambda item: (len(item[1]), float(np.mean(item[1]))))
    return name, float(np.mean(scores))


def _box_center(box: tuple[int, int, int, int]) -> tuple[float, float]:
    x, y, width, height = box
    return (float(x + (width - 1) / 2), float(y + (height - 1) / 2))


def _distance(a: tuple[float, float], b: tuple[float, float]) -> float:
    return float(np.linalg.norm(np.array(a, dtype=np.float32) - np.array(b, dtype=np.float32)))


def _tensor_scalar(value: object) -> float:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    array = np.asarray(value).reshape(-1)
    return float(array[0]) if len(array) else 0.0


def _tensor_array(value: object) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value, dtype=np.float32)
