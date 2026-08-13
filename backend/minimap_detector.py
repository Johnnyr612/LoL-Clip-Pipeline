from __future__ import annotations

import json
import logging
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Optional

import cv2
import numpy as np
from PIL import Image

from . import config

Rect = tuple[int, int, int, int]

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RawIconDetection:
    circle_center: tuple[int, int]
    radius: int
    team: Literal["ally", "enemy", "unknown"]
    champion_name: str
    match_score: float
    is_uncertain: bool


@dataclass(frozen=True)
class ChampionResult:
    champion_name: str
    confidence: float
    team: Literal["ally", "enemy", "unknown"]
    mean_pos: tuple[float, float]
    is_player: bool = False


@dataclass(frozen=True)
class FightParticipants:
    player: ChampionResult
    allies: list[ChampionResult]
    enemies: list[ChampionResult]
    fight_type: str
    flags: list[str]


class MinimapDetector:
    def __init__(self, icons_dir: Path, manifest_path: Path):
        self.icons_dir = Path(icons_dir)
        self.manifest_path = Path(manifest_path)
        self.name_to_key: dict[str, str] = {}
        self.key_to_name: dict[str, str] = {}
        self._champion_name_lookup: dict[str, str] = {}
        self.templates: dict[str, np.ndarray] = {}
        self.base_names: list[str] = []
        self.base_matrix = np.empty((0, 90 * 90), dtype=np.float32)
        self.phash_templates: dict[str, list[np.ndarray]] = {}
        self._yolo_model: Any | None = None
        self._yolo_load_error: str | None = None
        self._minimap_rect: Optional[Rect] = None
        self.minimap_boundary_estimated = False

        self._load_manifest()
        self._load_icons()

    def _load_manifest(self) -> None:
        manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        for item in manifest:
            self.name_to_key[item["name"]] = item["id"]
            self.key_to_name[item["id"]] = item["name"]
            self._champion_name_lookup[_champion_lookup_key(item["name"])] = item["name"]
            self._champion_name_lookup[_champion_lookup_key(item["id"])] = item["name"]

    def _load_icons(self) -> None:
        images_dir = self.icons_dir / "images"
        for path in sorted(images_dir.glob("*.png")):
            name = path.stem
            try:
                with Image.open(path) as image:
                    image = image.convert("RGB")
                    if image.size != (120, 120):
                        raise ValueError(f"expected 120x120, got {image.size}")
                    rgb = np.array(image)
            except Exception as exc:  # noqa: BLE001 - corrupt icons must not crash startup.
                logger.warning("Skipping champion icon %s: %s", path, exc)
                continue

            self.templates[name] = self._center_crop_gray(rgb)
            self.phash_templates[name] = [self._phash(rgb), self._phash(rgb[15:105, 15:105])]
        self._build_match_indexes()

    def _build_match_indexes(self) -> None:
        self.base_names = list(self.templates)
        if self.base_names:
            self.base_matrix = self._normalize_rows([self.templates[name] for name in self.base_names])

    @staticmethod
    def _normalize_rows(images: list[np.ndarray]) -> np.ndarray:
        if not images:
            return np.empty((0, 90 * 90), dtype=np.float32)
        matrix = np.asarray([image.reshape(-1) for image in images], dtype=np.float32)
        matrix -= matrix.mean(axis=1, keepdims=True)
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        return matrix / np.maximum(norms, 1e-6)

    @staticmethod
    def _normalize_roi(roi_gray: np.ndarray) -> np.ndarray:
        roi = cv2.resize(roi_gray, (90, 90), interpolation=cv2.INTER_AREA).reshape(-1).astype(np.float32)
        roi -= float(roi.mean())
        norm = float(np.linalg.norm(roi))
        if norm <= 1e-6:
            return np.zeros_like(roi)
        return roi / norm

    @staticmethod
    def _center_crop_gray(rgb: np.ndarray, crop: int = 90) -> np.ndarray:
        start = (rgb.shape[0] - crop) // 2
        center = rgb[start : start + crop, start : start + crop]
        return cv2.cvtColor(center, cv2.COLOR_RGB2GRAY)

    @staticmethod
    def _phash(rgb: np.ndarray) -> np.ndarray:
        resized = cv2.resize(rgb, (96, 96), interpolation=cv2.INTER_AREA)
        lab = cv2.cvtColor(resized, cv2.COLOR_RGB2LAB)
        lightness, a, b = cv2.split(lab)
        equalized = cv2.equalizeHist(lightness)
        normalized = cv2.cvtColor(cv2.merge([equalized, a, b]), cv2.COLOR_LAB2RGB)
        gray = cv2.cvtColor(cv2.resize(normalized, (32, 32), interpolation=cv2.INTER_AREA), cv2.COLOR_RGB2GRAY).astype(np.float32)
        dct = cv2.dct(gray)
        low_freq = dct[:8, :8].reshape(-1)
        median = float(np.median(low_freq[1:]))
        return low_freq > median

    def locate_minimap(self, frame: np.ndarray) -> Rect:
        if self._minimap_rect is not None:
            return self._minimap_rect

        h, w = frame.shape[:2]
        qx, qy = int(w * 0.50), int(h * 0.50)
        roi = frame[qy:h, qx:w]
        hsv = cv2.cvtColor(roi, cv2.COLOR_RGB2HSV)
        mask = cv2.inRange(hsv, (0, 0, 0), (180, 255, 40))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        min_area = w * h * 0.01
        candidates: list[tuple[float, Rect]] = []
        for contour in contours:
            x, y, bw, bh = cv2.boundingRect(contour)
            area = float(cv2.contourArea(contour))
            aspect = bw / max(bh, 1)
            if 0.8 <= aspect <= 1.2 and area >= min_area:
                candidates.append((area, (x + qx, y + qy, bw, bh)))

        if candidates:
            self._minimap_rect = max(candidates, key=lambda item: item[0])[1]
            self.minimap_boundary_estimated = False
            return self._minimap_rect

        self.minimap_boundary_estimated = True
        self._minimap_rect = (
            int(w * config.MINIMAP_CROP_X_PCT),
            int(h * config.MINIMAP_CROP_Y_PCT),
            int(w * (1.0 - config.MINIMAP_CROP_X_PCT)),
            int(h * (1.0 - config.MINIMAP_CROP_Y_PCT)),
        )
        return self._minimap_rect

    @staticmethod
    def classify_team(roi_rgb: np.ndarray) -> Literal["ally", "enemy", "unknown"]:
        h, w = roi_rgb.shape[:2]
        cy, cx = h / 2, w / 2
        radius = min(h, w) / 2 - 1
        yy, xx = np.ogrid[:h, :w]
        distance = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
        ring = np.abs(distance - radius) <= 8
        if not np.any(ring):
            return "unknown"
        hsv = cv2.cvtColor(roi_rgb, cv2.COLOR_RGB2HSV)
        pixels = hsv[ring]
        red = (((pixels[:, 0] < 15) | (pixels[:, 0] > 165)) & (pixels[:, 1] > 100)).sum()
        blue = ((pixels[:, 0] >= 100) & (pixels[:, 0] <= 130) & (pixels[:, 1] > 100)).sum()
        total = max(len(pixels), 1)
        if red / total > 0.10:
            return "enemy"
        if blue / total > 0.10:
            return "ally"
        return "unknown"

    def match_champion(self, roi_gray: np.ndarray) -> tuple[str, float]:
        if self.base_matrix.size == 0:
            return "unknown", -1.0

        roi = self._normalize_roi(roi_gray)
        base_scores = self.base_matrix @ roi
        candidate_count = min(config.MINIMAP_TOP_TEMPLATE_CANDIDATES, len(self.base_names))
        if candidate_count == len(self.base_names):
            candidate_indices = np.arange(len(self.base_names))
        else:
            candidate_indices = np.argpartition(base_scores, -candidate_count)[-candidate_count:]
        best_name = "unknown"
        best_score = -1.0

        for idx in candidate_indices:
            name = self.base_names[int(idx)]
            score = float(base_scores[int(idx)])
            if score > best_score:
                best_name = name
                best_score = score
        return best_name, best_score

    def match_champion_rgb(self, roi_rgb: np.ndarray) -> tuple[str, float]:
        gray = cv2.cvtColor(cv2.resize(roi_rgb, (90, 90), interpolation=cv2.INTER_AREA), cv2.COLOR_RGB2GRAY)
        gray_name, gray_score = self.match_champion(gray)
        phash_name, phash_distance = self._match_phash(roi_rgb)
        if (
            phash_distance <= config.ICON_PHASH_CONFIRM_MAX_DISTANCE
            and (gray_score >= config.ICON_PHASH_MIN_GRAY_SCORE or gray_name == phash_name)
        ):
            confidence = max(gray_score, 1.0 - phash_distance / 64.0)
            return phash_name, float(confidence)
        if phash_distance <= config.ICON_PHASH_SUPPORT_MAX_DISTANCE and gray_name == phash_name:
            return gray_name, max(gray_score, float(1.0 - phash_distance / 64.0))
        return gray_name, gray_score

    def _match_phash(self, roi_rgb: np.ndarray) -> tuple[str, int]:
        roi_hash = self._phash(roi_rgb)
        best_name = "unknown"
        best_distance = 65
        for name, hashes in self.phash_templates.items():
            distance = min(int(np.count_nonzero(roi_hash != template_hash)) for template_hash in hashes)
            if distance < best_distance:
                best_name = name
                best_distance = distance
        return best_name, best_distance

    def detect_player_hud_champion(self, frame: np.ndarray) -> tuple[str, float]:
        h, w = frame.shape[:2]
        candidates = [
            # Top-left player panel portrait.
            (int(w * 0.039), int(h * 0.005), int(w * 0.081), int(h * 0.080)),
            # Bottom HUD portrait fallback.
            (int(w * 0.315), int(h * 0.900), int(w * 0.370), h),
        ]
        matches: list[tuple[str, float]] = []
        for x1, y1, x2, y2 in candidates:
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            square = _center_square(crop)
            matches.append(self.match_champion_rgb(cv2.resize(square, (120, 120), interpolation=cv2.INTER_AREA)))
        if not matches:
            return "unknown", -1.0
        return max(matches, key=lambda item: item[1])

    def detect_icons(self, minimap_frame: np.ndarray) -> list[RawIconDetection]:
        return self.detect_icons_yolo(minimap_frame)

    def detect_icons_yolo(self, minimap_frame: np.ndarray) -> list[RawIconDetection]:
        model = self._load_yolo_model()
        if model is None:
            return []

        kwargs: dict[str, object] = {
            "conf": config.MINIMAP_YOLO_CONFIDENCE,
            "max_det": config.MINIMAP_YOLO_MAX_DETECTIONS,
            "verbose": False,
        }
        if config.MINIMAP_YOLO_DEVICE:
            kwargs["device"] = config.MINIMAP_YOLO_DEVICE

        try:
            bgr_frame = cv2.cvtColor(minimap_frame, cv2.COLOR_RGB2BGR)
            results = model.predict(bgr_frame, **kwargs)
        except Exception as exc:  # noqa: BLE001 - failed YOLO inference should not kill aggregation.
            logger.warning("YOLO minimap champion inference failed: %s", exc)
            return []

        detections: list[RawIconDetection] = []
        height, width = minimap_frame.shape[:2]
        for result in results:
            names = getattr(result, "names", None) or getattr(model, "names", {})
            boxes = getattr(result, "boxes", None)
            if boxes is None:
                continue
            for box in boxes:
                confidence = float(_tensor_scalar(box.conf))
                if confidence < config.MINIMAP_YOLO_CONFIDENCE:
                    continue
                cls = int(_tensor_scalar(box.cls))
                champion_name = self._normalize_yolo_champion_name(_class_name(names, cls))
                if not champion_name:
                    continue

                xyxy = _tensor_array(box.xyxy).reshape(-1)[:4]
                if len(xyxy) != 4:
                    continue
                x1 = int(np.clip(np.floor(xyxy[0]), 0, max(width - 1, 0)))
                y1 = int(np.clip(np.floor(xyxy[1]), 0, max(height - 1, 0)))
                x2 = int(np.clip(np.ceil(xyxy[2]), x1 + 1, width))
                y2 = int(np.clip(np.ceil(xyxy[3]), y1 + 1, height))
                roi = minimap_frame[y1:y2, x1:x2]
                if roi.size == 0:
                    continue

                center = (int(round((x1 + x2) / 2)), int(round((y1 + y2) / 2)))
                radius = max(1, int(round(max(x2 - x1, y2 - y1) / 2)))
                detections.append(
                    RawIconDetection(
                        center,
                        radius,
                        self.classify_team(roi),
                        champion_name,
                        confidence,
                        confidence < config.MINIMAP_YOLO_CONFIRM,
                    )
                )

        detections.sort(key=lambda item: item.match_score, reverse=True)
        return detections[: config.MINIMAP_YOLO_MAX_DETECTIONS]

    def _load_yolo_model(self):
        if self._yolo_model is not None:
            return self._yolo_model
        if self._yolo_load_error or not config.MINIMAP_YOLO_ENABLED:
            return None

        weights = Path(config.MINIMAP_YOLO_WEIGHTS)
        if not weights.exists():
            self._yolo_load_error = f"YOLO minimap weights not found: {weights}"
            logger.warning(self._yolo_load_error)
            return None

        try:
            from ultralytics import YOLO

            self._yolo_model = YOLO(str(weights))
            return self._yolo_model
        except Exception as exc:  # noqa: BLE001 - unavailable YOLO should produce no minimap detections.
            self._yolo_load_error = str(exc)
            logger.warning("YOLO minimap champion detector unavailable: %s", exc)
            return None

    def _normalize_yolo_champion_name(self, raw_name: str) -> str:
        normalized = raw_name.strip().replace("_", " ")
        normalized = re.sub(r"^(player|self|me|ally|blue|green|enemy|red)[\s:_-]+", "", normalized, flags=re.IGNORECASE)
        normalized = normalized.strip()
        if not normalized or normalized.lower() in {"player", "ally", "enemy", "champion", "unknown"}:
            return ""
        return self._champion_name_lookup.get(_champion_lookup_key(normalized), normalized)

    def aggregate_detections(
        self,
        detections_per_frame: list[list[RawIconDetection]],
        timestamps: np.ndarray,
        fight_start: float,
        fight_end: float,
        player_champion: str | None = None,
    ) -> FightParticipants:
        tracks: list[list[RawIconDetection]] = []
        last_points: list[tuple[int, int]] = []
        for frame_detections, timestamp in zip(detections_per_frame, timestamps):
            if not fight_start <= float(timestamp) <= fight_end:
                continue
            frame_player_pos = _player_detection_position(frame_detections, player_champion)
            frame_detections = _fight_relevant_detections(frame_detections, frame_player_pos)
            used_tracks: set[int] = set()
            for detection in frame_detections:
                point = np.array(detection.circle_center)
                best_idx: int | None = None
                best_dist = 1e9
                for idx, last in enumerate(last_points):
                    if idx in used_tracks:
                        continue
                    dist = float(np.linalg.norm(point - np.array(last)))
                    if dist < best_dist and dist < 15:
                        best_idx, best_dist = idx, dist
                if best_idx is None:
                    tracks.append([detection])
                    last_points.append(detection.circle_center)
                    used_tracks.add(len(tracks) - 1)
                else:
                    tracks[best_idx].append(detection)
                    last_points[best_idx] = detection.circle_center
                    used_tracks.add(best_idx)

        flags: list[str] = []
        if not tracks:
            unknown = ChampionResult(_known_player_champion(player_champion, "unknown_champion_0"), 0.0, "ally", (0.5, 0.5), True)
            return FightParticipants(unknown, [], [], "1v1", ["no_champions_identified"])

        results: list[ChampionResult] = []
        for track in tracks:
            certain_names = [d.champion_name for d in track if not d.is_uncertain and not d.champion_name.startswith("unknown")]
            if certain_names:
                champion_name, vote_count = Counter(certain_names).most_common(1)[0]
            else:
                champion_name, vote_count = Counter(d.champion_name for d in track).most_common(1)[0]
            team = Counter(d.team for d in track).most_common(1)[0][0]
            points = np.array([d.circle_center for d in track], dtype=np.float32)
            results.append(ChampionResult(champion_name, vote_count / len(track), team, tuple(points.mean(axis=0))))
        results = _merge_duplicate_champions(results)

        player_idx = next((i for i, r in enumerate(results) if r.team == "ally" and r.champion_name == player_champion), None)
        if player_idx is None:
            flags.append("player_pos_unknown")
            player_idx = next((i for i, r in enumerate(results) if r.team == "ally"), 0)

        player = ChampionResult(
            _known_player_champion(player_champion, results[player_idx].champion_name),
            results[player_idx].confidence,
            results[player_idx].team,
            results[player_idx].mean_pos,
            True,
        )
        allies = [r for i, r in enumerate(results) if i != player_idx and r.team == "ally"]
        enemies = [r for i, r in enumerate(results) if r.team == "enemy"]
        if all(r.champion_name.startswith("unknown") for r in results):
            flags.append("no_champions_identified")
        fight_type = _fight_type(player, allies, enemies)
        return FightParticipants(player, allies, enemies, fight_type, flags)


def _center_square(image: np.ndarray) -> np.ndarray:
    h, w = image.shape[:2]
    side = min(h, w)
    y = max((h - side) // 2, 0)
    x = max((w - side) // 2, 0)
    return image[y : y + side, x : x + side]


def _champion_lookup_key(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", name.lower())


def _class_name(names: object, cls: int) -> str:
    if isinstance(names, dict):
        return str(names.get(cls, cls))
    try:
        return str(names[cls])  # type: ignore[index]
    except Exception:  # noqa: BLE001 - malformed model metadata should skip cleanly.
        return str(cls)


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


def _known_player_champion(player_champion: str | None, fallback: str) -> str:
    if player_champion and not player_champion.startswith("unknown"):
        return player_champion
    return fallback


def _player_detection_position(
    detections: list[RawIconDetection],
    player_champion: str | None,
) -> tuple[float, float] | None:
    if not player_champion or player_champion.startswith("unknown"):
        return None
    candidates = [
        detection
        for detection in detections
        if detection.team == "ally" and detection.champion_name == player_champion and not detection.is_uncertain
    ]
    if not candidates:
        candidates = [
            detection
            for detection in detections
            if detection.champion_name == player_champion and not detection.is_uncertain
        ]
    if not candidates:
        return None
    best = max(candidates, key=lambda detection: detection.match_score)
    return (float(best.circle_center[0]), float(best.circle_center[1]))


def _fight_relevant_detections(
    detections: list[RawIconDetection],
    player_position: tuple[float, float] | None,
) -> list[RawIconDetection]:
    if player_position is None:
        return [d for d in detections if _is_usable_detection(d, detections)]

    center = np.array(player_position, dtype=np.float32)
    relevant: list[RawIconDetection] = []
    for detection in detections:
        distance = float(np.linalg.norm(np.array(detection.circle_center, dtype=np.float32) - center))
        if distance > config.MINIMAP_FIGHT_RADIUS_PX:
            continue
        if not _is_usable_detection(detection, detections):
            continue
        relevant.append(detection)
    return relevant


def _is_usable_detection(detection: RawIconDetection, frame_detections: list[RawIconDetection]) -> bool:
    if not detection.is_uncertain and not detection.champion_name.startswith("unknown"):
        return True
    for other in frame_detections:
        if other is detection:
            continue
        distance = float(np.linalg.norm(np.array(detection.circle_center) - np.array(other.circle_center)))
        overlap_distance = (detection.radius + other.radius) * config.MINIMAP_OVERLAP_DISTANCE_MULTIPLIER
        if distance < overlap_distance:
            return False
    return True


def _merge_duplicate_champions(results: list[ChampionResult]) -> list[ChampionResult]:
    grouped: dict[tuple[str, str], list[ChampionResult]] = {}
    merged: list[ChampionResult] = []
    unknown_counter = 0
    for result in results:
        if result.champion_name.startswith("unknown"):
            key = (f"unknown_{unknown_counter}", result.team)
            unknown_counter += 1
        else:
            key = (result.champion_name, result.team)
        grouped.setdefault(key, []).append(result)

    for group in grouped.values():
        if len(group) == 1:
            merged.append(group[0])
            continue
        weights = np.array([max(item.confidence, 0.01) for item in group], dtype=np.float32)
        positions = np.array([item.mean_pos for item in group], dtype=np.float32)
        mean_pos = tuple(np.average(positions, axis=0, weights=weights))
        representative = max(group, key=lambda item: item.confidence)
        merged.append(
            ChampionResult(
                representative.champion_name,
                float(np.mean([item.confidence for item in group])),
                representative.team,
                mean_pos,
                representative.is_player,
            )
        )
    return merged


def _fight_type(player: ChampionResult, allies: list[ChampionResult], enemies: list[ChampionResult]) -> str:
    ally_pool = [player, *allies]
    known_allies = [item for item in ally_pool if not item.champion_name.startswith("unknown")]
    known_enemies = [item for item in enemies if not item.champion_name.startswith("unknown")]
    ally_count = len(known_allies) if known_allies else len(ally_pool)
    enemy_count = len(known_enemies) if known_enemies else len(enemies)
    return f"{min(max(ally_count, 1), 5)}v{min(max(enemy_count, 1), 5)}"
