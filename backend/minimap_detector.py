from __future__ import annotations

import json
import logging
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional

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
        self.templates: dict[str, np.ndarray] = {}
        self.augmented: dict[str, list[np.ndarray]] = {}
        self.base_names: list[str] = []
        self.base_matrix = np.empty((0, 90 * 90), dtype=np.float32)
        self.augmented_matrices: dict[str, np.ndarray] = {}
        self._minimap_rect: Optional[Rect] = None
        self.minimap_boundary_estimated = False

        self._load_manifest()
        self._load_icons()

    def _load_manifest(self) -> None:
        manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        for item in manifest:
            self.name_to_key[item["name"]] = item["id"]
            self.key_to_name[item["id"]] = item["name"]

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
            self.augmented[name] = self._augment_template(rgb)
        self._build_match_indexes()

    def _build_match_indexes(self) -> None:
        self.base_names = list(self.templates)
        if self.base_names:
            self.base_matrix = self._normalize_rows([self.templates[name] for name in self.base_names])
        self.augmented_matrices = {
            name: self._normalize_rows(templates)
            for name, templates in self.augmented.items()
            if templates
        }

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
    def _scale_pad(rgb: np.ndarray, scale: float) -> np.ndarray:
        size = max(1, int(round(120 * scale)))
        resized = cv2.resize(rgb, (size, size), interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC)
        if size >= 120:
            offset = (size - 120) // 2
            return resized[offset : offset + 120, offset : offset + 120]
        canvas = np.zeros((120, 120, 3), dtype=np.uint8)
        offset = (120 - size) // 2
        canvas[offset : offset + size, offset : offset + size] = resized
        return canvas

    @staticmethod
    def _brightness(rgb: np.ndarray, factor: float) -> np.ndarray:
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV).astype(np.float32)
        hsv[:, :, 2] = np.clip(hsv[:, :, 2] * factor, 0, 255)
        return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)

    @staticmethod
    def _border_variant(rgb: np.ndarray, color: tuple[int, int, int]) -> np.ndarray:
        out = rgb.copy()
        cv2.circle(out, (60, 60), 59, color, thickness=15, lineType=cv2.LINE_AA)
        return out

    def _augment_template(self, rgb: np.ndarray) -> list[np.ndarray]:
        variants: list[np.ndarray] = []
        border_colors = [(220, 35, 35), (35, 110, 235)]
        blur_sigmas = [0.5, 1.0]
        scales = [0.85, 1.00, 1.10]
        brightness = [0.80, 1.00, 1.15]
        for border in border_colors:
            bordered = self._border_variant(rgb, border)
            for sigma in blur_sigmas:
                blurred = cv2.GaussianBlur(bordered, (0, 0), sigmaX=sigma)
                for scale in scales:
                    scaled = self._scale_pad(blurred, scale)
                    for factor in brightness:
                        variants.append(self._center_crop_gray(self._brightness(scaled, factor)))
        return variants

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

    def find_white_box(self, minimap_frame: np.ndarray) -> Optional[tuple[float, float]]:
        h, w = minimap_frame.shape[:2]
        hsv = cv2.cvtColor(minimap_frame, cv2.COLOR_RGB2HSV)
        mask = cv2.inRange(hsv, (0, 0, 220), (180, 30, 255))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in sorted(contours, key=cv2.contourArea, reverse=True):
            area = cv2.contourArea(contour)
            if not 400 <= area <= 2500:
                continue
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.04 * peri, True)
            x, y, bw, bh = cv2.boundingRect(approx)
            aspect = bw / max(bh, 1)
            if len(approx) >= 4 and 0.5 <= aspect <= 2.0:
                return ((x + bw / 2) / w, (y + bh / 2) / h)
        return None

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
            augmented = self.augmented_matrices.get(name)
            if augmented is None or augmented.size == 0:
                score = float(base_scores[int(idx)])
            else:
                score = float(np.max(augmented @ roi))
            if score > best_score:
                best_name = name
                best_score = score
        return best_name, best_score

    def detect_icons(self, minimap_frame: np.ndarray) -> list[RawIconDetection]:
        gray = cv2.cvtColor(minimap_frame, cv2.COLOR_RGB2GRAY)
        circles = cv2.HoughCircles(
            gray,
            cv2.HOUGH_GRADIENT,
            dp=1.2,
            minDist=config.HOUGH_MIN_DIST,
            param1=config.HOUGH_PARAM1,
            param2=config.HOUGH_PARAM2,
            minRadius=config.HOUGH_MIN_RADIUS,
            maxRadius=config.HOUGH_MAX_RADIUS,
        )
        if circles is None:
            return []

        detections: list[RawIconDetection] = []
        rounded_circles = np.round(circles[0]).astype(int)
        rounded_circles = sorted(rounded_circles, key=lambda item: int(item[2]), reverse=True)
        for idx, circle in enumerate(rounded_circles[: config.MINIMAP_MAX_CIRCLES_PER_FRAME]):
            x, y, radius = int(circle[0]), int(circle[1]), int(circle[2])
            x1, y1 = max(0, x - radius), max(0, y - radius)
            x2, y2 = min(minimap_frame.shape[1], x + radius), min(minimap_frame.shape[0], y + radius)
            roi = minimap_frame[y1:y2, x1:x2]
            if roi.size == 0:
                continue
            team = self.classify_team(roi)
            interior = roi.copy()
            if min(interior.shape[:2]) > 30:
                margin = max(1, int(radius * 0.35))
                interior = interior[margin:-margin, margin:-margin]
            roi_gray = cv2.cvtColor(cv2.resize(interior, (90, 90)), cv2.COLOR_RGB2GRAY)
            best, score = self.match_champion(roi_gray)
            if score >= config.TEMPLATE_MATCH_CONFIRM:
                champion, uncertain = best, False
            elif score >= config.TEMPLATE_MATCH_UNCERTAIN:
                champion, uncertain = best, True
            else:
                champion, uncertain = f"unknown_champion_{idx}", True
            detections.append(RawIconDetection((x, y), radius, team, champion, score, uncertain))
        return detections

    def aggregate_detections(
        self,
        detections_per_frame: list[list[RawIconDetection]],
        timestamps: np.ndarray,
        fight_start: float,
        fight_end: float,
        player_positions: list[Optional[tuple[float, float]]],
    ) -> FightParticipants:
        tracks: list[list[RawIconDetection]] = []
        last_points: list[tuple[int, int]] = []
        for frame_detections, timestamp in zip(detections_per_frame, timestamps):
            if not fight_start <= float(timestamp) <= fight_end:
                continue
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
            unknown = ChampionResult("unknown_champion_0", 0.0, "ally", (0.5, 0.5), True)
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

        valid_player_positions = [p for p in player_positions if p is not None]
        if not valid_player_positions:
            flags.append("player_pos_unknown")
            player_idx = next((i for i, r in enumerate(results) if r.team == "ally"), 0)
        else:
            median_player = np.median(np.array(valid_player_positions), axis=0)
            ally_indices = [i for i, r in enumerate(results) if r.team == "ally"] or list(range(len(results)))
            player_idx = min(
                ally_indices,
                key=lambda i: float(np.linalg.norm(np.array(results[i].mean_pos) - median_player)),
            )
            if len(valid_player_positions) / max(len(player_positions), 1) < 0.30:
                flags.append("player_pos_low_confidence")

        player = ChampionResult(
            results[player_idx].champion_name,
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


def map_pos_to_screen_hint(map_pos: tuple[float, float], frame_size: tuple[int, int]) -> tuple[float, float]:
    frame_w, frame_h = frame_size
    screen_x = frame_w * 0.5 + (map_pos[0] - 0.5) * frame_w * 0.6
    screen_y = frame_h * 0.5 + (map_pos[1] - 0.5) * frame_h * 0.6
    return (float(np.clip(screen_x, 0, frame_w)), float(np.clip(screen_y, 0, frame_h)))
