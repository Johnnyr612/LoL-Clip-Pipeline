from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


FEATURE_SIZE = 24
AUGMENTATIONS_PER_ICON = 36
RNG_SEED = 1337
MODEL_VERSION = 1

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ClassifierMatch:
    champion_name: str
    confidence: float
    margin: float


class MinimapIconClassifier:
    """Portrait-trained classifier using synthetic minimap-style augmentation.

    This is the practical first step of the PandaScore-style approach: generate
    many minimap-looking variants from every raw champion portrait, then classify
    minimap crops by nearest augmented portrait embedding.
    """

    def __init__(self, names: list[str], labels: np.ndarray, features: np.ndarray):
        self.names = names
        self.labels = labels.astype(np.int32)
        self.features = features.astype(np.float32)

    @classmethod
    def load_or_build(
        cls,
        images_dir: Path,
        cache_path: Path,
        augmentations_per_icon: int = AUGMENTATIONS_PER_ICON,
    ) -> MinimapIconClassifier:
        signature = _icons_signature(images_dir)
        try:
            if cache_path.exists():
                with np.load(cache_path, allow_pickle=False) as payload:
                    if (
                        int(payload["version"].item()) == MODEL_VERSION
                        and str(payload["signature"].item()) == signature
                    ):
                        names = [str(name) for name in payload["names"]]
                        return cls(names, payload["labels"], payload["features"])
        except Exception as exc:  # noqa: BLE001 - stale cache should not break detection.
            logger.warning("Ignoring minimap classifier cache: %s", exc)

        classifier = cls.from_icons_dir(images_dir, augmentations_per_icon)
        try:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                cache_path,
                version=np.asarray(MODEL_VERSION, dtype=np.int32),
                signature=np.asarray(signature),
                names=np.asarray(classifier.names),
                labels=classifier.labels,
                features=classifier.features,
            )
        except Exception as exc:  # noqa: BLE001 - cache is an optimization only.
            logger.warning("Unable to write minimap classifier cache: %s", exc)
        return classifier

    @classmethod
    def from_icons_dir(
        cls,
        images_dir: Path,
        augmentations_per_icon: int = AUGMENTATIONS_PER_ICON,
    ) -> MinimapIconClassifier:
        rng = np.random.default_rng(RNG_SEED)
        names: list[str] = []
        labels: list[int] = []
        features: list[np.ndarray] = []
        for class_idx, path in enumerate(sorted(Path(images_dir).glob("*.png"))):
            with Image.open(path) as image:
                rgb = np.array(image.convert("RGB"))
            names.append(path.stem)
            variants = [_prepare_icon(rgb)]
            variants.extend(_augment_icon(rgb, rng) for _ in range(augmentations_per_icon))
            for variant in variants:
                features.append(_feature_vector(variant))
                labels.append(class_idx)
        if not features:
            return cls(
                [],
                np.empty((0,), dtype=np.int32),
                np.empty((0, FEATURE_SIZE * FEATURE_SIZE * 3), dtype=np.float32),
            )
        return cls(names, np.asarray(labels, dtype=np.int32), np.vstack(features))

    def match(self, roi_rgb: np.ndarray) -> ClassifierMatch:
        if self.features.size == 0 or roi_rgb.size == 0:
            return ClassifierMatch("unknown", -1.0, 0.0)
        feature = _feature_vector(roi_rgb)
        similarities = self.features @ feature
        class_scores = np.full(len(self.names), -1.0, dtype=np.float32)
        np.maximum.at(class_scores, self.labels, similarities)
        if class_scores.size == 0:
            return ClassifierMatch("unknown", -1.0, 0.0)
        top_indices = np.argsort(class_scores)[-2:]
        best_idx = int(top_indices[-1])
        best_score = float(class_scores[best_idx])
        runner_up = float(class_scores[int(top_indices[-2])]) if len(top_indices) > 1 else -1.0
        margin = best_score - runner_up
        confidence = float(
            np.clip(
                ((best_score + 1.0) / 2.0) * 0.75
                + np.clip(margin / 0.22, 0.0, 1.0) * 0.25,
                0.0,
                1.0,
            )
        )
        return ClassifierMatch(self.names[best_idx], confidence, margin)


def _feature_vector(rgb: np.ndarray) -> np.ndarray:
    prepared = _prepare_icon(rgb)
    lab = cv2.cvtColor(prepared, cv2.COLOR_RGB2LAB).astype(np.float32)
    lab[:, :, 0] = cv2.equalizeHist(lab[:, :, 0].astype(np.uint8)).astype(np.float32)
    resized = cv2.resize(lab, (FEATURE_SIZE, FEATURE_SIZE), interpolation=cv2.INTER_AREA)
    vector = resized.reshape(-1)
    vector -= float(vector.mean())
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-6:
        return np.zeros_like(vector, dtype=np.float32)
    return (vector / norm).astype(np.float32)


def _icons_signature(images_dir: Path) -> str:
    records: list[tuple[str, int, int]] = []
    for path in sorted(Path(images_dir).glob("*.png")):
        stat = path.stat()
        records.append((path.name, stat.st_size, stat.st_mtime_ns))
    return json.dumps({"version": MODEL_VERSION, "icons": records}, separators=(",", ":"))


def _prepare_icon(rgb: np.ndarray) -> np.ndarray:
    square = _center_square(rgb)
    return cv2.resize(square, (120, 120), interpolation=cv2.INTER_AREA)


def _center_square(rgb: np.ndarray) -> np.ndarray:
    h, w = rgb.shape[:2]
    side = min(h, w)
    y = max(0, (h - side) // 2)
    x = max(0, (w - side) // 2)
    return rgb[y : y + side, x : x + side]


def _augment_icon(rgb: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    out = _prepare_icon(rgb)
    out = _scale_rotate(out, rng)
    out = _color_jitter(out, rng)
    out = _minimap_ring(out, rng)
    out = _artifact_mask(out, rng)
    out = _degrade(out, rng)
    return out


def _scale_rotate(rgb: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    h, w = rgb.shape[:2]
    scale = float(rng.uniform(0.72, 1.16))
    angle = float(rng.uniform(-12.0, 12.0))
    matrix = cv2.getRotationMatrix2D((w / 2, h / 2), angle, scale)
    return cv2.warpAffine(rgb, matrix, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)


def _color_jitter(rgb: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV).astype(np.float32)
    hsv[:, :, 1] *= float(rng.uniform(0.65, 1.35))
    hsv[:, :, 2] *= float(rng.uniform(0.55, 1.25))
    hsv[:, :, 0] += float(rng.uniform(-4.0, 4.0))
    hsv[:, :, 0] = np.mod(hsv[:, :, 0], 180)
    hsv[:, :, 1:] = np.clip(hsv[:, :, 1:], 0, 255)
    out = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
    contrast = float(rng.uniform(0.75, 1.30))
    bias = float(rng.uniform(-18, 18))
    return np.clip(out.astype(np.float32) * contrast + bias, 0, 255).astype(np.uint8)


def _minimap_ring(rgb: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    out = rgb.copy()
    colors = [(215, 40, 36), (48, 112, 235), (235, 235, 235)]
    color = colors[int(rng.integers(0, len(colors)))]
    thickness = int(rng.integers(7, 15))
    radius = int(rng.integers(50, 60))
    cv2.circle(out, (60, 60), radius, color, thickness=thickness, lineType=cv2.LINE_AA)
    return out


def _artifact_mask(rgb: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    out = rgb.copy()
    overlay = out.copy()
    for _ in range(int(rng.integers(0, 4))):
        color = tuple(int(v) for v in rng.integers(60, 256, size=3))
        center = tuple(int(v) for v in rng.integers(15, 105, size=2))
        radius = int(rng.integers(4, 20))
        cv2.circle(overlay, center, radius, color, thickness=-1, lineType=cv2.LINE_AA)
    if rng.random() < 0.35:
        x1, y1 = (int(v) for v in rng.integers(0, 120, size=2))
        x2, y2 = (int(v) for v in rng.integers(0, 120, size=2))
        cv2.line(
            overlay,
            (x1, y1),
            (x2, y2),
            (90, 220, 255),
            thickness=int(rng.integers(1, 4)),
            lineType=cv2.LINE_AA,
        )
    alpha = float(rng.uniform(0.05, 0.28))
    out = cv2.addWeighted(overlay, alpha, out, 1.0 - alpha, 0)
    return out


def _degrade(rgb: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    out = rgb.copy()
    if rng.random() < 0.65:
        sigma = float(rng.uniform(0.2, 1.15))
        out = cv2.GaussianBlur(out, (0, 0), sigmaX=sigma)
    noise = rng.normal(0, float(rng.uniform(1.0, 8.0)), size=out.shape)
    out = np.clip(out.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    if rng.random() < 0.45:
        quality = int(rng.integers(45, 88))
        ok, encoded = cv2.imencode(
            ".jpg",
            cv2.cvtColor(out, cv2.COLOR_RGB2BGR),
            [int(cv2.IMWRITE_JPEG_QUALITY), quality],
        )
        if ok:
            decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
            if decoded is not None:
                out = cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB)
    return out
