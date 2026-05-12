from __future__ import annotations

import shutil

import cv2
import numpy as np
import pytest
from PIL import Image

from backend import config
from backend.minimap_detector import MinimapDetector, RawIconDetection


@pytest.fixture(scope="module")
def detector() -> MinimapDetector:
    return MinimapDetector(config.MINIMAP_ICONS_DIR, config.MANIFEST_PATH)


def test_load_manifest(detector: MinimapDetector):
    assert len(detector.name_to_key) >= 172
    assert detector.name_to_key["Wukong"] == "MonkeyKing"


def test_load_icons(detector: MinimapDetector):
    assert len(detector.templates) >= 170
    assert "Aatrox" in detector.templates


def test_minimap_boundary_detection(detector: MinimapDetector):
    detector._minimap_rect = None
    frame = np.full((1080, 1920, 3), 120, dtype=np.uint8)
    cv2.rectangle(frame, (1500, 720), (1830, 1050), (5, 5, 5), thickness=-1)
    x, y, w, h = detector.locate_minimap(frame)
    assert abs(x - 1500) <= 5
    assert abs(y - 720) <= 5
    assert abs(w - 331) <= 5
    assert abs(h - 331) <= 5


def test_white_box_detection(detector: MinimapDetector):
    minimap = np.zeros((540, 345, 3), dtype=np.uint8)
    cv2.rectangle(minimap, (100, 120), (140, 160), (255, 255, 255), thickness=-1)
    centroid = detector.find_white_box(minimap)
    assert centroid is not None
    assert abs(centroid[0] * 345 - 120) <= 3
    assert abs(centroid[1] * 540 - 140) <= 3


def _circle(color_rgb: tuple[int, int, int]) -> np.ndarray:
    frame = np.zeros((48, 48, 3), dtype=np.uint8)
    cv2.circle(frame, (24, 24), 20, color_rgb, thickness=6)
    return frame


def test_circle_border_red(detector: MinimapDetector):
    assert detector.classify_team(_circle((220, 20, 20))) == "enemy"


def test_circle_border_blue(detector: MinimapDetector):
    assert detector.classify_team(_circle((20, 80, 230))) == "ally"


def test_template_match_aatrox(detector: MinimapDetector):
    image = np.array(Image.open(config.MINIMAP_ICONS_DIR / "images" / "Aatrox.png").convert("RGB"))
    blurred = cv2.GaussianBlur(image, (0, 0), sigmaX=1)
    gray = detector._center_crop_gray(blurred)
    name, score = detector.match_champion(gray)
    assert name == "Aatrox"
    assert score > 0.55


def test_template_match_unknown(detector: MinimapDetector):
    rng = np.random.default_rng(123)
    noise = rng.integers(0, 255, size=(90, 90), dtype=np.uint8)
    _, score = detector.match_champion(noise)
    assert score < 0.35


def test_vote_aggregation(detector: MinimapDetector):
    detections = []
    for idx in range(10):
        name = "Aatrox" if idx < 8 else "Ahri"
        detections.append([RawIconDetection((50, 50), 12, "ally", name, 0.9, False)])
    result = detector.aggregate_detections(detections, np.arange(10), 0, 9, [(50, 50)] * 10)
    assert result.player.champion_name == "Aatrox"


def test_aggregation_dedupes_champion_tracks(detector: MinimapDetector):
    detections = [
        [
            RawIconDetection((50, 50), 12, "ally", "Aatrox", 0.9, False),
            RawIconDetection((120, 120), 12, "ally", "Aatrox", 0.8, False),
            RawIconDetection((100, 100), 12, "enemy", "Ahri", 0.9, False),
            RawIconDetection((115, 115), 12, "enemy", "Ahri", 0.8, False),
        ]
    ]
    result = detector.aggregate_detections(detections, np.array([0]), 0, 1, [(50, 50)])
    assert result.fight_type == "1v1"
    assert [enemy.champion_name for enemy in result.enemies] == ["Ahri"]


def test_aggregation_filters_champions_far_from_fight(detector: MinimapDetector):
    detections = [
        [
            RawIconDetection((50, 50), 12, "ally", "Aatrox", 0.9, False),
            RawIconDetection((100, 100), 12, "enemy", "Ahri", 0.9, False),
            RawIconDetection((300, 250), 12, "enemy", "Thresh", 0.9, False),
        ]
    ]
    result = detector.aggregate_detections(detections, np.array([0]), 0, 1, [(50, 50)])
    assert [enemy.champion_name for enemy in result.enemies] == ["Ahri"]


def test_player_hud_detection(detector: MinimapDetector):
    frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
    icon = np.array(Image.open(config.MINIMAP_ICONS_DIR / "images" / "Aatrox.png").convert("RGB"))
    frame[972:1080, 604:710] = cv2.resize(icon, (106, 108), interpolation=cv2.INTER_AREA)
    name, score = detector.detect_player_hud_champion(frame)
    assert name == "Aatrox"
    assert score > 0.45


def test_augmentation_count(detector: MinimapDetector):
    assert len(detector.augmented["Aatrox"]) == 36


def test_corrupt_icon_skipped(tmp_path):
    icons_dir = tmp_path / "minimap_icons"
    images_dir = icons_dir / "images"
    images_dir.mkdir(parents=True)
    shutil.copy(config.MANIFEST_PATH, icons_dir / "champions_manifest.json")
    (images_dir / "Fake.png").write_bytes(b"")
    detector = MinimapDetector(icons_dir, icons_dir / "champions_manifest.json")
    assert "Fake" not in detector.templates
