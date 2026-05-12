from __future__ import annotations

import numpy as np

from backend.minimap_classifier import MinimapIconClassifier


def test_minimap_icon_classifier_matches_augmented_portrait(tmp_path):
    images_dir = tmp_path / "images"
    images_dir.mkdir()
    red = np.zeros((120, 120, 3), dtype=np.uint8)
    red[:, :, 0] = 220
    blue = np.zeros((120, 120, 3), dtype=np.uint8)
    blue[:, :, 2] = 220

    import cv2

    cv2.imwrite(str(images_dir / "RedChamp.png"), cv2.cvtColor(red, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(images_dir / "BlueChamp.png"), cv2.cvtColor(blue, cv2.COLOR_RGB2BGR))

    classifier = MinimapIconClassifier.from_icons_dir(images_dir, augmentations_per_icon=3)
    match = classifier.match(red)

    assert match.champion_name == "RedChamp"
    assert match.confidence > 0.5


def test_minimap_icon_classifier_uses_cache(tmp_path):
    images_dir = tmp_path / "images"
    images_dir.mkdir()
    red = np.zeros((120, 120, 3), dtype=np.uint8)
    red[:, :, 0] = 220

    import cv2

    cv2.imwrite(str(images_dir / "RedChamp.png"), cv2.cvtColor(red, cv2.COLOR_RGB2BGR))
    cache_path = tmp_path / "classifier.npz"

    first = MinimapIconClassifier.load_or_build(images_dir, cache_path, augmentations_per_icon=1)
    second = MinimapIconClassifier.load_or_build(images_dir, cache_path, augmentations_per_icon=1)

    assert cache_path.exists()
    assert second.names == first.names
