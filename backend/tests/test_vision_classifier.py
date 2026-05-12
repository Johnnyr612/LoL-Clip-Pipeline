from __future__ import annotations

import numpy as np

from backend import config
from backend.vision_classifier import classify_fight_participants


def test_vision_classifier_disabled_without_api_key(monkeypatch):
    monkeypatch.setattr(config, "OPENAI_API_KEY", "")
    result = classify_fight_participants(np.zeros((1, 16, 16, 3), dtype=np.uint8), np.array([0], dtype=np.float32), 0, 1)
    assert result is None
