import asyncio
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import torch

from backend import config, main, vjepa_runtime
from backend.fight_detector import FightDetector, HighlightEditorError, TrimResult, TrimSettings, apply_highlight_trim_settings


def test_discovery_preserves_videomae_and_adds_vjepa(tmp_path, monkeypatch):
    root = tmp_path / 'checkpoints'
    root.mkdir()
    for name in ['videomae_lol_highlight_old.pt', 'vjepa21_highlight_best_10ep.pt', 'minimap.pt']:
        (root / name).touch()
    monkeypatch.setattr(config, 'PROJECT_ROOT', tmp_path)
    result = asyncio.run(main.list_highlight_checkpoints())
    assert {r['filename'] for r in result['checkpoints']} == {'videomae_lol_highlight_old.pt', 'vjepa21_highlight_best_10ep.pt'}
    assert main._resolve_highlight_checkpoint('vjepa21_highlight_best_10ep.pt') == root / 'vjepa21_highlight_best_10ep.pt'
    with pytest.raises(main.HTTPException):
        main._resolve_highlight_checkpoint('../outside.pt')


def test_vjepa_routing_requires_raw_video_and_preserves_model_only():
    detector = FightDetector()
    frames, times = np.empty((0, 224, 224, 3), dtype=np.uint8), np.empty(0)
    weights = Path('vjepa21_highlight_best_10ep.pt')
    with pytest.raises(HighlightEditorError, match='original MP4'):
        detector.predict_highlight_trim(frames, times, 60, weights)
    trim = TrimResult(21, 55, 25, 50, 25, [], ['vjepa21_highlight_editor'])
    with patch('backend.vjepa_detector.predict_trim', return_value=trim) as predict:
        assert detector.predict_highlight_trim(frames, times, 60, weights, Path('raw.mp4')) == trim
        predict.assert_called_once_with(Path('raw.mp4'), 60, weights)
    result = apply_highlight_trim_settings(trim, frames, times, 60, TrimSettings(model_only=True))
    assert (result.clip_start, result.clip_end) == (21, 55)


def test_vjepa_decoder_keeps_single_span_and_fractional_end():
    edges = vjepa_runtime.timeline_edges(10.25, 1)
    logits = torch.zeros(11, 4)
    logits[:, 0] = -10
    logits[5:, 0] = 10
    assert vjepa_runtime.decode_highlight(logits, edges) == (5, 10.25)
    logits[:, 0] = -10
    assert vjepa_runtime.decode_highlight(logits, edges) is None
