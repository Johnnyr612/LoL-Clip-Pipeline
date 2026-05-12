from __future__ import annotations

from backend.encoder import _linear_crop_expression


def test_linear_crop_expression_escapes_ffmpeg_commas():
    expr = _linear_crop_expression(
        [(100, 0, 810, 1080), (300, 0, 810, 1080)],
        [10.0, 12.0],
        10.0,
        12.0,
    )

    assert r"\," in expr
    assert "(100)*(t-0)" in expr
