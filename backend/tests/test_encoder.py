from __future__ import annotations

from pathlib import Path

from backend import config
from backend.encoder import VideoEncoder, _step_crop_expression


def test_step_crop_expression_escapes_ffmpeg_commas_without_sliding():
    expr = _step_crop_expression(
        [(100, 0, 810, 1080), (300, 0, 810, 1080)],
        [10.0, 12.0],
        10.0,
        12.0,
    )

    assert r"\," in expr
    assert "+(" not in expr
    assert "if(lt(t\\,2)" in expr


def test_encoder_uses_seekable_mp4_options(monkeypatch, tmp_path):
    captured: dict[str, list[str]] = {}
    monkeypatch.setattr("backend.encoder.shutil.which", lambda _name: "ffmpeg")
    monkeypatch.setattr(config, "TEMP_DIR", tmp_path / "temp")
    monkeypatch.setattr(config, "OUTPUT_DIR", tmp_path / "outputs")

    def fake_run(args: list[str]) -> None:
        captured["args"] = args

    monkeypatch.setattr("backend.encoder._run_ffmpeg", fake_run)

    output = VideoEncoder().encode(
        "job",
        Path("source.mp4"),
        0.0,
        2.0,
        [(100, 0, 810, 1080)],
        [0.0],
    )

    assert output.name == "job_final.mp4"
    assert "-movflags" in captured["args"]
    assert "+faststart" in captured["args"]
    assert "-g" in captured["args"]
    assert str(config.OUTPUT_FPS) in captured["args"]
