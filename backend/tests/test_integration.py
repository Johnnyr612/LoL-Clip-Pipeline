from __future__ import annotations

import asyncio
import json
import shutil
import subprocess
from fractions import Fraction
from pathlib import Path

import aiosqlite
import pytest

from backend import config
from backend.media_probe import probe_media_profile
from backend.pipeline import ClipPipeline


def test_sample_clip_pipeline_stages_1_to_6(tmp_path, monkeypatch):
    asyncio.run(_run_sample_clip_pipeline_stages_1_to_6(tmp_path, monkeypatch))


async def _run_sample_clip_pipeline_stages_1_to_6(tmp_path, monkeypatch):
    if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
        pytest.skip("ffmpeg/ffprobe not installed")

    source = config.PROJECT_ROOT / "TestClip.mp4"
    if not source.is_file():
        pytest.skip("TestClip.mp4 not found")
    checkpoint = _highlight_checkpoint()
    source_profile = probe_media_profile(source)

    output_dir = tmp_path / "outputs"
    temp_dir = tmp_path / "temp"
    output_dir.mkdir()
    temp_dir.mkdir()
    monkeypatch.setattr(config, "OUTPUT_DIR", output_dir)
    monkeypatch.setattr(config, "TEMP_DIR", temp_dir)

    db_path = tmp_path / "test.sqlite3"
    pipeline = ClipPipeline(db_path)
    job_id = await pipeline.run(source, "sample_clip", highlight_checkpoint_path=checkpoint)
    async with aiosqlite.connect(db_path) as db:
        db.row_factory = aiosqlite.Row
        row = await (await db.execute("SELECT * FROM jobs WHERE id=?", (job_id,))).fetchone()
    assert row["status"] == "complete", row["error_detail"]
    assert row["stage_failed"] is None
    assert row["output_path"]

    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height,r_frame_rate", "-of", "json", row["output_path"]],
        check=True,
        capture_output=True,
        text=True,
    )
    stream = json.loads(probe.stdout)["streams"][0]
    assert stream["width"] == 1080
    assert stream["height"] == 1440
    assert float(Fraction(stream["r_frame_rate"])) == pytest.approx(source_profile.fps, abs=0.01)


def _highlight_checkpoint() -> Path:
    checkpoints = sorted((config.PROJECT_ROOT / "checkpoints").glob("videomae_lol_highlight_editor*.pt"))
    if not checkpoints:
        pytest.skip("highlight editor checkpoint not found")
    return checkpoints[0]
