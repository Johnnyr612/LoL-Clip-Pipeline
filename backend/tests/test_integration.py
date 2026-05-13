from __future__ import annotations

import json
import shutil
import subprocess

import aiosqlite
import pytest

from backend.pipeline import ClipPipeline


@pytest.mark.asyncio
async def test_synthetic_pipeline_stages_1_to_6(tmp_path):
    if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
        pytest.skip("ffmpeg/ffprobe not installed")

    source = tmp_path / "synthetic.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "color=c=green:s=1920x1080:r=120:d=60",
            "-f",
            "lavfi",
            "-i",
            "anullsrc=channel_layout=mono:sample_rate=16000",
            "-vf",
            "drawbox=x=500:y=120:w=500:h=180:color=red:t=fill:enable='between(t,15,30)'",
            "-shortest",
            str(source),
        ],
        check=True,
        capture_output=True,
    )

    db_path = tmp_path / "test.sqlite3"
    pipeline = ClipPipeline(db_path)
    job_id = await pipeline.run(source, "synthetic")
    output = tmp_path / "dummy"
    async with aiosqlite.connect(db_path) as db:
        db.row_factory = aiosqlite.Row
        row = await (await db.execute("SELECT * FROM jobs WHERE id=?", (job_id,))).fetchone()
    assert row["status"] == "complete"
    assert row["stage_failed"] is None
    captions = json.loads(row["captions"])
    assert {"caption", "hashtags", "hook_line"} <= captions["default"].keys()
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
    assert stream["r_frame_rate"] == "60/1"
