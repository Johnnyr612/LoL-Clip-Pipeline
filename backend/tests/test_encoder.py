from __future__ import annotations

from pathlib import Path

from backend import config
from backend.encoder import VideoEncoder, _step_crop_expression, _video_encode_args
from backend.media_probe import MediaProfile


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
    assert "-pix_fmt" in captured["args"]
    assert "yuv420p" in captured["args"]
    assert config.FFMPEG_AUDIO_BITRATE in captured["args"]


def test_video_encode_args_use_crf_by_default(monkeypatch):
    monkeypatch.setattr(config, "FFMPEG_VIDEO_ENCODER", "libx264")
    monkeypatch.setattr(config, "FFMPEG_VIDEO_BITRATE", "")
    monkeypatch.setattr(config, "FFMPEG_CRF", 18)
    monkeypatch.setattr(config, "FFMPEG_PRESET", "slow")

    args = _video_encode_args()

    assert "-crf" in args
    assert "18" in args
    assert "-b:v" not in args
    assert "slow" in args


def test_video_encode_args_can_use_target_bitrate(monkeypatch):
    monkeypatch.setattr(config, "FFMPEG_VIDEO_ENCODER", "libx264")
    monkeypatch.setattr(config, "FFMPEG_VIDEO_BITRATE", "20M")
    monkeypatch.setattr(config, "FFMPEG_VIDEO_MAXRATE", "")
    monkeypatch.setattr(config, "FFMPEG_VIDEO_BUFSIZE", "")

    args = _video_encode_args()

    assert "-crf" not in args
    assert args[args.index("-b:v") + 1] == "20M"
    assert args[args.index("-maxrate") + 1] == "20M"
    assert args[args.index("-bufsize") + 1] == "20M"


def test_video_encode_args_can_use_nvenc(monkeypatch):
    monkeypatch.setattr(config, "FFMPEG_VIDEO_ENCODER", "h264_nvenc")
    monkeypatch.setattr(config, "FFMPEG_NVENC_PRESET", "p5")
    monkeypatch.setattr(config, "FFMPEG_NVENC_RC", "vbr")
    monkeypatch.setattr(config, "FFMPEG_NVENC_CQ", "")
    monkeypatch.setattr(config, "FFMPEG_VIDEO_BITRATE", "25M")
    monkeypatch.setattr(config, "FFMPEG_VIDEO_MAXRATE", "25M")
    monkeypatch.setattr(config, "FFMPEG_VIDEO_BUFSIZE", "50M")

    args = _video_encode_args()

    assert args[args.index("-c:v") + 1] == "h264_nvenc"
    assert args[args.index("-preset") + 1] == "p5"
    assert args[args.index("-rc") + 1] == "vbr"
    assert args[args.index("-b:v") + 1] == "25M"
    assert args[args.index("-bufsize") + 1] == "50M"


def test_video_encode_args_can_match_source_bitrate(monkeypatch):
    source_profile = MediaProfile(
        duration=36.0,
        has_audio=True,
        video_codec="h264",
        width=1920,
        height=1080,
        fps=59.94,
        fps_rate="60000/1001",
        video_bitrate=24_910_000,
    )
    monkeypatch.setattr(config, "FFMPEG_MATCH_SOURCE_ENCODING", True)
    monkeypatch.setattr(config, "FFMPEG_VIDEO_ENCODER", "h264_nvenc")
    monkeypatch.setattr(config, "FFMPEG_NVENC_PRESET", "p5")
    monkeypatch.setattr(config, "FFMPEG_NVENC_RC", "vbr")
    monkeypatch.setattr(config, "FFMPEG_NVENC_CQ", "")
    monkeypatch.setattr(config, "FFMPEG_VIDEO_BITRATE", "20M")
    monkeypatch.setattr(config, "FFMPEG_VIDEO_MAXRATE", "")
    monkeypatch.setattr(config, "FFMPEG_VIDEO_BUFSIZE", "")

    args = _video_encode_args(source_profile)

    assert args[args.index("-b:v") + 1] == "24.9M"
    assert args[args.index("-maxrate") + 1] == "24.9M"
    assert args[args.index("-bufsize") + 1] == "24.9M"
