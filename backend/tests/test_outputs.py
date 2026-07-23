from __future__ import annotations

from fastapi.testclient import TestClient

from backend import config
from backend.main import app


def test_output_mp4_supports_byte_range_requests(tmp_path, monkeypatch) -> None:
    output_dir = tmp_path / "outputs"
    output_dir.mkdir()
    video = output_dir / "sample.mp4"
    video.write_bytes(b"0123456789")
    monkeypatch.setattr(config, "OUTPUT_DIR", output_dir)

    response = TestClient(app).get("/outputs/sample.mp4", headers={"Range": "bytes=2-5"})

    assert response.status_code == 206
    assert response.headers["accept-ranges"] == "bytes"
    assert response.headers["content-range"] == "bytes 2-5/10"
    assert response.content == b"2345"


def test_output_file_rejects_paths_outside_output_dir(tmp_path, monkeypatch) -> None:
    output_dir = tmp_path / "outputs"
    output_dir.mkdir()
    outside = tmp_path / "outside.mp4"
    outside.write_bytes(b"not allowed")
    monkeypatch.setattr(config, "OUTPUT_DIR", output_dir)

    response = TestClient(app).get("/outputs/../outside.mp4")

    assert response.status_code == 404
