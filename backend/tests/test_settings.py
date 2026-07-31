from __future__ import annotations

from fastapi.testclient import TestClient

from backend import config
from backend.main import app


def test_settings_folders_lists_known_directories(tmp_path, monkeypatch) -> None:
    output_dir = tmp_path / "outputs"
    appdata_dir = tmp_path / "appdata"
    project_dir = tmp_path / "project"
    for path in (output_dir, appdata_dir / "uploads", appdata_dir / "logs", appdata_dir / "temp", project_dir / "checkpoints"):
        path.mkdir(parents=True)
    monkeypatch.setattr(config, "OUTPUT_DIR", output_dir)
    monkeypatch.setattr(config, "APPDATA_DIR", appdata_dir)
    monkeypatch.setattr(config, "LOG_DIR", appdata_dir / "logs")
    monkeypatch.setattr(config, "TEMP_DIR", appdata_dir / "temp")
    monkeypatch.setattr(config, "PROJECT_ROOT", project_dir)

    response = TestClient(app).get("/settings/folders")

    assert response.status_code == 200
    payload = response.json()
    keys = {folder["key"] for folder in payload["folders"]}
    assert {"output", "uploads", "project", "checkpoints", "logs", "temp"} <= keys
    assert all("path" in folder for folder in payload["folders"])


def test_open_settings_folder_rejects_missing_directory(tmp_path) -> None:
    response = TestClient(app).post("/settings/open-folder", json={"path": str(tmp_path / "missing")})

    assert response.status_code == 404


def test_open_settings_folder_opens_existing_directory(tmp_path, monkeypatch) -> None:
    opened: list[str] = []
    monkeypatch.setattr("backend.main._open_folder_in_file_manager", lambda path: opened.append(str(path)))

    response = TestClient(app).post("/settings/open-folder", json={"path": str(tmp_path)})

    assert response.status_code == 200
    assert response.json()["opened"] is True
    assert opened == [str(tmp_path.resolve())]
