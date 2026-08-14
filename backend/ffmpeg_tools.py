from __future__ import annotations

import shutil
from pathlib import Path

from . import config


def find_executable(name: str) -> str | None:
    found = shutil.which(name)
    if found:
        return found
    suffix = ".exe" if not name.lower().endswith(".exe") else ""
    local = config.PROJECT_ROOT / "tools" / "ffmpeg" / "bin" / f"{name}{suffix}"
    return str(local) if local.is_file() else None


def require_executable(name: str) -> str:
    found = find_executable(name)
    if found is None:
        raise FileNotFoundError(f"{name} executable not found on PATH or in tools/ffmpeg/bin")
    return found
