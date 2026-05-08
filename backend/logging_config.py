from __future__ import annotations

import logging
from logging.handlers import TimedRotatingFileHandler

from . import config


def setup_logging() -> None:
    config.LOG_DIR.mkdir(parents=True, exist_ok=True)
    handler = TimedRotatingFileHandler(
        config.LOG_DIR / "lol_clip_app.log",
        when="midnight",
        interval=1,
        backupCount=7,
        encoding="utf-8",
    )
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s [%(name)s] %(message)s"))
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    if not any(isinstance(existing, TimedRotatingFileHandler) for existing in root.handlers):
        root.addHandler(handler)
