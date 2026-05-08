from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
import uuid
from pathlib import Path
from typing import AsyncIterator

from . import config


class TrainingCoordinator:
    def __init__(self) -> None:
        self.run_id: str | None = None
        self.process: subprocess.Popen | None = None
        self.metrics_path: Path | None = None

    async def start(self) -> str:
        if self.process and self.process.poll() is None:
            return self.run_id or "running"
        self.run_id = uuid.uuid4().hex
        slice_dir = config.PROJECT_ROOT / "checkpoints" / "slice_0"
        slice_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_path = slice_dir / "metrics.json"
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = get_mig_device_uuid(0)
        self.process = subprocess.Popen(
            [sys.executable, str(config.PROJECT_ROOT / "backend" / "trainer_worker.py"), "--slice", "0", "--run-id", self.run_id],
            cwd=config.PROJECT_ROOT,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        return self.run_id

    async def stream(self) -> AsyncIterator[dict]:
        last_seen = ""
        while True:
            metric: dict = {"status": "idle"}
            if self.metrics_path and self.metrics_path.exists():
                text = self.metrics_path.read_text(encoding="utf-8")
                if text != last_seen:
                    last_seen = text
                    metric = json.loads(text)
            if self.process and self.process.poll() is not None:
                metric = metric | {"status": "complete" if self.process.returncode == 0 else "failed"}
                yield metric
                return
            yield metric
            await asyncio.sleep(30)


def get_mig_device_uuid(slice_index: int) -> str:
    # In production this can parse nvidia-smi -L. Defaulting to the numeric slice
    # keeps the worker launch deterministic in dev environments.
    return os.environ.get(f"MIG_SLICE_{slice_index}_UUID", str(slice_index))
