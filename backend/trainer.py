from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import subprocess
import sys
import uuid
from pathlib import Path
from typing import AsyncIterator

sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

try:
    from . import config
except ImportError:  # Allows: python backend/trainer.py ...
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from backend import config

logger = logging.getLogger(__name__)


class TrainingCoordinator:
    def __init__(self) -> None:
        self.run_id: str | None = None
        self.process: subprocess.Popen | None = None
        self.metrics_path: Path | None = None

    async def start(
        self,
        clips_dir: Path | None = None,
        labels: Path | None = None,
        epochs: int = 25,
        batch_size: int = 4,
        output_dir: Path | None = None,
    ) -> str:
        if self.process and self.process.poll() is None:
            return self.run_id or "running"
        self.run_id = uuid.uuid4().hex
        resolved_output_dir = (output_dir or (config.PROJECT_ROOT / "checkpoints")).expanduser().resolve()
        slice_dir = resolved_output_dir / "slice_0"
        slice_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_path = slice_dir / "metrics.json"
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = get_mig_device_uuid(0)
        command = [
            sys.executable,
            str(config.PROJECT_ROOT / "backend" / "trainer_worker.py"),
            "--slice",
            "0",
            "--run-id",
            self.run_id,
            "--epochs",
            str(epochs),
            "--batch-size",
            str(batch_size),
            "--output-dir",
            str(resolved_output_dir),
        ]
        if clips_dir is not None:
            command.extend(["--clips-dir", str(clips_dir.expanduser().resolve())])
        if labels is not None:
            command.extend(["--labels", str(labels.expanduser().resolve())])
        self.process = subprocess.Popen(command, cwd=config.PROJECT_ROOT, env=env)
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch LoL clip fight-boundary training on a MIG slice.")
    parser.add_argument("--clips_dir", required=True, type=Path, help="Folder containing training .mp4 files.")
    parser.add_argument("--labels", required=True, type=Path, help="Path to labels JSON file.")
    parser.add_argument("--epochs", type=int, default=25, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size per step.")
    parser.add_argument("--output_dir", type=Path, default=Path("./checkpoints"), help="Where to save checkpoints.")
    return parser.parse_args()


def _configure_cli_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        stream=sys.stdout,
        force=True,
    )


def main() -> int:
    _configure_cli_logging()
    args = parse_args()
    clips_dir = args.clips_dir.expanduser().resolve()
    labels = args.labels.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()

    if not clips_dir.exists() or not clips_dir.is_dir():
        raise SystemExit(f"--clips_dir must be an existing directory: {clips_dir}")
    if not labels.exists() or not labels.is_file():
        raise SystemExit(f"--labels must be an existing JSON file: {labels}")

    run_id = uuid.uuid4().hex
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = get_mig_device_uuid(0)
    command = [
        sys.executable,
        str(config.PROJECT_ROOT / "backend" / "trainer_worker.py"),
        "--slice",
        "0",
        "--run-id",
        run_id,
        "--clips-dir",
        str(clips_dir),
        "--labels",
        str(labels),
        "--epochs",
        str(args.epochs),
        "--batch-size",
        str(args.batch_size),
        "--output-dir",
        str(output_dir),
    ]

    logger.info("Starting training run %s", run_id)
    logger.info("clips_dir=%s labels=%s epochs=%s batch_size=%s output_dir=%s", clips_dir, labels, args.epochs, args.batch_size, output_dir)
    process = subprocess.Popen(command, cwd=config.PROJECT_ROOT, env=env)
    return process.wait()


if __name__ == "__main__":
    raise SystemExit(main())
