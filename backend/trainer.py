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
from typing import AsyncIterator, Optional

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
        batch_size: Optional[int] = None,
        output_dir: Path | None = None,
        freeze_backbone: bool = True,
        unfreeze_last_n_layers: int = 2,
        classifier_lr: float = 1e-4,
        backbone_lr: float = 1e-5,
        val_fraction: float = 0.15,
        progress_interval: int = 5,
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
            "--output-dir",
            str(resolved_output_dir),
            "--unfreeze-last-n-layers",
            str(unfreeze_last_n_layers),
            "--classifier-lr",
            str(classifier_lr),
            "--backbone-lr",
            str(backbone_lr),
            "--val-fraction",
            str(val_fraction),
            "--progress-interval",
            str(progress_interval),
        ]
        command.append("--freeze-backbone" if freeze_backbone else "--no-freeze-backbone")
        if batch_size is not None:
            command.extend(["--batch-size", str(batch_size)])
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
            await asyncio.sleep(2)


def get_mig_device_uuid(slice_index: int) -> str:
    # In production this can parse nvidia-smi -L. Defaulting to the numeric slice
    # keeps the worker launch deterministic in dev environments.
    return os.environ.get(f"MIG_SLICE_{slice_index}_UUID", str(slice_index))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch LoL clip fight-boundary training on a MIG slice.")
    parser.add_argument("--clips_dir", type=Path, default=None, help="Fallback folder containing training .mp4 files when labels omit raw_path.")
    parser.add_argument("--labels", required=True, type=Path, help="Path to labels JSON file.")
    parser.add_argument("--epochs", type=int, default=25, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size per step.")
    parser.add_argument("--output_dir", type=Path, default=Path("./checkpoints"), help="Where to save checkpoints.")
    parser.add_argument("--freeze_backbone", action=argparse.BooleanOptionalAction, default=True, help="Freeze VideoMAE except the selected final layers.")
    parser.add_argument("--unfreeze_last_n_layers", type=int, default=2, help="Number of final VideoMAE encoder layers to fine-tune when frozen.")
    parser.add_argument("--classifier_lr", type=float, default=1e-4, help="Learning rate for the classifier head.")
    parser.add_argument("--backbone_lr", type=float, default=1e-5, help="Learning rate for unfrozen VideoMAE layers.")
    parser.add_argument("--val_fraction", type=float, default=0.15, help="Fraction of source groups held out for validation.")
    parser.add_argument("--progress_interval", type=int, default=5, help="Batches between progress updates.")
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
    clips_dir = args.clips_dir.expanduser().resolve() if args.clips_dir is not None else None
    labels = args.labels.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()

    if clips_dir is not None and (not clips_dir.exists() or not clips_dir.is_dir()):
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
        "--labels",
        str(labels),
        "--epochs",
        str(args.epochs),
        "--batch-size",
        str(args.batch_size),
        "--output-dir",
        str(output_dir),
        "--unfreeze-last-n-layers",
        str(args.unfreeze_last_n_layers),
        "--classifier-lr",
        str(args.classifier_lr),
        "--backbone-lr",
        str(args.backbone_lr),
        "--val-fraction",
        str(args.val_fraction),
        "--progress-interval",
        str(args.progress_interval),
    ]
    command.append("--freeze-backbone" if args.freeze_backbone else "--no-freeze-backbone")
    if clips_dir is not None:
        command.extend(["--clips-dir", str(clips_dir)])

    logger.info("Starting training run %s", run_id)
    logger.info(
        "clips_dir=%s labels=%s epochs=%s batch_size=%s output_dir=%s freeze_backbone=%s unfreeze_last_n_layers=%s",
        clips_dir,
        labels,
        args.epochs,
        args.batch_size,
        output_dir,
        args.freeze_backbone,
        args.unfreeze_last_n_layers,
    )
    process = subprocess.Popen(command, cwd=config.PROJECT_ROOT, env=env)
    return process.wait()


if __name__ == "__main__":
    raise SystemExit(main())
