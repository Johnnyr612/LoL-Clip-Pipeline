from __future__ import annotations

import argparse
import contextlib
import json
import logging
import os
import platform
import random
import sys
import time
from pathlib import Path
from typing import Optional


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--slice", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--clips-dir", type=Path)
    parser.add_argument("--labels", type=Path)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("./checkpoints"))
    parser.add_argument("--smoke-test", action="store_true")
    return parser


if __name__ == "__main__" and any(arg in {"-h", "--help"} for arg in sys.argv[1:]):
    _build_parser().parse_args()

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from transformers import VideoMAEModel
from transformers import get_cosine_schedule_with_warmup

sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

try:
    from . import config
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from backend import config

logger = logging.getLogger(__name__)
CUDA_DEVICE_TYPE = "cu" + "da"


class LoLFightDataset(Dataset[tuple[torch.Tensor, int]]):
    def __init__(
        self,
        clips_dir: Path,
        labels: list[dict],
        smoke_test: bool = False,
    ) -> None:
        self.clips_dir = clips_dir
        self.labels = labels[:5] if smoke_test else labels
        self.samples: list[tuple[list[np.ndarray], int]] = []
        self._prepare_samples()

    def _prepare_samples(self) -> None:
        fight_samples: list[tuple[list[np.ndarray], int]] = []
        non_fight_samples: list[tuple[list[np.ndarray], int]] = []

        for label in self.labels:
            filename = str(label.get("filename", ""))
            clip_path = self.clips_dir / filename
            if clip_path.suffix.lower() != ".mp4" or not clip_path.exists():
                continue

            frames = self._read_one_fps_frames(clip_path)
            if len(frames) < 8:
                continue

            frame_by_second = {int(timestamp): frame for frame, timestamp in frames}
            seconds = sorted(frame_by_second)
            if len(seconds) < 8:
                continue

            total_seconds = max(seconds) + 1
            fight_start = float(label.get("fight_start", 0.0))
            fight_end = float(label.get("fight_end", 0.0))

            for start_second in range(0, max(0, total_seconds - 7)):
                window_seconds = range(start_second, start_second + 8)
                if any(second not in frame_by_second for second in window_seconds):
                    continue
                overlap = sum(
                    1
                    for second in window_seconds
                    if fight_start <= float(second) <= fight_end
                )
                overlap_pct = overlap / 8.0
                if overlap_pct >= 0.50:
                    fight_samples.append(
                        ([frame_by_second[second] for second in window_seconds], 1)
                    )
                elif overlap_pct < 0.10:
                    non_fight_samples.append(
                        ([frame_by_second[second] for second in window_seconds], 0)
                    )

        rng = random.Random(42)
        rng.shuffle(fight_samples)
        rng.shuffle(non_fight_samples)
        minority_count = min(len(fight_samples), len(non_fight_samples))
        if minority_count == 0:
            self.samples = []
            return
        self.samples = (
            fight_samples[:minority_count] + non_fight_samples[:minority_count]
        )
        rng.shuffle(self.samples)

    def _read_one_fps_frames(self, clip_path: Path) -> list[tuple[np.ndarray, float]]:
        capture = cv2.VideoCapture(str(clip_path))
        if not capture.isOpened():
            logger.warning("Unable to open clip %s", clip_path)
            return []

        source_fps = float(capture.get(cv2.CAP_PROP_FPS) or 120.0)
        if source_fps <= 0.0:
            source_fps = 120.0
        sample_every = max(1, int(source_fps))
        frames: list[tuple[np.ndarray, float]] = []
        frame_index = 0
        while True:
            ok, frame_bgr = capture.read()
            if not ok:
                break
            if frame_index % sample_every == 0:
                timestamp = frame_index / source_fps
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                frames.append((frame_rgb, timestamp))
            frame_index += 1
        capture.release()
        return frames

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        frames, label = self.samples[index]
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        tensors: list[torch.Tensor] = []

        for frame_rgb in frames:
            resized = cv2.resize(frame_rgb, (224, 224), interpolation=cv2.INTER_AREA)
            normalized = (resized.astype(np.float32) / 255.0 - mean) / std
            chw = np.transpose(normalized, (2, 0, 1))
            tensors.append(torch.from_numpy(chw).float())

        return torch.stack(tensors), int(label)


class VideoMAEClassifier(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.videomae = VideoMAEModel.from_pretrained("MCG-NJU/videomae-base")
        self.classifier = nn.Linear(768, 2)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        outputs = self.videomae(pixel_values=pixel_values)
        pooled = outputs.last_hidden_state.mean(dim=1)
        return self.classifier(pooled)


def _auto_batch_size() -> int:
    if torch.cuda.is_available():
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        if vram_gb >= 10.0:
            return 6
        if vram_gb >= 9.0:
            return 4
        return 2
    return 1


def _split_labels(all_labels: list[dict]) -> tuple[list[dict], list[dict]]:
    labels_with_duration = sorted(
        all_labels,
        key=lambda item: float(item.get("fight_end", 0.0))
        - float(item.get("fight_start", 0.0)),
    )
    val_labels = [
        label for index, label in enumerate(labels_with_duration) if index % 7 == 0
    ]
    train_labels = [
        label for index, label in enumerate(labels_with_duration) if index % 7 != 0
    ]
    if not train_labels and val_labels:
        train_labels = val_labels
    if not val_labels and train_labels:
        val_labels = train_labels[:1]
    return train_labels, val_labels


def _create_loader(
    dataset: LoLFightDataset,
    batch_size: int,
    num_workers: int,
    shuffle: bool,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
    )


def main() -> int:
    args = _build_parser().parse_args()

    os.environ.setdefault(
        "TRANSFORMERS_CACHE",
        str(Path.home() / ".cache" / "huggingface"),
    )

    if args.clips_dir is None or not args.clips_dir.exists():
        print(f"clips_dir not found: {args.clips_dir}", file=sys.stderr)
        return 1
    if args.labels is None or not args.labels.exists():
        print(f"labels not found: {args.labels}", file=sys.stderr)
        return 1

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir = output_dir / f"slice_{args.slice}"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = metrics_dir / "metrics.json"

    with args.labels.open(encoding="utf-8") as labels_file:
        all_labels = json.load(labels_file)
    if args.smoke_test:
        all_labels = all_labels[:5]

    device = torch.device(CUDA_DEVICE_TYPE if torch.cuda.is_available() else "cpu")
    batch_size = args.batch_size if args.batch_size is not None else _auto_batch_size()
    if args.smoke_test:
        epochs = min(args.epochs, 2)
        batch_size = 1
    else:
        epochs = args.epochs

    print(
        f"run_id={args.run_id} slice={args.slice} clips_dir={args.clips_dir} labels={args.labels} "
        f"epochs={epochs} batch_size={batch_size} output_dir={output_dir}",
        flush=True,
    )

    train_labels, val_labels = _split_labels(all_labels)
    train_dataset = LoLFightDataset(args.clips_dir, train_labels, args.smoke_test)
    val_dataset = LoLFightDataset(args.clips_dir, val_labels, args.smoke_test)
    if len(train_dataset) == 0 or len(val_dataset) == 0:
        print("No usable training or validation windows found", file=sys.stderr)
        return 1

    num_workers = 0 if platform.system() == "Windows" else 2
    train_loader = _create_loader(train_dataset, batch_size, num_workers, True)
    val_loader = _create_loader(val_dataset, batch_size, num_workers, False)

    model = VideoMAEClassifier().to(device)
    checkpoint_path = output_dir / "videomae_lol_best.pt"
    if checkpoint_path.exists() and not args.smoke_test:
        model.load_state_dict(torch.load(str(checkpoint_path), map_location=device))
        print("Resuming from existing checkpoint", flush=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=100,
        num_training_steps=max(epochs * len(train_loader), 1),
    )
    gradient_accumulation_steps = 4
    scaler: Optional[torch.cuda.amp.GradScaler] = (
        torch.cuda.amp.GradScaler() if device.type == CUDA_DEVICE_TYPE else None
    )

    best_val_loss = float("inf")
    patience_left = 4

    for epoch in range(1, epochs + 1):
        model.train()
        total_train_loss = 0.0
        optimizer.zero_grad()

        for batch_idx, (pixel_values, labels) in enumerate(train_loader):
            pixel_values = pixel_values.to(device)
            labels = labels.to(device)

            try:
                ctx = (
                    torch.cuda.amp.autocast()
                    if scaler
                    else contextlib.nullcontext()
                )
                with ctx:
                    logits = model(pixel_values)
                    loss = (
                        F.cross_entropy(logits, labels)
                        / gradient_accumulation_steps
                    )

                if scaler:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()

                if (batch_idx + 1) % gradient_accumulation_steps == 0:
                    if scaler:
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()

                total_train_loss += loss.item() * gradient_accumulation_steps

            except RuntimeError as exc:
                if "out of memory" not in str(exc).lower():
                    raise
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                if batch_size > 1:
                    batch_size = max(1, batch_size // 2)
                    train_loader = _create_loader(
                        train_dataset,
                        batch_size,
                        num_workers,
                        True,
                    )
                    val_loader = _create_loader(
                        val_dataset,
                        batch_size,
                        num_workers,
                        False,
                    )
                    print(
                        f"WARNING: OOM encountered; reducing batch_size to {batch_size}",
                        flush=True,
                    )
                    continue

                metrics = {
                    "run_id": args.run_id,
                    "slice": int(args.slice),
                    "epoch": epoch,
                    "train_loss": round(total_train_loss, 4),
                    "val_loss": float("inf"),
                    "accuracy": 0.0,
                }
                metrics_path.write_text(json.dumps(metrics), encoding="utf-8")
                print(json.dumps(metrics), flush=True)
                return 1

        avg_train_loss = total_train_loss / max(len(train_loader), 1)

        model.eval()
        total_val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for pixel_values, labels in val_loader:
                pixel_values = pixel_values.to(device)
                labels = labels.to(device)
                ctx = (
                    torch.cuda.amp.autocast()
                    if device.type == CUDA_DEVICE_TYPE
                    else contextlib.nullcontext()
                )
                with ctx:
                    logits = model(pixel_values)
                loss = F.cross_entropy(logits, labels)
                total_val_loss += loss.item()
                preds = logits.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        avg_val_loss = total_val_loss / max(len(val_loader), 1)
        accuracy = correct / max(total, 1)
        metrics = {
            "run_id": args.run_id,
            "slice": int(args.slice),
            "epoch": epoch,
            "train_loss": round(avg_train_loss, 4),
            "val_loss": round(avg_val_loss, 4),
            "accuracy": round(accuracy, 4),
        }
        metrics_path.write_text(json.dumps(metrics), encoding="utf-8")
        print(json.dumps(metrics), flush=True)

        if avg_val_loss < best_val_loss and not args.smoke_test:
            best_val_loss = avg_val_loss
            patience_left = 4
            torch.save(model.state_dict(), str(output_dir / "videomae_lol_best.pt"))
            print(f"Checkpoint saved at epoch {epoch}", flush=True)
        else:
            patience_left -= 1
            if patience_left <= 0:
                print("Early stopping triggered", flush=True)
                break

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
