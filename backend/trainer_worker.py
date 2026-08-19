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
    parser.add_argument("--freeze-backbone", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--unfreeze-last-n-layers", type=int, default=2)
    parser.add_argument("--classifier-lr", type=float, default=1e-4)
    parser.add_argument("--backbone-lr", type=float, default=1e-5)
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--progress-interval", type=int, default=5)
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--task", choices=["highlight", "fight"], default="highlight")
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

sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

try:
    from . import config
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from backend import config

logger = logging.getLogger(__name__)
CUDA_DEVICE_TYPE = "cuda"
PRECOMPUTED_DIR = Path("precomputed")
WINDOW_SECONDS = 16
POSITIVE_OVERLAP_THRESHOLD = 0.50
NEGATIVE_OVERLAP_THRESHOLD = 0.10
HIGHLIGHT_CONTEXT_SECONDS = config.HIGHLIGHT_CONTEXT_SECONDS
HIGHLIGHT_INPUT_FRAMES = config.HIGHLIGHT_INPUT_FRAMES
PHASE_EXCLUDE = 0
PHASE_BUILDUP = 1
PHASE_FIGHT = 2
PHASE_PAYOFF = 3
PHASE_NAMES = ("exclude", "buildup", "fight", "payoff")
IGNORE_INDEX = -100


def _clean_segments(label: dict) -> list[tuple[float, float]]:
    raw_segments = label.get("fight_segments")
    if not raw_segments:
        raw_segments = [[label.get("fight_start", 0.0), label.get("fight_end", 0.0)]]

    segments: list[tuple[float, float]] = []
    for item in raw_segments:
        if not isinstance(item, (list, tuple)) or len(item) < 2:
            continue
        start = float(item[0])
        end = float(item[1])
        if end > start:
            segments.append((start, end))
    return sorted(segments)


def _window_overlap_pct(start_second: int, segments: list[tuple[float, float]]) -> float:
    window_start = float(start_second)
    window_end = window_start + WINDOW_SECONDS
    overlap = 0.0
    for fight_start, fight_end in segments:
        overlap += max(0.0, min(window_end, fight_end) - max(window_start, fight_start))
    return min(overlap / WINDOW_SECONDS, 1.0)


def _clip_bounds(label: dict, duration: float) -> tuple[float, float] | None:
    if "clip_start" not in label or "clip_end" not in label:
        return None
    clip_start = max(0.0, min(float(label.get("clip_start") or 0.0), duration))
    clip_end = max(clip_start, min(float(label.get("clip_end") or duration), duration))
    return clip_start, clip_end


def _duration_for_label(label: dict, fallback_end: float = 0.0) -> float:
    duration_value = (
        label.get("duration")
        or label.get("source_duration")
        or label.get("raw_duration")
        or label.get("clip_duration")
        or fallback_end
        or HIGHLIGHT_CONTEXT_SECONDS
    )
    return max(1.0, float(duration_value))


def _highlight_targets_for_label(label: dict) -> tuple[torch.Tensor, torch.Tensor]:
    fight_segments = _clean_segments(label)
    last_fight_end = max((end for _, end in fight_segments), default=0.0)
    duration = _duration_for_label(label, last_fight_end + WINDOW_SECONDS)
    clip_bounds = _clip_bounds(label, duration)
    if clip_bounds is None:
        return (
            torch.full((HIGHLIGHT_CONTEXT_SECONDS,), IGNORE_INDEX, dtype=torch.long),
            torch.full((HIGHLIGHT_CONTEXT_SECONDS,), IGNORE_INDEX, dtype=torch.long),
        )

    clip_start, clip_end = clip_bounds
    fight_start = min((start for start, _ in fight_segments), default=clip_start)
    fight_end = max((end for _, end in fight_segments), default=clip_end)
    fight_start = max(clip_start, min(float(fight_start), clip_end))
    fight_end = max(fight_start, min(float(fight_end), clip_end))

    include = torch.zeros(HIGHLIGHT_CONTEXT_SECONDS, dtype=torch.long)
    phase = torch.zeros(HIGHLIGHT_CONTEXT_SECONDS, dtype=torch.long)
    valid_seconds = max(0, min(HIGHLIGHT_CONTEXT_SECONDS, int(np.ceil(duration))))
    if valid_seconds < HIGHLIGHT_CONTEXT_SECONDS:
        include[valid_seconds:] = IGNORE_INDEX
        phase[valid_seconds:] = IGNORE_INDEX

    for second in range(valid_seconds):
        center = second + 0.5
        if clip_start <= center < clip_end:
            include[second] = 1
            phase[second] = PHASE_BUILDUP
        if fight_start <= center < fight_end:
            phase[second] = PHASE_FIGHT
        elif fight_end <= center < clip_end:
            phase[second] = PHASE_PAYOFF

    return include, phase


def _has_highlight_bounds(label: dict) -> bool:
    duration = _duration_for_label(label)
    bounds = _clip_bounds(label, duration)
    return bounds is not None and bounds[1] > bounds[0]


def _window_outside_posted_clip(start_second: int, clip_bounds: tuple[float, float] | None) -> bool:
    if clip_bounds is None:
        return False
    window_start = float(start_second)
    window_end = window_start + WINDOW_SECONDS
    clip_start, clip_end = clip_bounds
    return window_end <= clip_start or window_start >= clip_end


def _resolve_clip_path(clips_dir: Path | None, label: dict) -> Path | None:
    raw_path = str(label.get("raw_path") or "").strip()
    if raw_path:
        candidate = Path(raw_path).expanduser()
        if candidate.exists() and candidate.suffix.lower() == ".mp4":
            return candidate

    filename = str(label.get("filename", "")).strip()
    if clips_dir is not None and filename:
        candidate = clips_dir / filename
        if candidate.exists() and candidate.suffix.lower() == ".mp4":
            return candidate

    return None


def _label_group_key(label: dict) -> str:
    raw_path = str(label.get("raw_path") or "").strip()
    if raw_path:
        path = Path(raw_path)
        return f"{path.parent}|{path.stem}"
    filename = str(label.get("filename") or "").strip()
    return Path(filename).stem or filename


def _progress_bar(current: int, total: int, width: int = 24) -> str:
    if total <= 0:
        return "[" + "-" * width + "]"
    pct = max(0.0, min(float(current) / float(total), 1.0))
    filled = int(round(pct * width))
    return "[" + "#" * filled + "." * (width - filled) + "]"


def _progress_line(phase: str, current: int, total: int, *, epoch: int | None = None, epochs: int | None = None, detail: str = "") -> str:
    pct = 100.0 * (float(current) / float(total)) if total else 0.0
    epoch_text = f" epoch {epoch}/{epochs}" if epoch is not None and epochs is not None else ""
    suffix = f" {detail}" if detail else ""
    return f"{phase}{epoch_text} {_progress_bar(current, total)} {current}/{total} {pct:5.1f}%{suffix}"


class LoLFightDataset(Dataset[tuple[torch.Tensor, int]]):
    def __init__(
        self,
        clips_dir: Path | None,
        labels: list[dict],
        smoke_test: bool = False,
    ) -> None:
        self.clips_dir = clips_dir
        self.labels = labels[:5] if smoke_test else labels
        self.sample_index: list[tuple[Path, int, int]] = []
        self.missing_labels: list[str] = []
        self._prepare_samples()

    def _prepare_samples(self) -> None:
        fight_samples: list[tuple[Path, int, int]] = []
        non_fight_samples: list[tuple[Path, int, int]] = []

        for label in self.labels:
            clip_path = _resolve_clip_path(self.clips_dir, label)
            if clip_path is None:
                self.missing_labels.append(str(label.get("filename") or label.get("raw_path") or "<unknown>"))
                continue

            fight_segments = _clean_segments(label)
            if not fight_segments:
                self.missing_labels.append(str(label.get("filename") or clip_path.name))
                continue

            last_fight_end = max(end for _, end in fight_segments)
            duration_value = (
                label.get("duration")
                or label.get("source_duration")
                or label.get("clip_duration")
                or last_fight_end + WINDOW_SECONDS
            )
            total_seconds = max(WINDOW_SECONDS, int(np.ceil(float(duration_value))))
            clip_bounds = _clip_bounds(label, float(duration_value))

            for start_second in range(0, max(0, total_seconds - WINDOW_SECONDS + 1)):
                overlap_pct = _window_overlap_pct(start_second, fight_segments)
                if overlap_pct >= POSITIVE_OVERLAP_THRESHOLD:
                    fight_samples.append((clip_path, start_second, 1))
                elif clip_bounds is None and overlap_pct < NEGATIVE_OVERLAP_THRESHOLD:
                    non_fight_samples.append((clip_path, start_second, 0))
                elif _window_outside_posted_clip(start_second, clip_bounds) and overlap_pct < NEGATIVE_OVERLAP_THRESHOLD:
                    non_fight_samples.append((clip_path, start_second, 0))

        rng = random.Random(42)
        rng.shuffle(fight_samples)
        rng.shuffle(non_fight_samples)
        minority_count = min(len(fight_samples), len(non_fight_samples))
        if minority_count == 0:
            self.sample_index = []
            return
        self.sample_index = fight_samples[:minority_count] + non_fight_samples[:minority_count]
        rng.shuffle(self.sample_index)

    def __len__(self) -> int:
        return len(self.sample_index)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        clip_path, start_second, label = self.sample_index[index]
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        tensors: list[torch.Tensor] = []

        precomputed_path = PRECOMPUTED_DIR / (clip_path.stem + ".npy")
        if precomputed_path.exists():
            frames = np.load(str(precomputed_path), mmap_mode="r")
            for second_offset in range(WINDOW_SECONDS):
                frame_index = min(start_second + second_offset, len(frames) - 1)
                frame = frames[frame_index].copy()
                normalized = (frame.astype(np.float32) / 255.0 - mean) / std
                chw = np.transpose(normalized, (2, 0, 1))
                tensors.append(torch.from_numpy(chw).float())
        else:
            capture = cv2.VideoCapture(str(clip_path))
            source_fps = float(capture.get(cv2.CAP_PROP_FPS) or 120.0)
            if source_fps <= 0.0:
                source_fps = 120.0
            for second_offset in range(WINDOW_SECONDS):
                target_second = start_second + second_offset
                frame_number = int(target_second * source_fps)
                capture.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
                ok, frame_bgr = capture.read()
                if ok:
                    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                else:
                    frame_rgb = np.zeros((224, 224, 3), dtype=np.uint8)
                resized = cv2.resize(
                    frame_rgb,
                    (224, 224),
                    interpolation=cv2.INTER_AREA,
                )
                normalized = (resized.astype(np.float32) / 255.0 - mean) / std
                chw = np.transpose(normalized, (2, 0, 1))
                tensors.append(torch.from_numpy(chw).float())
            capture.release()

        return torch.stack(tensors), int(label)


class LoLHighlightDataset(Dataset[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]):
    def __init__(
        self,
        clips_dir: Path | None,
        labels: list[dict],
        smoke_test: bool = False,
    ) -> None:
        self.clips_dir = clips_dir
        self.labels = labels[:5] if smoke_test else labels
        self.sample_index: list[tuple[Path, dict]] = []
        self.missing_labels: list[str] = []
        self._prepare_samples()

    def _prepare_samples(self) -> None:
        for label in self.labels:
            clip_path = _resolve_clip_path(self.clips_dir, label)
            if clip_path is None:
                self.missing_labels.append(str(label.get("filename") or label.get("raw_path") or "<unknown>"))
                continue
            if not _has_highlight_bounds(label):
                self.missing_labels.append(str(label.get("filename") or clip_path.name))
                continue
            self.sample_index.append((clip_path, label))

    def __len__(self) -> int:
        return len(self.sample_index)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        clip_path, label = self.sample_index[index]
        pixel_values = _sample_context_tensor(clip_path, HIGHLIGHT_CONTEXT_SECONDS, HIGHLIGHT_INPUT_FRAMES)
        include_targets, phase_targets = _highlight_targets_for_label(label)
        return pixel_values, include_targets, phase_targets


def _sample_context_tensor(
    clip_path: Path,
    context_seconds: int,
    input_frames: int,
) -> torch.Tensor:
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    sample_seconds = np.linspace(0.0, max(float(context_seconds) - 1.0, 0.0), max(1, input_frames))
    tensors: list[torch.Tensor] = []

    precomputed_path = PRECOMPUTED_DIR / (clip_path.stem + ".npy")
    if precomputed_path.exists():
        frames = np.load(str(precomputed_path), mmap_mode="r")
        for second in sample_seconds:
            frame_index = min(int(round(float(second))), len(frames) - 1)
            frame = frames[max(0, frame_index)].copy()
            if frame.shape[:2] != (224, 224):
                frame = cv2.resize(frame, (224, 224), interpolation=cv2.INTER_AREA)
            normalized = (frame.astype(np.float32) / 255.0 - mean) / std
            tensors.append(torch.from_numpy(np.transpose(normalized, (2, 0, 1))).float())
        return torch.stack(tensors)

    capture = cv2.VideoCapture(str(clip_path))
    source_fps = float(capture.get(cv2.CAP_PROP_FPS) or 120.0)
    if source_fps <= 0.0:
        source_fps = 120.0
    for second in sample_seconds:
        frame_number = int(float(second) * source_fps)
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ok, frame_bgr = capture.read()
        if ok:
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        else:
            frame_rgb = np.zeros((224, 224, 3), dtype=np.uint8)
        resized = cv2.resize(frame_rgb, (224, 224), interpolation=cv2.INTER_AREA)
        normalized = (resized.astype(np.float32) / 255.0 - mean) / std
        tensors.append(torch.from_numpy(np.transpose(normalized, (2, 0, 1))).float())
    capture.release()
    return torch.stack(tensors)


class VideoMAEClassifier(nn.Module):
    def __init__(self, freeze_backbone: bool = True, unfreeze_last_n_layers: int = 2) -> None:
        super().__init__()
        from transformers import VideoMAEModel

        self.videomae = VideoMAEModel.from_pretrained("MCG-NJU/videomae-base")
        self.classifier = nn.Linear(768, 2)
        if freeze_backbone:
            self._freeze_backbone(max(0, unfreeze_last_n_layers))

    def _freeze_backbone(self, unfreeze_last_n_layers: int) -> None:
        for parameter in self.videomae.parameters():
            parameter.requires_grad = False

        encoder = getattr(self.videomae, "encoder", None)
        layers = list(getattr(encoder, "layer", []) or [])
        if unfreeze_last_n_layers > 0 and layers:
            for layer in layers[-unfreeze_last_n_layers:]:
                for parameter in layer.parameters():
                    parameter.requires_grad = True

        for attr_name in ("layernorm", "fc_norm", "norm"):
            module = getattr(self.videomae, attr_name, None)
            if module is not None:
                for parameter in module.parameters():
                    parameter.requires_grad = True

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        outputs = self.videomae(pixel_values=pixel_values)
        pooled = outputs.last_hidden_state.mean(dim=1)
        return self.classifier(pooled)


def _temporal_video_features(
    hidden_states: torch.Tensor,
    pixel_values: torch.Tensor,
    videomae_config: object,
    context_seconds: int,
) -> torch.Tensor:
    batch_size, sequence_length, hidden_size = hidden_states.shape
    input_frames = int(pixel_values.shape[1])
    tubelet_size = int(getattr(videomae_config, "tubelet_size", 2) or 2)
    temporal_tokens = max(1, input_frames // max(1, tubelet_size))
    if sequence_length % temporal_tokens != 0:
        temporal_tokens = min(input_frames, sequence_length)
        while temporal_tokens > 1 and sequence_length % temporal_tokens != 0:
            temporal_tokens -= 1
    spatial_tokens = max(1, sequence_length // max(1, temporal_tokens))
    usable_tokens = temporal_tokens * spatial_tokens
    temporal = hidden_states[:, :usable_tokens, :].reshape(
        batch_size,
        temporal_tokens,
        spatial_tokens,
        hidden_size,
    ).mean(dim=2)
    if temporal_tokens != context_seconds:
        temporal = F.interpolate(
            temporal.transpose(1, 2),
            size=context_seconds,
            mode="linear",
            align_corners=False,
        ).transpose(1, 2)
    return temporal


class VideoMAEHighlightEditor(nn.Module):
    def __init__(
        self,
        context_seconds: int = HIGHLIGHT_CONTEXT_SECONDS,
        freeze_backbone: bool = True,
        unfreeze_last_n_layers: int = 2,
    ) -> None:
        super().__init__()
        self.context_seconds = int(context_seconds)
        from transformers import VideoMAEModel

        self.videomae = VideoMAEModel.from_pretrained("MCG-NJU/videomae-base")
        self.include_head = nn.Linear(768, 2)
        self.phase_head = nn.Linear(768, len(PHASE_NAMES))
        if freeze_backbone:
            self._freeze_backbone(max(0, unfreeze_last_n_layers))

    def _freeze_backbone(self, unfreeze_last_n_layers: int) -> None:
        for parameter in self.videomae.parameters():
            parameter.requires_grad = False

        encoder = getattr(self.videomae, "encoder", None)
        layers = list(getattr(encoder, "layer", []) or [])
        if unfreeze_last_n_layers > 0 and layers:
            for layer in layers[-unfreeze_last_n_layers:]:
                for parameter in layer.parameters():
                    parameter.requires_grad = True

        for attr_name in ("layernorm", "fc_norm", "norm"):
            module = getattr(self.videomae, attr_name, None)
            if module is not None:
                for parameter in module.parameters():
                    parameter.requires_grad = True

    def forward(self, pixel_values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        outputs = self.videomae(pixel_values=pixel_values)
        temporal = _temporal_video_features(
            outputs.last_hidden_state,
            pixel_values,
            self.videomae.config,
            self.context_seconds,
        )
        include_logits = self.include_head(temporal)
        phase_logits = self.phase_head(temporal)
        return include_logits, phase_logits


def _auto_batch_size() -> int:
    if torch.cuda.is_available():
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        if vram_gb >= 10.0:
            return 6
        if vram_gb >= 9.0:
            return 4
        return 2
    return 1


def _split_labels(all_labels: list[dict], val_fraction: float = 0.15) -> tuple[list[dict], list[dict]]:
    groups: dict[str, list[dict]] = {}
    for label in all_labels:
        groups.setdefault(_label_group_key(label), []).append(label)

    group_items = sorted(groups.items(), key=lambda item: item[0])
    rng = random.Random(42)
    rng.shuffle(group_items)
    val_group_count = max(1, int(round(len(group_items) * max(0.0, min(val_fraction, 0.5))))) if group_items else 0
    val_keys = {key for key, _ in group_items[:val_group_count]}

    train_labels = [label for key, labels in group_items if key not in val_keys for label in labels]
    val_labels = [label for key, labels in group_items if key in val_keys for label in labels]
    if not train_labels and val_labels:
        train_labels = val_labels
    if not val_labels and train_labels:
        val_labels = train_labels[:1]
    return train_labels, val_labels


def _create_loader(
    dataset: Dataset,
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


def _trainable_counts(model: nn.Module) -> tuple[int, int]:
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    return trainable, total


def _optimizer(model: nn.Module, classifier_lr: float, backbone_lr: float) -> torch.optim.Optimizer:
    head_prefixes = ("classifier.", "include_head.", "phase_head.")
    classifier_params = [
        parameter
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and name.startswith(head_prefixes)
    ]
    backbone_params = [
        parameter
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and not name.startswith(head_prefixes)
    ]
    parameter_groups: list[dict] = []
    if backbone_params:
        parameter_groups.append({"params": backbone_params, "lr": backbone_lr})
    if classifier_params:
        parameter_groups.append({"params": classifier_params, "lr": classifier_lr})
    if not parameter_groups:
        raise RuntimeError("No trainable parameters are enabled")
    return torch.optim.AdamW(parameter_groups, weight_decay=0.01)


def _write_metrics(metrics_path: Path, payload: dict) -> None:
    metrics_path.write_text(json.dumps(payload), encoding="utf-8")


def _highlight_loss(
    include_logits: torch.Tensor,
    phase_logits: torch.Tensor,
    include_targets: torch.Tensor,
    phase_targets: torch.Tensor,
) -> torch.Tensor:
    include_loss = F.cross_entropy(
        include_logits.reshape(-1, 2),
        include_targets.reshape(-1),
        ignore_index=IGNORE_INDEX,
    )
    phase_loss = F.cross_entropy(
        phase_logits.reshape(-1, len(PHASE_NAMES)),
        phase_targets.reshape(-1),
        ignore_index=IGNORE_INDEX,
    )
    return include_loss + phase_loss


def _masked_accuracy(logits: torch.Tensor, targets: torch.Tensor) -> tuple[int, int]:
    mask = targets != IGNORE_INDEX
    total = int(mask.sum().item())
    if total == 0:
        return 0, 0
    preds = logits.argmax(dim=-1)
    correct = int(((preds == targets) & mask).sum().item())
    return correct, total


def _span_from_include_labels(labels: torch.Tensor) -> tuple[int, int] | None:
    indexes = torch.nonzero(labels == 1, as_tuple=False).flatten()
    if indexes.numel() == 0:
        return None
    return int(indexes[0].item()), int(indexes[-1].item()) + 1


def _span_from_include_logits(logits: torch.Tensor) -> tuple[int, int] | None:
    include_probs = torch.softmax(logits, dim=-1)[:, 1]
    mask = include_probs >= 0.5
    best: tuple[int, int, float] | None = None
    start: int | None = None
    score = 0.0
    for idx, active in enumerate(mask.tolist()):
        if active:
            if start is None:
                start = idx
                score = 0.0
            score += float(include_probs[idx].item())
        elif start is not None:
            candidate = (start, idx, score)
            if best is None or candidate[2] > best[2]:
                best = candidate
            start = None
    if start is not None:
        candidate = (start, len(mask), score)
        if best is None or candidate[2] > best[2]:
            best = candidate
    return None if best is None else (best[0], best[1])


def _boundary_mae_seconds(include_logits: torch.Tensor, include_targets: torch.Tensor) -> float | None:
    errors: list[float] = []
    for logits, targets in zip(include_logits.detach().cpu(), include_targets.detach().cpu()):
        predicted = _span_from_include_logits(logits)
        expected = _span_from_include_labels(targets)
        if predicted is None or expected is None:
            continue
        errors.append(abs(predicted[0] - expected[0]))
        errors.append(abs(predicted[1] - expected[1]))
    if not errors:
        return None
    return float(np.mean(errors))


def main() -> int:
    args = _build_parser().parse_args()

    os.environ.setdefault(
        "TRANSFORMERS_CACHE",
        str(Path.home() / ".cache" / "huggingface"),
    )

    if args.clips_dir is not None and not args.clips_dir.exists():
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
        f"run_id={args.run_id} slice={args.slice} task={args.task} clips_dir={args.clips_dir} labels={args.labels} "
        f"epochs={epochs} batch_size={batch_size} output_dir={output_dir} device={device} "
        f"freeze_backbone={args.freeze_backbone} unfreeze_last_n_layers={args.unfreeze_last_n_layers} "
        f"classifier_lr={args.classifier_lr} backbone_lr={args.backbone_lr} val_fraction={args.val_fraction}",
        flush=True,
    )
    print("Building dataset index...", flush=True)

    train_labels, val_labels = _split_labels(all_labels, args.val_fraction)
    if args.task == "highlight":
        train_dataset = LoLHighlightDataset(args.clips_dir, train_labels, args.smoke_test)
        val_dataset = LoLHighlightDataset(args.clips_dir, val_labels, args.smoke_test)
        sample_label = "clips"
    else:
        train_dataset = LoLFightDataset(args.clips_dir, train_labels, args.smoke_test)
        val_dataset = LoLFightDataset(args.clips_dir, val_labels, args.smoke_test)
        sample_label = "windows"
    print(f"Train labels: {len(train_labels)} {sample_label}: {len(train_dataset)} missing: {len(train_dataset.missing_labels)}", flush=True)
    print(f"Val labels: {len(val_labels)} {sample_label}: {len(val_dataset)} missing: {len(val_dataset.missing_labels)}", flush=True)
    if train_dataset.missing_labels:
        print(f"WARNING: skipped {len(train_dataset.missing_labels)} train labels with missing clips", flush=True)
    if val_dataset.missing_labels:
        print(f"WARNING: skipped {len(val_dataset.missing_labels)} val labels with missing clips", flush=True)
    print("Starting training loop...", flush=True)
    if len(train_dataset) == 0 or len(val_dataset) == 0:
        print("No usable training or validation windows found", file=sys.stderr)
        return 1

    num_workers = 0 if platform.system() == "Windows" else min(4, os.cpu_count() or 1)
    train_loader = _create_loader(train_dataset, batch_size, num_workers, True)
    val_loader = _create_loader(val_dataset, batch_size, num_workers, False)

    if args.task == "highlight":
        model = VideoMAEHighlightEditor(
            freeze_backbone=args.freeze_backbone,
            unfreeze_last_n_layers=args.unfreeze_last_n_layers,
        ).to(device)
        checkpoint_path = output_dir / "videomae_lol_highlight_editor_temporal.pt"
    else:
        model = VideoMAEClassifier(
            freeze_backbone=args.freeze_backbone,
            unfreeze_last_n_layers=args.unfreeze_last_n_layers,
        ).to(device)
        checkpoint_path = output_dir / "videomae_lol_best.pt"
    trainable, total = _trainable_counts(model)
    print(f"Trainable parameters: {trainable:,}/{total:,} ({100.0 * trainable / max(total, 1):.2f}%)", flush=True)

    if checkpoint_path.exists() and not args.smoke_test:
        checkpoint = torch.load(str(checkpoint_path), map_location=device)
        state_dict = checkpoint.get("model_state", checkpoint) if isinstance(checkpoint, dict) else checkpoint
        model.load_state_dict(state_dict)
        print("Resuming from existing checkpoint", flush=True)

    optimizer = _optimizer(model, args.classifier_lr, args.backbone_lr)
    from transformers import get_cosine_schedule_with_warmup

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=min(100, max(epochs * len(train_loader) // 10, 1)),
        num_training_steps=max(epochs * len(train_loader), 1),
    )
    gradient_accumulation_steps = 4
    scaler: Optional[torch.cuda.amp.GradScaler] = (
        torch.cuda.amp.GradScaler() if device.type == CUDA_DEVICE_TYPE else None
    )

    best_val_loss = float("inf")
    patience_left = 4
    progress_interval = max(1, args.progress_interval)

    for epoch in range(1, epochs + 1):
        model.train()
        total_train_loss = 0.0
        optimizer.zero_grad()
        epoch_started = time.monotonic()

        for batch_idx, batch in enumerate(train_loader, start=1):
            if args.task == "highlight":
                pixel_values, include_targets, phase_targets = batch
                pixel_values = pixel_values.to(device)
                include_targets = include_targets.to(device)
                phase_targets = phase_targets.to(device)
            else:
                pixel_values, labels = batch
                pixel_values = pixel_values.to(device)
                labels = labels.to(device)

            try:
                ctx = torch.cuda.amp.autocast() if scaler else contextlib.nullcontext()
                with ctx:
                    if args.task == "highlight":
                        include_logits, phase_logits = model(pixel_values)
                        loss = _highlight_loss(
                            include_logits,
                            phase_logits,
                            include_targets,
                            phase_targets,
                        ) / gradient_accumulation_steps
                    else:
                        logits = model(pixel_values)
                        loss = F.cross_entropy(logits, labels) / gradient_accumulation_steps

                if scaler:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()

                if batch_idx % gradient_accumulation_steps == 0 or batch_idx == len(train_loader):
                    if scaler:
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()

                batch_loss = loss.item() * gradient_accumulation_steps
                total_train_loss += batch_loss

                if batch_idx == 1 or batch_idx == len(train_loader) or batch_idx % progress_interval == 0:
                    elapsed = max(time.monotonic() - epoch_started, 1e-6)
                    batches_per_sec = batch_idx / elapsed
                    remaining = (len(train_loader) - batch_idx) / max(batches_per_sec, 1e-6)
                    detail = f"loss={batch_loss:.4f} eta={remaining / 60.0:.1f}m"
                    print(_progress_line("train", batch_idx, len(train_loader), epoch=epoch, epochs=epochs, detail=detail), flush=True)
                    _write_metrics(
                        metrics_path,
                        {
                            "run_id": args.run_id,
                            "slice": int(args.slice),
                            "task": args.task,
                            "status": "running",
                            "phase": "train",
                            "epoch": epoch,
                            "epochs": epochs,
                            "batch": batch_idx,
                            "batches": len(train_loader),
                            "progress": round(((epoch - 1) + batch_idx / max(len(train_loader), 1)) / max(epochs, 1), 4),
                            "train_loss": round(batch_loss, 4),
                        },
                    )

            except RuntimeError as exc:
                if "out of memory" not in str(exc).lower():
                    raise
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                if batch_size > 1:
                    batch_size = max(1, batch_size // 2)
                    train_loader = _create_loader(train_dataset, batch_size, num_workers, True)
                    val_loader = _create_loader(val_dataset, batch_size, num_workers, False)
                    print(f"WARNING: OOM encountered; reducing batch_size to {batch_size}", flush=True)
                    continue

                metrics = {
                    "run_id": args.run_id,
                    "slice": int(args.slice),
                    "task": args.task,
                    "status": "failed",
                    "phase": "train",
                    "epoch": epoch,
                    "train_loss": round(total_train_loss, 4),
                    "val_loss": float("inf"),
                    "accuracy": 0.0,
                }
                _write_metrics(metrics_path, metrics)
                print(json.dumps(metrics), flush=True)
                return 1

        avg_train_loss = total_train_loss / max(len(train_loader), 1)

        model.eval()
        total_val_loss = 0.0
        correct = 0
        total = 0
        include_correct = 0
        include_total = 0
        phase_correct = 0
        phase_total = 0
        boundary_errors: list[float] = []

        with torch.no_grad():
            for val_idx, batch in enumerate(val_loader, start=1):
                if args.task == "highlight":
                    pixel_values, include_targets, phase_targets = batch
                    pixel_values = pixel_values.to(device)
                    include_targets = include_targets.to(device)
                    phase_targets = phase_targets.to(device)
                else:
                    pixel_values, labels = batch
                    pixel_values = pixel_values.to(device)
                    labels = labels.to(device)
                ctx = torch.cuda.amp.autocast() if device.type == CUDA_DEVICE_TYPE else contextlib.nullcontext()
                with ctx:
                    if args.task == "highlight":
                        include_logits, phase_logits = model(pixel_values)
                        loss = _highlight_loss(include_logits, phase_logits, include_targets, phase_targets)
                    else:
                        logits = model(pixel_values)
                        loss = F.cross_entropy(logits, labels)
                total_val_loss += loss.item()
                if args.task == "highlight":
                    batch_include_correct, batch_include_total = _masked_accuracy(include_logits, include_targets)
                    batch_phase_correct, batch_phase_total = _masked_accuracy(phase_logits, phase_targets)
                    include_correct += batch_include_correct
                    include_total += batch_include_total
                    phase_correct += batch_phase_correct
                    phase_total += batch_phase_total
                    boundary_mae = _boundary_mae_seconds(include_logits, include_targets)
                    if boundary_mae is not None:
                        boundary_errors.append(boundary_mae)
                else:
                    preds = logits.argmax(dim=1)
                    correct += (preds == labels).sum().item()
                    total += labels.size(0)
                if val_idx == 1 or val_idx == len(val_loader) or val_idx % progress_interval == 0:
                    print(_progress_line("valid", val_idx, len(val_loader), epoch=epoch, epochs=epochs), flush=True)

        avg_val_loss = total_val_loss / max(len(val_loader), 1)
        accuracy = include_correct / max(include_total, 1) if args.task == "highlight" else correct / max(total, 1)
        metrics = {
            "run_id": args.run_id,
            "slice": int(args.slice),
            "task": args.task,
            "status": "running",
            "phase": "epoch_complete",
            "epoch": epoch,
            "epochs": epochs,
            "progress": round(epoch / max(epochs, 1), 4),
            "train_loss": round(avg_train_loss, 4),
            "val_loss": round(avg_val_loss, 4),
            "accuracy": round(accuracy, 4),
        }
        if args.task == "highlight":
            metrics.update(
                {
                    "include_accuracy": round(include_correct / max(include_total, 1), 4),
                    "phase_accuracy": round(phase_correct / max(phase_total, 1), 4),
                    "boundary_mae_sec": round(float(np.mean(boundary_errors)), 3) if boundary_errors else None,
                }
            )
        _write_metrics(metrics_path, metrics)
        print(json.dumps(metrics), flush=True)

        if avg_val_loss < best_val_loss and not args.smoke_test:
            best_val_loss = avg_val_loss
            patience_left = 4
            if args.task == "highlight":
                torch.save(
                    {
                        "task": "highlight",
                        "architecture": "temporal_token_head",
                        "context_seconds": HIGHLIGHT_CONTEXT_SECONDS,
                        "input_frames": HIGHLIGHT_INPUT_FRAMES,
                        "phase_names": PHASE_NAMES,
                        "model_state": model.state_dict(),
                    },
                    str(checkpoint_path),
                )
            else:
                torch.save(model.state_dict(), str(checkpoint_path))
            print(f"Checkpoint saved at epoch {epoch}", flush=True)
        else:
            patience_left -= 1
            if patience_left <= 0:
                print("Early stopping triggered", flush=True)
                break

    final_metrics = {
        "run_id": args.run_id,
        "slice": int(args.slice),
        "task": args.task,
        "status": "complete",
        "phase": "complete",
        "epoch": min(epoch, epochs),
        "epochs": epochs,
        "progress": 1.0,
        "best_val_loss": round(best_val_loss, 4) if best_val_loss < float("inf") else None,
    }
    _write_metrics(metrics_path, final_metrics)
    print(json.dumps(final_metrics), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
