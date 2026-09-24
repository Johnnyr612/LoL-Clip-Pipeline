"""Inference architecture/preprocessing matching vjepa21_vitb_window_tcn_highlight_v1.

No training or automatic downloads. Keep this version paired with schema 1 weights.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import importlib
import math
import sys
import cv2
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from tqdm import tqdm

@dataclass(frozen=True)
class TimelineConfig:
    # 16 densely sampled frames per local window, NOT 16 frames per minute.
    input_frames: int = 16
    image_size: int = 384
    window_seconds: float = 2.0
    step_seconds: float = 1.0
    head_width: int = 128
    spatial_resize: str = "full_frame_bilinear"


def timeline_edges(duration: float, step: float) -> torch.Tensor:
    n = math.ceil(duration / step)
    return torch.tensor([min(i * step, duration) for i in range(n + 1)], dtype=torch.float32)


def window_frame_indexes(duration, fps, count, config):
    if not math.isfinite(duration) or not 0 < duration <= 120 or fps <= 0 or count <= 0:
        raise ValueError("Invalid short-video sampling metadata")
    edges = timeline_edges(duration, config.step_seconds).numpy()
    centers = (edges[1:] + edges[:-1]) / 2
    offsets = (np.arange(config.input_frames) + 0.5) / config.input_frames * config.window_seconds - config.window_seconds / 2
    times = np.clip(centers[:, None] + offsets, 0, min(duration, count / fps) - 1 / fps)
    indexes = np.clip(np.rint(times * fps).astype(int), 0, count - 1)
    return indexes


@dataclass
class DecodedWindows:
    frames: dict
    indexes: np.ndarray
    config: TimelineConfig
    duration: float

    def windows(self):
        for ids in self.indexes:
            yield torch.from_numpy(np.stack([self.frames[int(i)] for i in ids])).permute(3, 0, 1, 2)


def sample_windows(path: Path, duration: float, config: TimelineConfig):
    """Decode required frames sequentially once; yield uint8 C,T,H,W windows.

    Full-frame resizing preserves HUD and matches the old app's geometry policy.
    This deliberate alternative to native center-cropping is stored in checkpoints.
    """
    cap = cv2.VideoCapture(str(path))
    try:
        fps, count = cap.get(cv2.CAP_PROP_FPS), int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if not cap.isOpened() or not math.isfinite(fps) or fps <= 0 or count <= 0:
            raise ValueError(f"Cannot decode {path}")
        indexes = window_frame_indexes(duration, fps, count, config)
        needed = set(indexes.flatten().tolist())
        frames = {}
        for index in tqdm(range(int(indexes.max()) + 1), desc="Decode raw frames", unit="frame", position=2, leave=False, dynamic_ncols=True):
            if not cap.grab():
                raise ValueError(f"Decode failed at frame {index}: {path}")
            if index in needed:
                ok, frame = cap.retrieve()
                if not ok:
                    raise ValueError(f"Decode failed at frame {index}: {path}")
                frame = cv2.resize(frame, (config.image_size, config.image_size), interpolation=cv2.INTER_LINEAR)
                frames[index] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        for ids in indexes:
            yield torch.from_numpy(np.stack([frames[int(i)] for i in ids])).permute(3, 0, 1, 2)
    finally:
        cap.release()


def construct_encoder(repo: Path) -> nn.Module:
    """Construct the native architecture without loading any weights."""
    repo = repo.expanduser().resolve()
    model_file = repo / "app" / "vjepa_2_1" / "models" / "vision_transformer.py"
    if not model_file.is_file() or not (repo / "LICENSE").is_file():
        raise ValueError(f"Missing official V-JEPA 2.1 source checkout: {repo}. See docs/vjepa21-trainer.md")
    # Deliberately avoid torch.hub downloads, predictor allocation, and remote code execution.
    sys.path.insert(0, str(repo))
    module = importlib.import_module("app.vjepa_2_1.models.vision_transformer")
    if Path(module.__file__).resolve() != model_file:
        raise ValueError("Another 'app' package shadows the requested V-JEPA repository; use a clean Python process")
    print("Constructing V-JEPA encoder...", flush=True)
    encoder = module.vit_base(img_size=(384, 384), patch_size=16, num_frames=64, tubelet_size=2,
                              use_sdpa=True, use_SiLU=False, wide_SiLU=True, uniform_power=False,
                              use_rope=True, img_temporal_dim_size=1, interpolate_rope=True)
    # Matches the OFFICIAL factory, including native 64-frame reference geometry.
    # Runtime local windows may have 16 frames; the encoder computes T dynamically.
    return encoder


class TemporalHead(nn.Module):
    def __init__(self, feature_dim: int, width: int):
        super().__init__()
        self.norm = nn.LayerNorm(feature_dim)
        self.project = nn.Linear(feature_dim, width)
        self.blocks = nn.ModuleList([nn.Conv1d(width, width, 3, padding=d, dilation=d) for d in (1, 2, 4, 8)])
        self.dropout = nn.Dropout(0.15)
        self.output = nn.Linear(width, 4)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        x = self.project(self.norm(features)).transpose(1, 2)
        for block in self.blocks:
            x = x + self.dropout(F.gelu(block(x)))
        return self.output(x.transpose(1, 2))


class VJEPAHighlightModel(nn.Module):
    def __init__(self, encoder: nn.Module, config: TimelineConfig):
        super().__init__()
        self.encoder, self.config = encoder, config
        self.head = TemporalHead(encoder.embed_dim, config.head_width)

    def train(self, mode: bool = True):
        super().train(mode)
        # Native encoder is deterministic for both frozen and partial fine-tuning.
        # eval() does not disable gradients. Head dropout still follows mode.
        self.encoder.eval()
        return self

    def encode_window(self, uint8_window: torch.Tensor) -> torch.Tensor:
        return self.encode_windows(uint8_window.unsqueeze(0))[0]

    def encode_windows(self, uint8_windows: torch.Tensor) -> torch.Tensor:
        """Encode B,C,T,H,W windows together, preserving timeline order."""
        device = next(self.encoder.parameters()).device
        x = uint8_windows.to(device=device, dtype=torch.float32) / 255
        mean = x.new_tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1, 1)
        std = x.new_tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1, 1)
        tokens = self.encoder((x - mean) / std)
        expected = self.config.input_frames // 2 * (self.config.image_size // 16) ** 2
        if not isinstance(tokens, torch.Tensor) or tokens.ndim != 3 or tokens.shape[1:] != (expected, self.encoder.embed_dim):
            raise ValueError("Unexpected V-JEPA token shape; refusing to guess temporal/spatial layout")
        return tokens.mean(dim=1)


def decode_highlight(logits: torch.Tensor, edges: torch.Tensor, threshold: float = 0.5,
                     min_seconds: float = 4.0, max_seconds: float = 60.0,
                     boundary_weight: float = 1.0) -> tuple[float, float] | None:
    """Find one coherent span. Include evidence must be positive, or reject."""
    logits = logits.detach().float().cpu()
    odds = logits[:, 0] - math.log(threshold / (1 - threshold))
    cumulative = torch.cat([torch.zeros(1), (odds * edges.diff()).cumsum(0)])
    start_scores, end_scores = logits[:, 2].log_softmax(0), logits[:, 3].log_softmax(0)
    best_score, best = -math.inf, None
    for a in range(len(logits)):
        for b in range(a + 1, len(logits) + 1):
            duration = float(edges[b] - edges[a])
            if not min_seconds <= duration <= max_seconds:
                continue
            evidence = float(cumulative[b] - cumulative[a])
            if evidence <= 0:
                continue
            score = evidence + boundary_weight * float(start_scores[a] + end_scores[b - 1])
            if score > best_score:
                best_score, best = score, (float(edges[a]), float(edges[b]))
    return best
