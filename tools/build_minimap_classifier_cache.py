from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend import config
from backend.minimap_classifier import MinimapIconClassifier, _feature_vector, _icons_signature


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the minimap classifier cache, optionally adding GAN augmentations.")
    parser.add_argument("--icons", type=Path, default=config.MINIMAP_ICONS_DIR / "images")
    parser.add_argument("--out", type=Path, default=config.MINIMAP_CLASSIFIER_CACHE_PATH)
    parser.add_argument("--gan-checkpoint", type=Path, default=Path("checkpoints/minimap_mask_gan.pt"))
    parser.add_argument("--gan-samples-per-icon", type=int, default=12)
    parser.add_argument("--base-augmentations-per-icon", type=int, default=36)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    return parser.parse_args()


def load_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.array(image.convert("RGB").resize((120, 120)))


def gan_features(args: argparse.Namespace, names: list[str]) -> tuple[list[int], list[np.ndarray]]:
    if not args.gan_checkpoint.exists() or args.gan_samples_per_icon <= 0:
        return [], []
    try:
        import torch
        from train_minimap_mask_gan import build_models, require_torch, to_tensor
    except Exception as exc:  # noqa: BLE001
        print(f"warning: GAN augmentation skipped; PyTorch unavailable: {exc}")
        return [], []

    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else args.device)
    if args.device == "auto" and not torch.cuda.is_available():
        device = torch.device("cpu")

    _, nn, *_ = require_torch()
    generator_class, _ = build_models(torch, nn)
    checkpoint = torch.load(args.gan_checkpoint, map_location=device)
    generator = generator_class(checkpoint.get("latent_dim", 64)).to(device)
    generator.load_state_dict(checkpoint["generator"])
    generator.eval()

    labels: list[int] = []
    features: list[np.ndarray] = []
    icon_paths = sorted(args.icons.glob("*.png"))
    with torch.no_grad():
        for class_idx, path in enumerate(icon_paths):
            portrait = to_tensor(load_rgb(path), torch).unsqueeze(0).to(device)
            for _ in range(args.gan_samples_per_icon):
                noise = torch.randn(1, checkpoint.get("latent_dim", 64), device=device)
                fake = generator(portrait, noise)[0].detach().cpu()
                rgb = ((fake.permute(1, 2, 0).numpy() + 1.0) * 127.5).clip(0, 255).astype(np.uint8)
                rgb = cv2.resize(rgb, (120, 120), interpolation=cv2.INTER_CUBIC)
                labels.append(class_idx)
                features.append(_feature_vector(rgb))
    print(f"Added {len(features)} GAN-generated feature samples.")
    return labels, features


def main() -> None:
    args = parse_args()
    classifier = MinimapIconClassifier.from_icons_dir(args.icons, args.base_augmentations_per_icon)
    extra_labels, extra_features = gan_features(args, classifier.names)
    labels = classifier.labels
    features = classifier.features
    if extra_features:
        labels = np.concatenate([labels, np.asarray(extra_labels, dtype=np.int32)])
        features = np.vstack([features, np.vstack(extra_features)])

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out,
        version=np.asarray(1, dtype=np.int32),
        signature=np.asarray(_icons_signature(args.icons)),
        names=np.asarray(classifier.names),
        labels=labels,
        features=features,
    )
    print(f"Wrote {args.out} with {len(features)} feature samples for {len(classifier.names)} champions.")


if __name__ == "__main__":
    main()
