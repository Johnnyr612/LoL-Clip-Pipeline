from __future__ import annotations

import argparse
import random
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a small mask GAN for minimap-style champion icon augmentation.")
    parser.add_argument("--real-crops", type=Path, default=Path("data/minimap_gan/real"))
    parser.add_argument("--icons", type=Path, default=Path("data/minimap_icons/images"))
    parser.add_argument("--out", type=Path, default=Path("checkpoints/minimap_mask_gan.pt"))
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    return parser.parse_args()


def require_torch():
    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as functional
        from torch.utils.data import DataLoader, Dataset
    except Exception as exc:  # noqa: BLE001
        raise SystemExit(
            "PyTorch is required for GAN training. Install it first, for example:\n"
            "  .\\.venv\\Scripts\\python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121"
        ) from exc
    return torch, nn, functional, DataLoader, Dataset


def load_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.array(image.convert("RGB").resize((120, 120)))


def to_tensor(rgb: np.ndarray, torch):
    arr = cv2.resize(rgb, (64, 64), interpolation=cv2.INTER_AREA).astype(np.float32) / 127.5 - 1.0
    return torch.from_numpy(arr).permute(2, 0, 1)


def build_models(torch, nn):
    class Generator(nn.Module):
        def __init__(self, latent_dim: int = 64):
            super().__init__()
            self.net = nn.Sequential(
                nn.ConvTranspose2d(latent_dim + 3, 128, 4, 1, 0),
                nn.BatchNorm2d(128),
                nn.ReLU(True),
                nn.ConvTranspose2d(128, 96, 4, 2, 1),
                nn.BatchNorm2d(96),
                nn.ReLU(True),
                nn.ConvTranspose2d(96, 64, 4, 2, 1),
                nn.BatchNorm2d(64),
                nn.ReLU(True),
                nn.ConvTranspose2d(64, 32, 4, 2, 1),
                nn.BatchNorm2d(32),
                nn.ReLU(True),
                nn.ConvTranspose2d(32, 4, 4, 2, 1),
                nn.Tanh(),
            )

        def forward(self, portrait, noise):
            pooled = portrait.mean(dim=(2, 3), keepdim=True)
            z = noise[:, :, None, None]
            mask = self.net(torch.cat([z, pooled], dim=1))
            color = mask[:, :3]
            alpha = (mask[:, 3:4] + 1.0) / 2.0
            return portrait * (1.0 - alpha) + color * alpha

    class Discriminator(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv2d(3, 32, 4, 2, 1),
                nn.LeakyReLU(0.2, True),
                nn.Conv2d(32, 64, 4, 2, 1),
                nn.BatchNorm2d(64),
                nn.LeakyReLU(0.2, True),
                nn.Conv2d(64, 128, 4, 2, 1),
                nn.BatchNorm2d(128),
                nn.LeakyReLU(0.2, True),
                nn.Conv2d(128, 1, 8, 1, 0),
            )

        def forward(self, image):
            return self.net(image).flatten(1)

    return Generator, Discriminator


def make_dataset_class(torch, Dataset):
    class MinimapGanDataset(Dataset):
        def __init__(self, real_dir: Path, icons_dir: Path):
            self.real_paths = sorted(real_dir.glob("*.png"))
            self.icon_paths = sorted(icons_dir.glob("*.png"))
            if not self.real_paths:
                raise SystemExit(f"No real minimap crops found in {real_dir}. Run tools/collect_minimap_gan_crops.py first.")
            if not self.icon_paths:
                raise SystemExit(f"No champion icons found in {icons_dir}.")

        def __len__(self) -> int:
            return max(len(self.real_paths), len(self.icon_paths))

        def __getitem__(self, index: int):
            real = to_tensor(load_rgb(self.real_paths[index % len(self.real_paths)]), torch)
            portrait = to_tensor(load_rgb(random.choice(self.icon_paths)), torch)
            return portrait, real

    return MinimapGanDataset


def main() -> None:
    args = parse_args()
    torch, nn, functional, DataLoader, Dataset = require_torch()
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else args.device)
    if args.device == "auto" and not torch.cuda.is_available():
        device = torch.device("cpu")

    Generator, Discriminator = build_models(torch, nn)
    dataset_class = make_dataset_class(torch, Dataset)
    loader = DataLoader(dataset_class(args.real_crops, args.icons), batch_size=args.batch_size, shuffle=True, drop_last=True)
    generator = Generator().to(device)
    discriminator = Discriminator().to(device)
    g_opt = torch.optim.Adam(generator.parameters(), lr=args.lr, betas=(0.5, 0.999))
    d_opt = torch.optim.Adam(discriminator.parameters(), lr=args.lr, betas=(0.5, 0.999))

    latent_dim = 64
    for epoch in range(args.epochs):
        d_loss_total = 0.0
        g_loss_total = 0.0
        for portrait, real in loader:
            portrait = portrait.to(device)
            real = real.to(device)
            noise = torch.randn(portrait.size(0), latent_dim, device=device)

            fake = generator(portrait, noise).detach()
            real_logits = discriminator(real)
            fake_logits = discriminator(fake)
            d_loss = (
                functional.binary_cross_entropy_with_logits(real_logits, torch.ones_like(real_logits))
                + functional.binary_cross_entropy_with_logits(fake_logits, torch.zeros_like(fake_logits))
            ) / 2.0
            d_opt.zero_grad()
            d_loss.backward()
            d_opt.step()

            noise = torch.randn(portrait.size(0), latent_dim, device=device)
            fake = generator(portrait, noise)
            fake_logits = discriminator(fake)
            g_loss = functional.binary_cross_entropy_with_logits(fake_logits, torch.ones_like(fake_logits))
            g_opt.zero_grad()
            g_loss.backward()
            g_opt.step()

            d_loss_total += float(d_loss.item())
            g_loss_total += float(g_loss.item())
        steps = max(1, len(loader))
        print(f"epoch {epoch + 1}/{args.epochs} d_loss={d_loss_total / steps:.4f} g_loss={g_loss_total / steps:.4f}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"generator": generator.state_dict(), "latent_dim": latent_dim}, args.out)
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
