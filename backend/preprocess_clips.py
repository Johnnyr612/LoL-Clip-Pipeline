from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clips-dir", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("precomputed"))
    return parser


if __name__ == "__main__" and any(arg in {"-h", "--help"} for arg in sys.argv[1:]):
    _build_parser().parse_args()

import cv2
import numpy as np


def preprocess(clips_dir: Path, labels_path: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    with labels_path.open() as handle:
        labels = json.load(handle)

    valid_clips = [
        label
        for label in labels
        if (clips_dir / label["filename"]).exists()
    ]
    total = len(valid_clips)

    for index, label in enumerate(valid_clips, 1):
        clip_path = clips_dir / label["filename"]
        out_path = output_dir / (clip_path.stem + ".npy")

        bar_filled = int((index / total) * 40) if total else 40
        bar = "█" * bar_filled + "░" * (40 - bar_filled)
        print(
            f"\r[{bar}] {index}/{total} — {clip_path.name[:50]}",
            end="",
            flush=True,
        )

        if out_path.exists():
            continue

        capture = cv2.VideoCapture(str(clip_path))
        if not capture.isOpened():
            print(f"\n  WARNING: Cannot open {clip_path.name}, skipping")
            continue

        source_fps = float(capture.get(cv2.CAP_PROP_FPS) or 120.0)
        if source_fps <= 0:
            source_fps = 120.0
        sample_every = max(1, int(source_fps))

        frames = []
        frame_index = 0
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if frame_index % sample_every == 0:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                resized = cv2.resize(
                    rgb,
                    (224, 224),
                    interpolation=cv2.INTER_AREA,
                )
                frames.append(resized)
            frame_index += 1
        capture.release()

        if frames:
            np.save(str(out_path), np.array(frames, dtype=np.uint8))
        else:
            print(f"\n  WARNING: No frames extracted from {clip_path.name}")

    print(f"\nDone. Preprocessed {total} clips to {output_dir}")


if __name__ == "__main__":
    args = _build_parser().parse_args()
    preprocess(args.clips_dir, args.labels, args.output_dir)
