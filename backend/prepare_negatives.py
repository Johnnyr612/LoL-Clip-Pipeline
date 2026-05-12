"""
prepare_negatives.py
====================
Extracts non-fight frames from full 60-second raw clips and
saves them as additional negative training samples.

These supplement the existing trainer_labels_all.json with
extra non-fight windows from the full source clips.

USAGE:
  python backend/prepare_negatives.py
    --clips-dir "D:/Medal/Clips/League of Legends"
    --labels data/trainer_labels_all.json
    --output-labels data/trainer_labels_v2.json
    --precomputed-dir precomputed/
    --max-negatives-per-clip 3
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path
from typing import Optional


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--clips-dir",
        type=Path,
        action="append",
        required=True,
        dest="clips_dirs",
    )
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--output-labels", type=Path, required=True)
    parser.add_argument(
        "--precomputed-dir",
        type=Path,
        default=Path("precomputed"),
    )
    parser.add_argument(
        "--max-negatives-per-clip",
        type=int,
        default=3,
    )
    return parser


if __name__ == "__main__" and any(arg in {"-h", "--help"} for arg in sys.argv[1:]):
    _build_parser().parse_args()

import cv2
import numpy as np


def extract_source_stem(edit_filename: str) -> str:
    """Strip edit suffixes to find source clip stem."""
    stem = Path(edit_filename).stem
    stem = re.sub(r"-trim-\d+$", "", stem, flags=re.IGNORECASE)
    stem = re.sub(r"-tr-edit.*$", "", stem, flags=re.IGNORECASE)
    return stem


def find_source_clip(
    edit_filename: str,
    clips_dirs: list[Path],
) -> Optional[Path]:
    source_stem = extract_source_stem(edit_filename)
    for clips_dir in clips_dirs:
        candidate = clips_dir / f"{source_stem}.mp4"
        if candidate.exists():
            return candidate
    return None


def get_non_fight_windows(
    fight_start: float,
    fight_end: float,
    clip_duration: float,
    window_size: int = 16,
    max_windows: int = 3,
) -> list[tuple[float, float]]:
    """Find non-fight time ranges in a full clip."""
    windows = []

    if fight_start > window_size + 2:
        start = random.uniform(0, fight_start - window_size - 2)
        windows.append((start, start + window_size))

    if clip_duration - fight_end > window_size + 2:
        start = random.uniform(
            fight_end + 2,
            clip_duration - window_size,
        )
        windows.append((start, start + window_size))

    random.shuffle(windows)
    return windows[:max_windows]


def preprocess_clip_segment(
    clip_path: Path,
    start_sec: float,
    duration: float,
    output_path: Path,
) -> bool:
    """Extract frames from a clip segment and save as .npy."""
    capture = cv2.VideoCapture(str(clip_path))
    if not capture.isOpened():
        return False

    fps = float(capture.get(cv2.CAP_PROP_FPS) or 120.0)
    frames = []

    for second in range(int(duration)):
        frame_num = int((start_sec + second) * fps)
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ok, frame = capture.read()
        if ok:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            resized = cv2.resize(
                rgb,
                (224, 224),
                interpolation=cv2.INTER_AREA,
            )
            frames.append(resized)
        elif frames:
            frames.append(frames[-1])
        else:
            frames.append(
                np.zeros((224, 224, 3), dtype=np.uint8)
            )

    capture.release()
    if frames:
        np.save(str(output_path), np.array(frames, dtype=np.uint8))
        return True
    return False


def main() -> int:
    args = _build_parser().parse_args()

    args.precomputed_dir.mkdir(parents=True, exist_ok=True)

    with args.labels.open() as handle:
        original_labels = json.load(handle)

    new_labels = list(original_labels)
    added = 0
    total = len(original_labels)

    for index, label in enumerate(original_labels, 1):
        bar_filled = int((index / total) * 40) if total else 40
        bar = "#" * bar_filled + "." * (40 - bar_filled)
        print(f"\r[{bar}] {index}/{total}", end="", flush=True)

        source = find_source_clip(label["filename"], args.clips_dirs)
        if source is None:
            continue

        capture = cv2.VideoCapture(str(source))
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 120.0)
        total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps
        capture.release()

        windows = get_non_fight_windows(
            label["fight_start"],
            label["fight_end"],
            duration,
            max_windows=args.max_negatives_per_clip,
        )

        for negative_index, (win_start, _win_end) in enumerate(windows):
            stem = extract_source_stem(label["filename"])
            neg_filename = f"{stem}_neg_{negative_index}.mp4"
            npy_path = args.precomputed_dir / f"{stem}_neg_{negative_index}.npy"

            if not npy_path.exists():
                success = preprocess_clip_segment(
                    source,
                    win_start,
                    16,
                    npy_path,
                )
                if not success:
                    continue

            new_labels.append(
                {
                    "filename": neg_filename,
                    "fight_start": 0.0,
                    "fight_end": 0.0,
                    "_is_synthetic_negative": True,
                    "_source_clip": source.name,
                    "_window_start": win_start,
                }
            )
            added += 1

    args.output_labels.parent.mkdir(parents=True, exist_ok=True)
    with args.output_labels.open("w") as handle:
        json.dump(new_labels, handle, indent=2)

    print(f"\nDone. Added {added} negative samples.")
    print(f"Output labels: {args.output_labels}")
    print(f"Total labels: {len(new_labels)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
