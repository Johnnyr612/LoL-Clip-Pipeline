from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect real LoL minimap champion crops for GAN training.")
    parser.add_argument("--clips", nargs="+", type=Path, required=True, help="Input Medal .mp4 files or directories.")
    parser.add_argument("--out", type=Path, default=Path("data/minimap_gan/real"), help="Output directory for 120x120 real minimap crops.")
    parser.add_argument("--sample-fps", type=float, default=2.0, help="Frames per second to sample from each video.")
    parser.add_argument("--max-crops", type=int, default=20000, help="Stop after this many crops.")
    return parser.parse_args()


def iter_video_paths(paths: list[Path]) -> list[Path]:
    videos: list[Path] = []
    for path in paths:
        if path.is_dir():
            videos.extend(sorted(path.rglob("*.mp4")))
        elif path.suffix.lower() == ".mp4":
            videos.append(path)
    return videos


def minimap_crop(frame_bgr: np.ndarray) -> np.ndarray:
    h, w = frame_bgr.shape[:2]
    return frame_bgr[int(h * 0.75) : h, int(w * 0.82) : w]


def detect_icon_rois(minimap_bgr: np.ndarray) -> list[np.ndarray]:
    gray = cv2.cvtColor(minimap_bgr, cv2.COLOR_BGR2GRAY)
    circles = cv2.HoughCircles(
        gray,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=20,
        param1=50,
        param2=18,
        minRadius=8,
        maxRadius=20,
    )
    if circles is None:
        return []

    rois: list[np.ndarray] = []
    h, w = minimap_bgr.shape[:2]
    for circle in np.round(circles[0]).astype(int):
        x, y, radius = int(circle[0]), int(circle[1]), int(circle[2])
        pad = max(radius + 4, 18)
        x1, y1 = max(0, x - pad), max(0, y - pad)
        x2, y2 = min(w, x + pad), min(h, y + pad)
        roi = minimap_bgr[y1:y2, x1:x2]
        if min(roi.shape[:2], default=0) < 20:
            continue
        rois.append(cv2.resize(roi, (120, 120), interpolation=cv2.INTER_AREA))
    return rois


def collect_from_video(path: Path, out_dir: Path, sample_fps: float, start_index: int, max_crops: int) -> int:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        print(f"warning: could not open {path}")
        return start_index
    fps = capture.get(cv2.CAP_PROP_FPS) or 60.0
    step = max(1, int(round(fps / sample_fps)))
    frame_index = 0
    crop_index = start_index
    while crop_index < max_crops:
        ok, frame = capture.read()
        if not ok:
            break
        if frame_index % step == 0:
            for roi in detect_icon_rois(minimap_crop(frame)):
                cv2.imwrite(str(out_dir / f"real_{crop_index:06d}.png"), roi)
                crop_index += 1
                if crop_index >= max_crops:
                    break
        frame_index += 1
    capture.release()
    return crop_index


def main() -> None:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    videos = iter_video_paths(args.clips)
    if not videos:
        raise SystemExit("No .mp4 files found.")

    crop_count = 0
    for video in videos:
        print(f"Scanning {video}")
        crop_count = collect_from_video(video, args.out, args.sample_fps, crop_count, args.max_crops)
        print(f"  crops: {crop_count}/{args.max_crops}")
        if crop_count >= args.max_crops:
            break
    print(f"Done. Wrote {crop_count} crops to {args.out}")


if __name__ == "__main__":
    main()
