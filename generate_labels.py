"""
generate_labels.py
==================
Matches trimmed edit clips to their full 1-minute source clips and extracts
fight_start / fight_end timestamps by finding where the edit appears in the
full clip using frame fingerprinting via FFmpeg.

HOW IT WORKS:
  1. Scans Edits folder for .mp4 files
  2. Finds the matching full clip in C:\\Medal\\Clips or D:\\Medal\\Clips
     by matching the filename prefix (strips "-tr-edit" or similar suffix)
  3. Extracts a fingerprint frame from the first second of the edit
  4. Scans the full clip to find that frame → this is fight_start
  5. fight_end = fight_start + duration of the edit clip
  6. Writes labels_all.json ready for trainer.py

REQUIREMENTS:
  pip install opencv-python numpy
  FFmpeg must be installed and on PATH (winget install ffmpeg)

USAGE:
  python generate_labels.py
  python generate_labels.py --edits "D:\\Medal\\Edits" --output "labels_all.json"
  python generate_labels.py --dry-run   (preview matches without writing)
"""

import argparse
import json
import logging
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Config — adjust these if your folder layout differs
# ---------------------------------------------------------------------------
DEFAULT_EDITS_DIR   = Path(r"D:\Medal\Edits")
DEFAULT_CLIPS_DIRS  = [
    Path(r"C:\Medal\Clips\League of Legends"),
    Path(r"D:\Medal\Clips\League of Legends"),
]
DEFAULT_OUTPUT      = Path(r"D:\Medal\labels_all.json")

# How many frames to extract per second when scanning the full clip
SCAN_FPS = 2

# Size to downscale frames to for fast comparison (width, height)
FINGERPRINT_SIZE = (160, 90)

# Match threshold — lower = stricter. 0.85 works well for same-source video.
MATCH_THRESHOLD = 0.85

# How many seconds into the edit to sample the fingerprint frame
# (avoid frame 0 which may be a fade-in)
FINGERPRINT_OFFSET_SEC = 1.0

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# FFmpeg helpers
# ---------------------------------------------------------------------------

def get_video_duration(path: Path) -> float:
    """Return video duration in seconds using ffprobe."""
    cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(path)
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe failed on {path.name}: {result.stderr}")
    return float(result.stdout.strip())


def extract_frame_at(path: Path, timestamp: float) -> np.ndarray:
    """Extract a single frame at the given timestamp, return as numpy array."""
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        tmp_path = Path(tmp.name)

    cmd = [
        "ffmpeg", "-y",
        "-ss", str(timestamp),
        "-i", str(path),
        "-vframes", "1",
        "-q:v", "2",
        str(tmp_path)
    ]
    result = subprocess.run(cmd, capture_output=True)
    if result.returncode != 0 or not tmp_path.exists():
        raise RuntimeError(f"Failed to extract frame from {path.name} at {timestamp}s")

    frame = cv2.imread(str(tmp_path))
    tmp_path.unlink(missing_ok=True)

    if frame is None:
        raise RuntimeError(f"Could not decode frame from {path.name}")

    return cv2.resize(frame, FINGERPRINT_SIZE)


def extract_frames_from_video(path: Path, fps: float = SCAN_FPS) -> list[tuple[float, np.ndarray]]:
    """
    Extract frames from full video at SCAN_FPS rate.
    Returns list of (timestamp_seconds, frame_array).
    Uses OpenCV VideoCapture for speed.
    """
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {path.name}")

    source_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(1, int(source_fps / fps))

    frames = []
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % step == 0:
            timestamp = frame_idx / source_fps
            small = cv2.resize(frame, FINGERPRINT_SIZE)
            frames.append((timestamp, small))
        frame_idx += 1

    cap.release()
    log.debug("  Extracted %d scan frames from %s (source %.1f fps, step=%d)",
              len(frames), path.name, source_fps, step)
    return frames


# ---------------------------------------------------------------------------
# Frame matching
# ---------------------------------------------------------------------------

def frame_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """
    Compute normalized cross-correlation between two frames.
    Returns value in [0, 1] where 1 = identical.
    """
    a_f = a.astype(np.float32)
    b_f = b.astype(np.float32)

    # Normalized cross-correlation
    num = np.sum(a_f * b_f)
    denom = np.sqrt(np.sum(a_f ** 2) * np.sum(b_f ** 2))
    if denom < 1e-6:
        return 0.0
    return float(num / denom)


def find_edit_in_full_clip(
    edit_path: Path,
    full_path: Path,
) -> tuple[float, float, float]:
    """
    Find where the edit clip appears in the full clip.

    Returns: (fight_start, fight_end, match_score)
    Raises: RuntimeError if no match found above threshold.
    """
    edit_duration = get_video_duration(edit_path)

    # Get fingerprint frame from edit (1 second in, avoids fade-in artifacts)
    fingerprint_ts = min(FINGERPRINT_OFFSET_SEC, edit_duration * 0.1)
    fingerprint = extract_frame_at(edit_path, fingerprint_ts)

    log.debug("  Edit duration: %.2fs, fingerprint at %.2fs", edit_duration, fingerprint_ts)

    # Scan full clip
    scan_frames = extract_frames_from_video(full_path, SCAN_FPS)

    # Find best matching frame
    best_score = 0.0
    best_timestamp = 0.0

    for ts, frame in scan_frames:
        score = frame_similarity(fingerprint, frame)
        if score > best_score:
            best_score = score
            best_timestamp = ts

    if best_score < MATCH_THRESHOLD:
        raise RuntimeError(
            f"No match found (best score={best_score:.3f}, threshold={MATCH_THRESHOLD}). "
            f"Edit: {edit_path.name}"
        )

    # fight_start = timestamp of matching frame minus fingerprint offset
    fight_start = max(0.0, best_timestamp - fingerprint_ts)
    fight_end = fight_start + edit_duration

    # Clamp fight_end to full clip duration
    full_duration = get_video_duration(full_path)
    fight_end = min(fight_end, full_duration)

    return round(fight_start, 2), round(fight_end, 2), best_score


# ---------------------------------------------------------------------------
# Filename matching
# ---------------------------------------------------------------------------

# Medal edit filename format:
#   MedalTVLeagueofLegends20250506180210-tr-edit.mp4
#   MedalTVLeagueofLegends20250430133650-tr-edit-tr-edit.mp4
#
# Medal source filename format:
#   MedalTVLeagueofLegends20250506180210.mp4
#   MedalTVLeagueofLegends20250430133650.mp4
#
# Rule: strip ALL occurrences of "-tr-edit" from the end of the stem.
# The remaining string is the exact source filename stem.

import re as _re

def extract_source_stem(edit_stem: str) -> str:
    """
    Strip Medal's edit suffixes to recover the original clip stem.

    Handles all known Medal edit suffix formats:
      Format 1: {timestamp}-tr-edit
      Format 2: {timestamp}-tr-edit-tr-edit   (double processed)
      Format 3: {timestamp}-tr-edit-{code}    (e.g. -tr-edit-ba27)
      Format 4: {timestamp}-trim-{number}     (newer Medal format)

    Examples:
      MedalTVLoL20250506180210-tr-edit              → MedalTVLoL20250506180210
      MedalTVLoL20250430133650-tr-edit-tr-edit      → MedalTVLoL20250430133650
      MedalTVLoL20260403230410260-tr-edit-ba27      → MedalTVLoL20260403230410260
      MedalTVLoL20260414160207762-trim-1776229111576→ MedalTVLoL20260414160207762
      MedalTVLoL20250411125038-tr-edit_Game         → MedalTVLoL20250411125038
    """
    stem = edit_stem

    # Format 4: -trim-{digits} (newer Medal export format)
    stem = _re.sub(r'-trim-\d+$', '', stem, flags=_re.IGNORECASE)

    # Format 3+2+1: strip all -tr-edit variants from the end
    # This handles: -tr-edit, -tr-edit-tr-edit, -tr-edit-{code}, -tr-edit_Game
    stem = _re.sub(r'-tr-edit.*$', '', stem, flags=_re.IGNORECASE)

    return stem


def find_matching_full_clip(
    edit_path: Path,
    clips_dirs: list[Path]
) -> Path | None:
    """
    Find the full 1-minute clip that corresponds to this edit.

    Medal naming convention:
      Edit formats:
        {timestamp}-tr-edit.mp4
        {timestamp}-tr-edit-tr-edit.mp4
        {timestamp}-tr-edit-{code}.mp4
        {timestamp}-trim-{number}.mp4
      Source: {timestamp}.mp4

    Searches all provided clips directories.
    """
    source_stem = extract_source_stem(edit_path.stem)
    source_filename = source_stem + ".mp4"

    log.debug("  Edit stem   : %s", edit_path.stem)
    log.debug("  Source stem : %s", source_stem)
    log.debug("  Looking for : %s", source_filename)

    for clips_dir in clips_dirs:
        if not clips_dir.exists():
            log.debug("  Clips dir not found, skipping: %s", clips_dir)
            continue

        # Exact match first (fast)
        candidate = clips_dir / source_filename
        if candidate.exists():
            log.debug("  Found at: %s", candidate)
            return candidate

        # Fallback: source stem might have extra chars in some Medal versions
        # Check if any clip file starts with source_stem
        for clip in clips_dir.glob(f"{source_stem}*.mp4"):
            # Only accept if the extra chars are non-alphanumeric (not another clip)
            extra = clip.stem[len(source_stem):]
            if not extra or not extra[0].isdigit():
                log.debug("  Found via prefix match: %s", clip)
                return clip

    return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate labels_all.json by matching edits to full clips."
    )
    p.add_argument(
        "--edits", type=Path, default=DEFAULT_EDITS_DIR,
        help=f"Folder containing trimmed edit .mp4 files (default: {DEFAULT_EDITS_DIR})"
    )
    p.add_argument(
        "--clips", nargs="+", type=Path, default=DEFAULT_CLIPS_DIRS,
        help="One or more folders containing full 1-minute clips"
    )
    p.add_argument(
        "--output", type=Path, default=DEFAULT_OUTPUT,
        help=f"Output labels JSON path (default: {DEFAULT_OUTPUT})"
    )
    p.add_argument(
        "--threshold", type=float, default=MATCH_THRESHOLD,
        help=f"Match confidence threshold 0-1 (default: {MATCH_THRESHOLD})"
    )
    p.add_argument(
        "--dry-run", action="store_true",
        help="Preview filename matches without doing frame analysis"
    )
    p.add_argument(
        "--verbose", action="store_true",
        help="Show debug output"
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    global MATCH_THRESHOLD
    MATCH_THRESHOLD = args.threshold

    # Find all edit MP4s
    if not args.edits.exists():
        log.error("Edits directory not found: %s", args.edits)
        return 1

    edit_files = sorted([
        f for f in args.edits.glob("*.mp4")
        # Skip files that are clearly not edits (thumbnail files, etc.)
        if not f.name.startswith(".")
    ])

    if not edit_files:
        log.error("No .mp4 files found in %s", args.edits)
        return 1

    log.info("Found %d edit clips in %s", len(edit_files), args.edits)
    log.info("Searching for full clips in: %s",
             ", ".join(str(d) for d in args.clips))

    labels = []
    matched = 0
    unmatched_files = []
    failed_files = []

    for i, edit_path in enumerate(edit_files, 1):
        log.info("[%d/%d] Processing: %s", i, len(edit_files), edit_path.name)

        # Step 1: Find matching full clip by filename
        full_clip = find_matching_full_clip(edit_path, args.clips)

        if full_clip is None:
            log.warning("  ✗ No matching full clip found for: %s", edit_path.name)
            unmatched_files.append(edit_path.name)
            # Still add to labels with placeholder values so user can fill manually
            labels.append({
                "filename": edit_path.name,
                "fight_start": 0.0,
                "fight_end": 0.0,
                "match_status": "no_source_found",
                "source_clip": None,
            })
            continue

        log.info("  → Matched to: %s", full_clip.name)

        if args.dry_run:
            log.info("  [DRY RUN] Skipping frame analysis")
            labels.append({
                "filename": edit_path.name,
                "fight_start": 0.0,
                "fight_end": 0.0,
                "match_status": "dry_run",
                "source_clip": full_clip.name,
            })
            continue

        # Step 2: Frame fingerprint matching
        try:
            fight_start, fight_end, score = find_edit_in_full_clip(
                edit_path, full_clip
            )
            log.info("  ✓ fight_start=%.2fs  fight_end=%.2fs  score=%.3f",
                     fight_start, fight_end, score)
            labels.append({
                "filename": edit_path.name,
                "fight_start": fight_start,
                "fight_end": fight_end,
                "match_status": "ok",
                "source_clip": full_clip.name,
                "match_score": round(score, 3),
            })
            matched += 1

        except RuntimeError as e:
            log.warning("  ✗ Frame match failed: %s", e)
            failed_files.append(edit_path.name)
            # Add with placeholder — user can fill manually
            labels.append({
                "filename": edit_path.name,
                "fight_start": 0.0,
                "fight_end": 0.0,
                "match_status": "frame_match_failed",
                "source_clip": full_clip.name,
            })

    # Write output
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(labels, f, indent=2)

    # Summary
    print("\n" + "=" * 60)
    print(f"  Total edit clips found : {len(edit_files)}")
    print(f"  Successfully matched   : {matched}")
    print(f"  No source clip found   : {len(unmatched_files)}")
    print(f"  Frame match failed     : {len(failed_files)}")
    print(f"  Output written to      : {args.output}")
    print("=" * 60)

    if unmatched_files:
        print("\nClips with no matching source (fill manually):")
        for f in unmatched_files:
            print(f"  - {f}")

    if failed_files:
        print("\nClips where frame matching failed (lower --threshold to retry):")
        for f in failed_files:
            print(f"  - {f}")

    if matched > 0:
        print(f"\n✓ labels_all.json is ready with {matched} auto-labeled clips.")
        print("  Review match_status='ok' entries before training.")
        print("  Manually fill fight_start/fight_end for any status != 'ok'.")

    # Write a clean version for the trainer (no metadata fields)
    trainer_labels = [
        {
            "filename": l["filename"],
            "fight_start": l["fight_start"],
            "fight_end": l["fight_end"],
        }
        for l in labels
        if l["match_status"] == "ok"
    ]
    trainer_output = args.output.parent / ("trainer_" + args.output.name)
    with open(trainer_output, "w", encoding="utf-8") as f:
        json.dump(trainer_labels, f, indent=2)
    print(f"\n  Trainer-ready file     : {trainer_output}")
    print(f"  ({len(trainer_labels)} clips with confirmed timestamps)\n")

    return 0 if (matched == len(edit_files)) else 1


if __name__ == "__main__":
    sys.exit(main())
