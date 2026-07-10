from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path

import cv2
import numpy as np


DEFAULT_EDIT_SUFFIX_PATTERNS = (
    r"-trim-\d+$",
    r"-tr-edit(?:-tr-edit)?(?:_Game)?$",
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Match Medal raw clips to edited trims and create reviewable "
            "VideoMAE fight-label candidates."
        )
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        nargs="+",
        required=True,
        help="One or more folders containing original 1-minute Medal .mp4 clips.",
    )
    parser.add_argument(
        "--edits-dir",
        type=Path,
        required=True,
        help="Folder containing manually trimmed/edited .mp4 clips.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/training/fight_label_candidates.json"),
        help="JSON file to write.",
    )
    parser.add_argument(
        "--trainer-labels-output",
        type=Path,
        default=Path("data/training/videomae_labels.json"),
        help="Trainer-compatible labels JSON to write.",
    )
    parser.add_argument(
        "--default-pre-fight-sec",
        type=float,
        default=2.0,
        help="Assumed edited lead-in before first contact.",
    )
    parser.add_argument(
        "--default-post-fight-sec",
        type=float,
        default=0.0,
        help="Assumed edited post-fight/dialog tail after meaningful combat.",
    )
    parser.add_argument(
        "--search-step-sec",
        type=float,
        default=0.5,
        help="Raw-video timestamp step used while matching edit start.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional max number of matched edits to process while testing.",
    )
    parser.add_argument(
        "--only-stem",
        default="",
        help="Optional raw clip stem to process, useful for checking one pair.",
    )
    parser.add_argument(
        "--skip-frame-match",
        action="store_true",
        help="Only match names and durations; do not estimate trim offsets.",
    )
    return parser

def _duration_seconds(path: Path) -> float | None:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        return None
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    frame_count = float(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0)
    capture.release()
    if fps <= 0.0 or frame_count <= 0.0:
        return None
    return frame_count / fps


def _base_name_for_edit(edit_stem: str) -> str:
    for pattern in DEFAULT_EDIT_SUFFIX_PATTERNS:
        stripped = re.sub(pattern, "", edit_stem, flags=re.IGNORECASE)
        if stripped != edit_stem:
            return stripped
    return edit_stem


def _read_match_frame(path: Path, second: float, size: tuple[int, int] = (160, 90)) -> np.ndarray | None:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        return None
    capture.set(cv2.CAP_PROP_POS_MSEC, max(0.0, second) * 1000.0)
    ok, frame_bgr = capture.read()
    capture.release()
    if not ok:
        return None
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, size, interpolation=cv2.INTER_AREA)
    equalized = cv2.equalizeHist(resized)
    return equalized.astype(np.float32)


def _frame_distance(left: np.ndarray, right: np.ndarray) -> float:
    left_norm = left - float(left.mean())
    right_norm = right - float(right.mean())
    denom = float(np.linalg.norm(left_norm) * np.linalg.norm(right_norm))
    if denom <= 1e-6:
        return 1.0
    corr = float(np.sum(left_norm * right_norm) / denom)
    return 1.0 - corr


def _sample_offsets(edit_duration: float) -> list[float]:
    candidates = [
        0.25,
        min(1.0, max(0.0, edit_duration - 0.25)),
        min(3.0, max(0.0, edit_duration * 0.25)),
        min(6.0, max(0.0, edit_duration * 0.50)),
    ]
    offsets: list[float] = []
    for value in candidates:
        if 0.0 <= value <= max(0.0, edit_duration - 0.1) and value not in offsets:
            offsets.append(value)
    return offsets or [0.0]


def _estimate_trim_start(
    raw_path: Path,
    edit_path: Path,
    raw_duration: float,
    edit_duration: float,
    search_step: float,
) -> tuple[float | None, float | None]:
    max_start = max(0.0, raw_duration - edit_duration)
    offsets = _sample_offsets(edit_duration)
    edit_frames = [
        (offset, frame)
        for offset in offsets
        if (frame := _read_match_frame(edit_path, offset)) is not None
    ]
    if not edit_frames:
        return None, None

    best_start: float | None = None
    best_score = float("inf")
    step = max(0.1, search_step)
    candidate_count = int(math.floor(max_start / step)) + 1
    candidates = [round(index * step, 3) for index in range(candidate_count)]
    if not candidates or candidates[-1] < max_start:
        candidates.append(round(max_start, 3))

    for start in candidates:
        distances: list[float] = []
        for offset, edit_frame in edit_frames:
            raw_frame = _read_match_frame(raw_path, start + offset)
            if raw_frame is None:
                continue
            distances.append(_frame_distance(edit_frame, raw_frame))
        if not distances:
            continue
        score = float(np.median(distances))
        if score < best_score:
            best_score = score
            best_start = start

    return best_start, best_score if best_start is not None else None


def _confidence_from_score(score: float | None) -> str:
    if score is None:
        return "unmatched"
    if score <= 0.08:
        return "high"
    if score <= 0.18:
        return "medium"
    return "low"


def _clip(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def _candidate_record(
    raw_path: Path,
    edit_path: Path,
    raw_duration: float | None,
    edit_duration: float | None,
    trim_start: float | None,
    match_score: float | None,
    default_pre_fight_sec: float,
    default_post_fight_sec: float,
    match_method: str = "frame_correlation",
) -> dict:
    if raw_duration is None:
        raw_duration = 0.0
    if edit_duration is None:
        edit_duration = 0.0
    if trim_start is None:
        trim_start = 0.0
        needs_review_reason = "trim_start_unmatched"
    else:
        needs_review_reason = ""

    clip_start = _clip(trim_start, 0.0, raw_duration)
    clip_end = _clip(trim_start + edit_duration, clip_start, raw_duration)
    fight_start = _clip(clip_start + default_pre_fight_sec, clip_start, clip_end)
    fight_end = _clip(clip_end - default_post_fight_sec, fight_start, clip_end)
    confidence = _confidence_from_score(match_score)

    return {
        "filename": raw_path.name,
        "raw_path": str(raw_path),
        "edit_path": str(edit_path),
        "edit_filename": edit_path.name,
        "raw_duration": round(float(raw_duration), 3),
        "edit_duration": round(float(edit_duration), 3),
        "clip_start": round(float(clip_start), 3),
        "clip_end": round(float(clip_end), 3),
        "fight_start": round(float(fight_start), 3),
        "fight_end": round(float(fight_end), 3),
        "fight_segments": [[round(float(fight_start), 3), round(float(fight_end), 3)]],
        "segments": {
            "pre_fight_context": [round(float(clip_start), 3), round(float(fight_start), 3)],
            "fight": [[round(float(fight_start), 3), round(float(fight_end), 3)]],
            "bridge": [],
            "post_fight_context": [round(float(fight_end), 3), round(float(clip_end), 3)],
        },
        "match": {
            "method": match_method,
            "score": round(float(match_score), 5) if match_score is not None else None,
            "confidence": confidence,
        },
        "needs_review": confidence != "high" or bool(needs_review_reason),
        "review_note": needs_review_reason,
    }


def _trainer_label(record: dict) -> dict:
    return {
        "filename": record["filename"],
        "fight_start": record["fight_start"],
        "fight_end": record["fight_end"],
        "fight_segments": record.get("fight_segments") or [[record["fight_start"], record["fight_end"]]],
        "duration": record["raw_duration"],
        "source_edit": record["edit_filename"],
        "label_confidence": record["match"]["confidence"],
        "needs_review": record["needs_review"],
    }


def main() -> int:
    args = _build_parser().parse_args()
    raw_dirs = [path.expanduser().resolve() for path in args.raw_dir]
    edits_dir = args.edits_dir.expanduser().resolve()
    missing_raw_dirs = [path for path in raw_dirs if not path.exists()]
    if missing_raw_dirs:
        raise SystemExit(f"raw dir not found: {missing_raw_dirs[0]}")
    if not edits_dir.exists():
        raise SystemExit(f"edits dir not found: {edits_dir}")

    raw_by_stem: dict[str, Path] = {}
    duplicate_raw_stems: list[str] = []
    for raw_dir in raw_dirs:
        for path in raw_dir.glob("*.mp4"):
            if path.stem in raw_by_stem:
                duplicate_raw_stems.append(path.stem)
                continue
            raw_by_stem[path.stem] = path
    edit_paths = sorted(edits_dir.glob("*.mp4"))

    records: list[dict] = []
    unmatched_edits: list[str] = []
    for edit_path in edit_paths:
        raw_stem = _base_name_for_edit(edit_path.stem)
        if args.only_stem and raw_stem != args.only_stem:
            continue
        raw_path = raw_by_stem.get(raw_stem)
        if raw_path is None:
            unmatched_edits.append(edit_path.name)
            continue
        if args.limit is not None and len(records) >= args.limit:
            break

        raw_duration = _duration_seconds(raw_path)
        edit_duration = _duration_seconds(edit_path)
        trim_start = None
        match_score = None
        match_method = "frame_correlation"
        if (
            args.skip_frame_match
            and raw_duration is not None
            and edit_duration is not None
            and edit_duration <= raw_duration + 0.5
        ):
            trim_start = max(0.0, raw_duration - edit_duration)
            match_method = "duration_tail_assumption"
        elif (
            not args.skip_frame_match
            and raw_duration is not None
            and edit_duration is not None
            and edit_duration <= raw_duration + 0.5
        ):
            trim_start, match_score = _estimate_trim_start(
                raw_path,
                edit_path,
                raw_duration,
                edit_duration,
                args.search_step_sec,
            )

        record = _candidate_record(
            raw_path,
            edit_path,
            raw_duration,
            edit_duration,
            trim_start,
            match_score,
            args.default_pre_fight_sec,
            args.default_post_fight_sec,
            match_method,
        )
        records.append(record)
        print(
            f"{edit_path.name} -> {raw_path.name} "
            f"clip={record['clip_start']}-{record['clip_end']} "
            f"fight={record['fight_start']}-{record['fight_end']} "
            f"match={record['match']['confidence']}",
            flush=True,
        )

    payload = {
        "schema_version": 1,
        "raw_dir": str(raw_dirs[0]) if raw_dirs else "",
        "raw_dirs": [str(path) for path in raw_dirs],
        "edits_dir": str(edits_dir),
        "defaults": {
            "pre_fight_context_sec": args.default_pre_fight_sec,
            "post_fight_context_sec": args.default_post_fight_sec,
        },
        "records": records,
        "unmatched_edits": unmatched_edits,
        "duplicate_raw_stems": sorted(set(duplicate_raw_stems)),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    trainer_labels = [_trainer_label(record) for record in records]
    args.trainer_labels_output.parent.mkdir(parents=True, exist_ok=True)
    args.trainer_labels_output.write_text(json.dumps(trainer_labels, indent=2), encoding="utf-8")

    print()
    print(f"matched_records={len(records)} unmatched_edits={len(unmatched_edits)}")
    print(f"wrote_candidates={args.output}")
    print(f"wrote_trainer_labels={args.trainer_labels_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())






