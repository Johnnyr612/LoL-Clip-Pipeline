from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CANDIDATES = PROJECT_ROOT / "data" / "training" / "fight_label_candidates.json"
DEFAULT_TRAINER_LABELS = PROJECT_ROOT / "data" / "training" / "videomae_labels.json"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Pixel-match edited clips back to their raw Medal clips.")
    parser.add_argument("--candidates", type=Path, default=DEFAULT_CANDIDATES)
    parser.add_argument("--trainer-labels-output", type=Path, default=DEFAULT_TRAINER_LABELS)
    parser.add_argument("--search-step-sec", type=float, default=0.5)
    parser.add_argument("--checkpoint-every", type=int, default=10)
    parser.add_argument("--top-candidates", type=int, default=8)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--end-index", type=int, default=None)
    parser.add_argument("--skip-high", action="store_true", help="Skip records that already have high frame-correlation matches.")
    return parser


def _read_frame(capture: cv2.VideoCapture, second: float, size: tuple[int, int] = (160, 90)) -> np.ndarray | None:
    capture.set(cv2.CAP_PROP_POS_MSEC, max(0.0, second) * 1000.0)
    ok, frame_bgr = capture.read()
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
        min(max(0.0, edit_duration - 1.0), max(0.0, edit_duration * 0.85)),
    ]
    offsets: list[float] = []
    for value in candidates:
        rounded = round(value, 3)
        if 0.0 <= rounded <= max(0.0, edit_duration - 0.1) and rounded not in offsets:
            offsets.append(rounded)
    return offsets or [0.0]


def _score_start(raw_capture: cv2.VideoCapture, edit_frames: list[tuple[float, np.ndarray]], start: float) -> float | None:
    distances: list[float] = []
    for offset, edit_frame in edit_frames:
        raw_frame = _read_frame(raw_capture, start + offset)
        if raw_frame is None:
            continue
        distances.append(_frame_distance(edit_frame, raw_frame))
    if not distances:
        return None
    return float(np.median(distances))


def _score_single_frame(raw_capture: cv2.VideoCapture, edit_offset: float, edit_frame: np.ndarray, start: float) -> float | None:
    raw_frame = _read_frame(raw_capture, start + edit_offset)
    if raw_frame is None:
        return None
    return _frame_distance(edit_frame, raw_frame)


def _estimate_trim_start(
    raw_path: Path,
    edit_path: Path,
    raw_duration: float,
    edit_duration: float,
    search_step: float,
    top_candidates: int,
) -> tuple[float | None, float | None]:
    edit_capture = cv2.VideoCapture(str(edit_path))
    raw_capture = cv2.VideoCapture(str(raw_path))
    try:
        if not edit_capture.isOpened() or not raw_capture.isOpened():
            return None, None

        edit_frames = [
            (offset, frame)
            for offset in _sample_offsets(edit_duration)
            if (frame := _read_frame(edit_capture, offset)) is not None
        ]
        if not edit_frames:
            return None, None

        max_start = max(0.0, raw_duration - edit_duration)
        coarse_step = max(0.1, search_step)
        candidate_count = int(math.floor(max_start / coarse_step)) + 1
        candidates = [round(index * coarse_step, 3) for index in range(candidate_count)]
        if not candidates or candidates[-1] < max_start:
            candidates.append(round(max_start, 3))

        primary_offset, primary_frame = edit_frames[0]
        coarse_scores: list[tuple[float, float]] = []
        for start in candidates:
            score = _score_single_frame(raw_capture, primary_offset, primary_frame, start)
            if score is not None:
                coarse_scores.append((score, start))

        if not coarse_scores:
            return None, None

        best_start: float | None = None
        best_score = float("inf")
        for _, start in sorted(coarse_scores)[: max(1, top_candidates)]:
            score = _score_start(raw_capture, edit_frames, start)
            if score is not None and score < best_score:
                best_score = score
                best_start = start

        if best_start is None:
            return None, None

        refined_start = best_start
        refined_score = best_score
        refine_min = max(0.0, best_start - coarse_step)
        refine_max = min(max_start, best_start + coarse_step)
        refine_count = int(math.floor((refine_max - refine_min) / 0.1)) + 1
        for index in range(refine_count + 1):
            start = round(refine_min + index * 0.1, 3)
            if start > refine_max:
                break
            score = _score_start(raw_capture, edit_frames, start)
            if score is not None and score < refined_score:
                refined_score = score
                refined_start = start

        return refined_start, refined_score
    finally:
        edit_capture.release()
        raw_capture.release()


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


def _trainer_label(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "filename": record["filename"],
        "raw_path": record.get("raw_path", ""),
        "edit_path": record.get("edit_path", ""),
        "clip_start": record["clip_start"],
        "clip_end": record["clip_end"],
        "fight_start": record["fight_start"],
        "fight_end": record["fight_end"],
        "fight_segments": record.get("fight_segments") or [[record["fight_start"], record["fight_end"]]],
        "duration": record["raw_duration"],
        "source_edit": record["edit_filename"],
        "label_confidence": record.get("match", {}).get("confidence", "reviewed"),
        "needs_review": bool(record.get("needs_review", True)),
    }


def _write_outputs(payload: dict[str, Any], candidates_path: Path, trainer_labels_path: Path) -> None:
    candidates_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    trainer_labels = [_trainer_label(record) for record in payload["records"]]
    trainer_labels_path.write_text(json.dumps(trainer_labels, indent=2), encoding="utf-8")


def _apply_match(record: dict[str, Any], payload: dict[str, Any], trim_start: float, score: float | None) -> None:
    raw_duration = float(record.get("raw_duration") or 0.0)
    edit_duration = float(record.get("edit_duration") or 0.0)
    previous_clip_start = float(record.get("clip_start") or 0.0)
    previous_clip_end = float(record.get("clip_end") or previous_clip_start)
    previous_fight_start = float(record.get("fight_start") or previous_clip_start)
    previous_fight_end = float(record.get("fight_end") or previous_clip_end)
    default_lead = float(payload.get("defaults", {}).get("pre_fight_context_sec", 2.0))

    lead_in = previous_fight_start - previous_clip_start
    if lead_in < 0.0 or lead_in > max(previous_clip_end - previous_clip_start, 0.0):
        lead_in = default_lead
    post_tail = max(0.0, previous_clip_end - previous_fight_end)

    clip_start = _clip(trim_start, 0.0, raw_duration)
    clip_end = _clip(trim_start + edit_duration, clip_start, raw_duration)
    fight_start = _clip(clip_start + lead_in, clip_start, clip_end)
    fight_end = _clip(clip_end - post_tail, fight_start, clip_end)

    record["clip_start"] = round(float(clip_start), 3)
    record["clip_end"] = round(float(clip_end), 3)
    record["fight_start"] = round(float(fight_start), 3)
    record["fight_end"] = round(float(fight_end), 3)
    record["fight_segments"] = [[record["fight_start"], record["fight_end"]]]
    record["segments"] = {
        "pre_fight_context": [record["clip_start"], record["fight_start"]],
        "fight": [[record["fight_start"], record["fight_end"]]],
        "bridge": [],
        "post_fight_context": [record["fight_end"], record["clip_end"]],
    }
    record["match"] = {
        "method": "frame_correlation",
        "score": round(float(score), 5) if score is not None else None,
        "confidence": _confidence_from_score(score),
    }
    record["needs_review"] = True
    record["review_note"] = f"pixel matched clip_start from {previous_clip_start:.3f}s to {clip_start:.3f}s"


def main() -> int:
    args = _build_parser().parse_args()
    payload = json.loads(args.candidates.read_text(encoding="utf-8"))
    records = payload.get("records", [])
    end_index = len(records) if args.end_index is None else min(args.end_index, len(records))
    processed = 0
    matched = 0
    failed = 0
    started_at = time.time()

    for index in range(max(0, args.start_index), end_index):
        record = records[index]
        match = record.get("match", {})
        if args.skip_high and match.get("method") == "frame_correlation" and match.get("confidence") == "high":
            continue

        raw_path = Path(str(record.get("raw_path", "")))
        edit_path = Path(str(record.get("edit_path", "")))
        raw_duration = float(record.get("raw_duration") or 0.0)
        edit_duration = float(record.get("edit_duration") or 0.0)
        processed += 1

        if not raw_path.exists() or not edit_path.exists() or raw_duration <= 0.0 or edit_duration <= 0.0:
            failed += 1
            print(f"[{index + 1}/{len(records)}] missing/invalid paths: {record.get('filename')}", flush=True)
            continue

        trim_start, score = _estimate_trim_start(
            raw_path,
            edit_path,
            raw_duration,
            edit_duration,
            args.search_step_sec,
            args.top_candidates,
        )
        if trim_start is None:
            failed += 1
            print(f"[{index + 1}/{len(records)}] no match: {record.get('filename')}", flush=True)
            continue

        _apply_match(record, payload, trim_start, score)
        matched += 1
        confidence = record["match"]["confidence"]
        print(
            f"[{index + 1}/{len(records)}] {record['filename']} "
            f"clip_start={record['clip_start']} score={record['match']['score']} confidence={confidence}",
            flush=True,
        )

        if args.checkpoint_every > 0 and matched % args.checkpoint_every == 0:
            _write_outputs(payload, args.candidates, args.trainer_labels_output)
            elapsed = time.time() - started_at
            print(f"checkpoint matched={matched} failed={failed} elapsed={elapsed:.1f}s", flush=True)

    _write_outputs(payload, args.candidates, args.trainer_labels_output)
    elapsed = time.time() - started_at
    print()
    print(f"processed={processed} matched={matched} failed={failed} elapsed={elapsed:.1f}s")
    print(f"wrote_candidates={args.candidates}")
    print(f"wrote_trainer_labels={args.trainer_labels_output}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
