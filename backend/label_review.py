from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from fastapi import HTTPException
from pydantic import BaseModel

from . import config


LABEL_CANDIDATES_PATH = config.PROJECT_ROOT / "data" / "training" / "fight_label_candidates.json"
TRAINER_LABELS_PATH = config.PROJECT_ROOT / "data" / "training" / "videomae_labels.json"


class LabelReviewUpdate(BaseModel):
    clip_start: float
    clip_end: float
    fight_start: float
    fight_end: float
    fight_segments: list[list[float]] | None = None
    approved: bool = False
    review_note: str = ""


def _load_payload() -> dict[str, Any]:
    if not LABEL_CANDIDATES_PATH.exists():
        raise HTTPException(status_code=404, detail=f"Label file not found: {LABEL_CANDIDATES_PATH}")
    try:
        payload = json.loads(LABEL_CANDIDATES_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=422, detail=f"Label file is not valid JSON: {exc}") from exc
    if not isinstance(payload, dict) or not isinstance(payload.get("records"), list):
        raise HTTPException(status_code=422, detail="Label file must contain a records list")
    return payload


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    temp_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temp_path.replace(path)


def _clip(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def _normalize_fight_segments(
    raw_segments: list[list[float]] | None,
    clip_start: float,
    clip_end: float,
    fallback_start: float,
    fallback_end: float,
) -> list[list[float]]:
    candidates = raw_segments or [[fallback_start, fallback_end]]
    normalized: list[list[float]] = []
    for item in candidates:
        if not isinstance(item, (list, tuple)) or len(item) < 2:
            continue
        start = _clip(float(item[0]), clip_start, clip_end)
        end = _clip(float(item[1]), start, clip_end)
        if end > start:
            normalized.append([round(start, 3), round(end, 3)])

    if not normalized:
        start = _clip(float(fallback_start), clip_start, clip_end)
        end = _clip(float(fallback_end), start, clip_end)
        normalized = (
            [[round(start, 3), round(end, 3)]]
            if end > start
            else [[round(clip_start, 3), round(clip_end, 3)]]
        )

    normalized.sort(key=lambda value: value[0])
    merged: list[list[float]] = []
    for start, end in normalized:
        if merged and start <= merged[-1][1]:
            merged[-1][1] = round(max(merged[-1][1], end), 3)
        else:
            merged.append([start, end])
    return merged


def _segments_from_fights(
    clip_start: float,
    clip_end: float,
    fight_segments: list[list[float]],
) -> dict[str, Any]:
    first_start = fight_segments[0][0]
    last_end = fight_segments[-1][1]
    bridge_segments = [
        [fight_segments[index][1], fight_segments[index + 1][0]]
        for index in range(len(fight_segments) - 1)
        if fight_segments[index + 1][0] > fight_segments[index][1]
    ]
    return {
        "pre_fight_context": [round(float(clip_start), 3), round(float(first_start), 3)],
        "fight": fight_segments,
        "bridge": bridge_segments,
        "post_fight_context": [round(float(last_end), 3), round(float(clip_end), 3)],
    }


def _fight_segments_for_record(record: dict[str, Any]) -> list[list[float]]:
    raw_segments = record.get("fight_segments")
    if raw_segments is None:
        segment_fight = record.get("segments", {}).get("fight")
        if (
            isinstance(segment_fight, list)
            and segment_fight
            and all(isinstance(item, list) for item in segment_fight)
        ):
            raw_segments = segment_fight
        elif isinstance(segment_fight, list) and len(segment_fight) >= 2:
            raw_segments = [[segment_fight[0], segment_fight[1]]]
    return _normalize_fight_segments(
        raw_segments,
        float(record.get("clip_start") or 0.0),
        float(record.get("clip_end") or record.get("raw_duration") or 0.0),
        float(record.get("fight_start") or record.get("clip_start") or 0.0),
        float(record.get("fight_end") or record.get("clip_end") or 0.0),
    )


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
        min(max(0.0, edit_duration - 1.0), max(0.0, edit_duration * 0.85)),
    ]
    offsets: list[float] = []
    for value in candidates:
        rounded = round(value, 3)
        if 0.0 <= rounded <= max(0.0, edit_duration - 0.1) and rounded not in offsets:
            offsets.append(rounded)
    return offsets or [0.0]


def _score_trim_start(raw_path: Path, edit_frames: list[tuple[float, np.ndarray]], start: float) -> float | None:
    distances: list[float] = []
    for offset, edit_frame in edit_frames:
        raw_frame = _read_match_frame(raw_path, start + offset)
        if raw_frame is None:
            continue
        distances.append(_frame_distance(edit_frame, raw_frame))
    if not distances:
        return None
    return float(np.median(distances))


def _estimate_trim_start(
    raw_path: Path,
    edit_path: Path,
    raw_duration: float,
    edit_duration: float,
    search_step: float = 0.5,
) -> tuple[float | None, float | None]:
    max_start = max(0.0, raw_duration - edit_duration)
    edit_frames = [
        (offset, frame)
        for offset in _sample_offsets(edit_duration)
        if (frame := _read_match_frame(edit_path, offset)) is not None
    ]
    if not edit_frames:
        return None, None

    best_start: float | None = None
    best_score = float("inf")
    coarse_step = max(0.1, search_step)
    candidate_count = int(math.floor(max_start / coarse_step)) + 1
    candidates = [round(index * coarse_step, 3) for index in range(candidate_count)]
    if not candidates or candidates[-1] < max_start:
        candidates.append(round(max_start, 3))

    for start in candidates:
        score = _score_trim_start(raw_path, edit_frames, start)
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
        score = _score_trim_start(raw_path, edit_frames, start)
        if score is not None and score < refined_score:
            refined_score = score
            refined_start = start

    return refined_start, refined_score


def _confidence_from_score(score: float | None) -> str:
    if score is None:
        return "unmatched"
    if score <= 0.08:
        return "high"
    if score <= 0.18:
        return "medium"
    return "low"


def _trainer_label(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "filename": record["filename"],
        "raw_path": record.get("raw_path", ""),
        "edit_path": record.get("edit_path", ""),
        "clip_start": record["clip_start"],
        "clip_end": record["clip_end"],
        "fight_start": record["fight_start"],
        "fight_end": record["fight_end"],
        "fight_segments": _fight_segments_for_record(record),
        "duration": record["raw_duration"],
        "source_edit": record["edit_filename"],
        "label_confidence": record.get("match", {}).get("confidence", "reviewed"),
        "needs_review": bool(record.get("needs_review", True)),
        "skipped": bool(record.get("skipped", False)),
    }


def _summary(records: list[dict[str, Any]]) -> dict[str, int]:
    return {
        "total": len(records),
        "approved": sum(
            1
            for record in records
            if not record.get("needs_review", True) and not record.get("skipped", False)
        ),
        "needs_review": sum(1 for record in records if record.get("needs_review", True)),
        "high_confidence": sum(1 for record in records if record.get("match", {}).get("confidence") == "high"),
        "unmatched": sum(1 for record in records if record.get("match", {}).get("confidence") == "unmatched"),
        "skipped": sum(1 for record in records if record.get("skipped", False)),
    }


def _public_payload(payload: dict[str, Any]) -> dict[str, Any]:
    records = payload["records"]
    return {
        "schema_version": payload.get("schema_version", 1),
        "raw_dirs": payload.get("raw_dirs", []),
        "edits_dir": payload.get("edits_dir", ""),
        "defaults": payload.get("defaults", {}),
        "summary": _summary(records),
        "records": records,
        "unmatched_edits": payload.get("unmatched_edits", []),
        "duplicate_raw_stems": payload.get("duplicate_raw_stems", []),
    }


def get_label_review_payload() -> dict[str, Any]:
    return _public_payload(_load_payload())


def save_label_review_record(index: int, update: LabelReviewUpdate) -> dict[str, Any]:
    payload = _load_payload()
    records = payload["records"]
    if index < 0 or index >= len(records):
        raise HTTPException(status_code=404, detail="Label record not found")

    record = records[index]
    raw_duration = float(record.get("raw_duration") or 0.0)
    clip_start = _clip(float(update.clip_start), 0.0, raw_duration)
    clip_end = _clip(float(update.clip_end), clip_start, raw_duration)
    fight_segments = _normalize_fight_segments(
        update.fight_segments,
        clip_start,
        clip_end,
        float(update.fight_start),
        float(update.fight_end),
    )

    record["clip_start"] = round(clip_start, 3)
    record["clip_end"] = round(clip_end, 3)
    record["fight_start"] = fight_segments[0][0]
    record["fight_end"] = fight_segments[-1][1]
    record["fight_segments"] = fight_segments
    record["segments"] = _segments_from_fights(record["clip_start"], record["clip_end"], fight_segments)
    record["needs_review"] = not update.approved
    record["review_note"] = update.review_note.strip()
    if update.approved:
        record["reviewed"] = True
        record["skipped"] = False
        record["review_status"] = "approved"
        record["review_note"] = record["review_note"] or "approved"
    else:
        record["skipped"] = False
        record["review_status"] = "needs_review"

    _write_json(LABEL_CANDIDATES_PATH, payload)
    regenerate_trainer_labels(payload)
    return {"record": record, "summary": _summary(records)}


def skip_label_review_record(index: int) -> dict[str, Any]:
    payload = _load_payload()
    records = payload["records"]
    if index < 0 or index >= len(records):
        raise HTTPException(status_code=404, detail="Label record not found")

    record = records[index]
    record["skipped"] = True
    record["needs_review"] = False
    record["reviewed"] = False
    record["review_status"] = "skip"
    note = str(record.get("review_note") or "").strip()
    record["review_note"] = note or "skipped"

    _write_json(LABEL_CANDIDATES_PATH, payload)
    regenerate_trainer_labels(payload)
    return {"record": record, "summary": _summary(records)}


def match_label_review_record(index: int, search_step: float = 0.5) -> dict[str, Any]:
    payload = _load_payload()
    records = payload["records"]
    if index < 0 or index >= len(records):
        raise HTTPException(status_code=404, detail="Label record not found")

    record = records[index]
    raw_path = validated_video_path(str(record.get("raw_path", "")))
    edit_path = validated_video_path(str(record.get("edit_path", "")))
    raw_duration = float(record.get("raw_duration") or 0.0)
    edit_duration = float(record.get("edit_duration") or 0.0)
    if raw_duration <= 0.0 or edit_duration <= 0.0:
        raise HTTPException(status_code=422, detail="Record is missing usable video duration metadata")
    if edit_duration > raw_duration + 0.5:
        raise HTTPException(status_code=422, detail="Edited video is longer than the raw video")

    trim_start, match_score = _estimate_trim_start(raw_path, edit_path, raw_duration, edit_duration, search_step)
    if trim_start is None:
        raise HTTPException(status_code=422, detail="Unable to match edited reference frames inside the raw video")

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
    confidence = _confidence_from_score(match_score)

    fight_segments = _normalize_fight_segments(
        None,
        clip_start,
        clip_end,
        fight_start,
        fight_end,
    )
    record["clip_start"] = round(float(clip_start), 3)
    record["clip_end"] = round(float(clip_end), 3)
    record["fight_start"] = fight_segments[0][0]
    record["fight_end"] = fight_segments[-1][1]
    record["fight_segments"] = fight_segments
    record["segments"] = _segments_from_fights(record["clip_start"], record["clip_end"], fight_segments)
    record["match"] = {
        "method": "frame_correlation",
        "score": round(float(match_score), 5) if match_score is not None else None,
        "confidence": confidence,
    }
    record["needs_review"] = True
    record["review_note"] = f"pixel matched clip_start from {previous_clip_start:.3f}s to {clip_start:.3f}s"

    _write_json(LABEL_CANDIDATES_PATH, payload)
    regenerate_trainer_labels(payload)
    return {"record": record, "summary": _summary(records)}


def regenerate_trainer_labels(payload: dict[str, Any] | None = None) -> dict[str, Any]:
    payload = payload or _load_payload()
    records = payload["records"]
    trainer_labels = [_trainer_label(record) for record in records if not record.get("skipped", False)]
    _write_json(TRAINER_LABELS_PATH, trainer_labels)
    return {"path": str(TRAINER_LABELS_PATH), "count": len(trainer_labels)}


def validated_video_path(path_value: str) -> Path:
    path = Path(path_value).expanduser()
    if not path.exists() or not path.is_file():
        raise HTTPException(status_code=404, detail="Video file not found")
    if path.suffix.lower() != ".mp4":
        raise HTTPException(status_code=422, detail="Only MP4 review videos are supported")
    return path
