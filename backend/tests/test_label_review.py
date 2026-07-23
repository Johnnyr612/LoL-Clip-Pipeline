from __future__ import annotations

import json

from backend import label_review


def _record(filename: str, *, skipped: bool = False) -> dict:
    return {
        "filename": filename,
        "raw_path": f"C:/raw/{filename}",
        "edit_path": f"D:/edits/{filename}",
        "edit_filename": f"{filename}-edit.mp4",
        "raw_duration": 60.0,
        "edit_duration": 20.0,
        "clip_start": 10.0,
        "clip_end": 30.0,
        "fight_start": 12.0,
        "fight_end": 28.0,
        "segments": {
            "pre_fight_context": [10.0, 12.0],
            "fight": [12.0, 28.0],
            "post_fight_context": [28.0, 30.0],
        },
        "match": {
            "method": "frame_correlation",
            "score": 0.01,
            "confidence": "high",
        },
        "needs_review": False,
        "reviewed": True,
        "skipped": skipped,
        "review_status": "skip" if skipped else "approved",
        "review_note": "approved",
    }


def test_skip_removes_existing_trainer_label(tmp_path, monkeypatch) -> None:
    candidates_path = tmp_path / "fight_label_candidates.json"
    trainer_labels_path = tmp_path / "videomae_labels.json"
    payload = {
        "schema_version": 1,
        "records": [_record("skip-me.mp4"), _record("keep-me.mp4")],
        "unmatched_edits": [],
        "duplicate_raw_stems": [],
    }
    candidates_path.write_text(json.dumps(payload), encoding="utf-8")
    trainer_labels_path.write_text(
        json.dumps(
            [
                {"filename": "skip-me.mp4", "fight_start": 12.0, "fight_end": 28.0},
                {"filename": "keep-me.mp4", "fight_start": 12.0, "fight_end": 28.0},
            ]
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(label_review, "LABEL_CANDIDATES_PATH", candidates_path)
    monkeypatch.setattr(label_review, "TRAINER_LABELS_PATH", trainer_labels_path)

    result = label_review.skip_label_review_record(0)

    updated_candidates = json.loads(candidates_path.read_text(encoding="utf-8"))
    updated_trainer_labels = json.loads(trainer_labels_path.read_text(encoding="utf-8"))
    assert updated_candidates["records"][0]["skipped"] is True
    assert updated_candidates["records"][0]["review_status"] == "skip"
    assert result["summary"]["skipped"] == 1
    assert result["summary"]["approved"] == 1
    assert [item["filename"] for item in updated_trainer_labels] == ["keep-me.mp4"]


def test_save_record_preserves_multiple_fight_segments(tmp_path, monkeypatch) -> None:
    candidates_path = tmp_path / "fight_label_candidates.json"
    trainer_labels_path = tmp_path / "videomae_labels.json"
    payload = {
        "schema_version": 1,
        "records": [_record("multi-fight.mp4")],
        "unmatched_edits": [],
        "duplicate_raw_stems": [],
    }
    candidates_path.write_text(json.dumps(payload), encoding="utf-8")
    trainer_labels_path.write_text("[]", encoding="utf-8")

    monkeypatch.setattr(label_review, "LABEL_CANDIDATES_PATH", candidates_path)
    monkeypatch.setattr(label_review, "TRAINER_LABELS_PATH", trainer_labels_path)

    result = label_review.save_label_review_record(
        0,
        label_review.LabelReviewUpdate(
            clip_start=10.0,
            clip_end=42.0,
            fight_start=12.0,
            fight_end=35.0,
            fight_segments=[[12.0, 18.0], [23.5, 35.0]],
            approved=True,
            review_note="two fights separated by non-combat chase",
        ),
    )

    record = result["record"]
    assert record["fight_start"] == 12.0
    assert record["fight_end"] == 35.0
    assert record["fight_segments"] == [[12.0, 18.0], [23.5, 35.0]]
    assert record["segments"] == {
        "pre_fight_context": [10.0, 12.0],
        "fight": [[12.0, 18.0], [23.5, 35.0]],
        "bridge": [[18.0, 23.5]],
        "post_fight_context": [35.0, 42.0],
    }
    assert record["review_status"] == "approved"
    assert result["summary"]["approved"] == 1

    saved_candidates = json.loads(candidates_path.read_text(encoding="utf-8"))
    saved_trainer_labels = json.loads(trainer_labels_path.read_text(encoding="utf-8"))
    assert saved_candidates["records"][0]["fight_segments"] == [[12.0, 18.0], [23.5, 35.0]]
    assert saved_trainer_labels == [
        {
            "filename": "multi-fight.mp4",
            "raw_path": "C:/raw/multi-fight.mp4",
            "edit_path": "D:/edits/multi-fight.mp4",
            "clip_start": 10.0,
            "clip_end": 42.0,
            "fight_start": 12.0,
            "fight_end": 35.0,
            "fight_segments": [[12.0, 18.0], [23.5, 35.0]],
            "duration": 60.0,
            "source_edit": "multi-fight.mp4-edit.mp4",
            "label_confidence": "high",
            "needs_review": False,
            "skipped": False,
        }
    ]

