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


def _pending_record(filename: str) -> dict:
    record = _record(filename)
    record["needs_review"] = True
    record["reviewed"] = False
    record["review_status"] = "needs_review"
    return record


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


def test_delete_removes_candidate_and_regenerates_trainer_labels(tmp_path, monkeypatch) -> None:
    candidates_path = tmp_path / "fight_label_candidates.json"
    trainer_labels_path = tmp_path / "videomae_labels.json"
    payload = {
        "schema_version": 1,
        "records": [_record("delete-me.mp4"), _record("keep-me.mp4")],
        "unmatched_edits": [],
        "duplicate_raw_stems": [],
    }
    candidates_path.write_text(json.dumps(payload), encoding="utf-8")
    trainer_labels_path.write_text(
        json.dumps(
            [
                {"filename": "delete-me.mp4", "fight_start": 12.0, "fight_end": 28.0},
                {"filename": "keep-me.mp4", "fight_start": 12.0, "fight_end": 28.0},
            ]
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(label_review, "LABEL_CANDIDATES_PATH", candidates_path)
    monkeypatch.setattr(label_review, "TRAINER_LABELS_PATH", trainer_labels_path)

    result = label_review.delete_label_review_record(0)

    updated_candidates = json.loads(candidates_path.read_text(encoding="utf-8"))
    updated_trainer_labels = json.loads(trainer_labels_path.read_text(encoding="utf-8"))
    assert result["deleted_record"]["filename"] == "delete-me.mp4"
    assert [item["filename"] for item in updated_candidates["records"]] == ["keep-me.mp4"]
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


def test_public_payload_marks_new_raw_files_as_holdout_candidates(tmp_path, monkeypatch) -> None:
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    trained_file = raw_dir / "trained.mp4"
    new_file = raw_dir / "new-eval.mp4"
    trained_file.write_bytes(b"trained")
    new_file.write_bytes(b"new")
    candidates_path = tmp_path / "fight_label_candidates.json"
    trainer_labels_path = tmp_path / "videomae_labels.json"
    payload = {
        "schema_version": 1,
        "raw_dirs": [str(raw_dir)],
        "records": [_record("trained.mp4")],
        "unmatched_edits": [],
        "duplicate_raw_stems": [],
    }
    candidates_path.write_text(json.dumps(payload), encoding="utf-8")
    trainer_labels_path.write_text(
        json.dumps([{"filename": "trained.mp4", "raw_path": str(trained_file)}]),
        encoding="utf-8",
    )

    monkeypatch.setattr(label_review, "LABEL_CANDIDATES_PATH", candidates_path)
    monkeypatch.setattr(label_review, "TRAINER_LABELS_PATH", trainer_labels_path)
    monkeypatch.setattr(label_review.config, "RAW_CLIP_SOURCE_DIR", raw_dir)

    result = label_review.get_label_review_payload()

    inventory = result["raw_file_inventory"]
    statuses = {item["filename"]: item["status"] for item in inventory["files"]}
    assert statuses["trained.mp4"] == "approved_for_training"
    assert statuses["new-eval.mp4"] == "new_holdout_candidate"
    assert inventory["summary"]["used_for_training"] == 1
    assert inventory["summary"]["approved_for_training"] == 1
    assert inventory["summary"]["new_holdout_candidates"] == 1


def test_public_payload_marks_tiktok_used_raw_files_as_unselectable(tmp_path, monkeypatch) -> None:
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    inbox_file = raw_dir / "sent.mp4"
    posted_file = raw_dir / "posted.mp4"
    new_file = raw_dir / "new.mp4"
    inbox_file.write_bytes(b"inbox")
    posted_file.write_bytes(b"posted")
    new_file.write_bytes(b"new")
    candidates_path = tmp_path / "fight_label_candidates.json"
    trainer_labels_path = tmp_path / "videomae_labels.json"
    payload = {
        "schema_version": 1,
        "raw_dirs": [str(raw_dir)],
        "records": [],
        "unmatched_edits": [],
        "duplicate_raw_stems": [],
    }
    candidates_path.write_text(json.dumps(payload), encoding="utf-8")
    trainer_labels_path.write_text("[]", encoding="utf-8")

    monkeypatch.setattr(label_review, "LABEL_CANDIDATES_PATH", candidates_path)
    monkeypatch.setattr(label_review, "TRAINER_LABELS_PATH", trainer_labels_path)
    monkeypatch.setattr(label_review.config, "RAW_CLIP_SOURCE_DIR", raw_dir)

    result = label_review.get_label_review_payload(
        {
            str(inbox_file): "sent_to_inbox",
            str(posted_file): "posted_to_tiktok",
        }
    )

    inventory = result["raw_file_inventory"]
    statuses = {item["filename"]: item["status"] for item in inventory["files"]}
    assert statuses["sent.mp4"] == "sent_to_inbox"
    assert statuses["posted.mp4"] == "posted_to_tiktok"
    assert statuses["new.mp4"] == "new_holdout_candidate"
    assert inventory["summary"]["sent_to_inbox"] == 1
    assert inventory["summary"]["posted_to_tiktok"] == 1
    assert inventory["summary"]["new_holdout_candidates"] == 1


def test_public_payload_marks_missing_review_files(tmp_path, monkeypatch) -> None:
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    existing_file = raw_dir / "existing.mp4"
    existing_file.write_bytes(b"raw")
    candidates_path = tmp_path / "fight_label_candidates.json"
    trainer_labels_path = tmp_path / "videomae_labels.json"
    existing = _pending_record("existing.mp4")
    existing["raw_path"] = str(existing_file)
    existing["edit_path"] = ""
    existing["edit_filename"] = ""
    missing = _pending_record("missing.mp4")
    missing["raw_path"] = str(raw_dir / "missing.mp4")
    missing["edit_path"] = str(raw_dir / "missing-edit.mp4")
    payload = {
        "schema_version": 1,
        "raw_dirs": [str(raw_dir)],
        "records": [existing, missing],
        "unmatched_edits": [],
        "duplicate_raw_stems": [],
    }
    candidates_path.write_text(json.dumps(payload), encoding="utf-8")
    trainer_labels_path.write_text("[]", encoding="utf-8")

    monkeypatch.setattr(label_review, "LABEL_CANDIDATES_PATH", candidates_path)
    monkeypatch.setattr(label_review, "TRAINER_LABELS_PATH", trainer_labels_path)

    result = label_review.get_label_review_payload()

    assert result["records"][0]["raw_exists"] is True
    assert result["records"][0]["edit_exists"] is False
    assert result["records"][1]["raw_exists"] is False
    assert result["records"][1]["edit_exists"] is False


def test_regenerate_trainer_labels_excludes_pending_review_records(tmp_path, monkeypatch) -> None:
    candidates_path = tmp_path / "fight_label_candidates.json"
    trainer_labels_path = tmp_path / "videomae_labels.json"
    payload = {
        "schema_version": 1,
        "records": [_record("approved.mp4"), _pending_record("pending.mp4")],
        "unmatched_edits": [],
        "duplicate_raw_stems": [],
    }
    candidates_path.write_text(json.dumps(payload), encoding="utf-8")
    trainer_labels_path.write_text("[]", encoding="utf-8")

    monkeypatch.setattr(label_review, "LABEL_CANDIDATES_PATH", candidates_path)
    monkeypatch.setattr(label_review, "TRAINER_LABELS_PATH", trainer_labels_path)

    result = label_review.regenerate_trainer_labels()

    saved = json.loads(trainer_labels_path.read_text(encoding="utf-8"))
    assert result["count"] == 1
    assert [item["filename"] for item in saved] == ["approved.mp4"]


def test_add_raw_file_to_review_queue_creates_pending_videomae_record(tmp_path, monkeypatch) -> None:
    raw_file = tmp_path / "fresh.mp4"
    raw_file.write_bytes(b"raw")
    candidates_path = tmp_path / "fight_label_candidates.json"
    trainer_labels_path = tmp_path / "videomae_labels.json"
    payload = {
        "schema_version": 1,
        "raw_dirs": [str(tmp_path)],
        "records": [],
        "unmatched_edits": [],
        "duplicate_raw_stems": [],
    }
    candidates_path.write_text(json.dumps(payload), encoding="utf-8")
    trainer_labels_path.write_text("[]", encoding="utf-8")

    monkeypatch.setattr(label_review, "LABEL_CANDIDATES_PATH", candidates_path)
    monkeypatch.setattr(label_review, "TRAINER_LABELS_PATH", trainer_labels_path)
    monkeypatch.setattr(
        label_review,
        "_record_from_videomae_detection",
        lambda path, *_args: _pending_record(path.name)
        | {
            "raw_path": str(path),
            "edit_path": "",
            "edit_filename": "",
            "match": {"method": "videomae_current_checkpoint", "score": 0.8, "confidence": "high"},
        },
    )

    result = label_review.add_raw_file_to_review_queue(str(raw_file))

    saved_candidates = json.loads(candidates_path.read_text(encoding="utf-8"))
    saved_trainer_labels = json.loads(trainer_labels_path.read_text(encoding="utf-8"))
    assert result["record_index"] == 0
    assert result["created"] is True
    assert saved_candidates["records"][0]["filename"] == "fresh.mp4"
    assert saved_candidates["records"][0]["needs_review"] is True
    assert saved_candidates["records"][0]["match"]["method"] == "videomae_current_checkpoint"
    assert saved_trainer_labels == []


def test_add_raw_file_to_review_queue_rejects_existing_record(tmp_path, monkeypatch) -> None:
    raw_file = tmp_path / "fresh.mp4"
    raw_file.write_bytes(b"raw")
    candidates_path = tmp_path / "fight_label_candidates.json"
    trainer_labels_path = tmp_path / "videomae_labels.json"
    payload = {
        "schema_version": 1,
        "raw_dirs": [str(tmp_path)],
        "records": [],
        "unmatched_edits": [],
        "duplicate_raw_stems": [],
    }
    candidates_path.write_text(json.dumps(payload), encoding="utf-8")
    trainer_labels_path.write_text("[]", encoding="utf-8")

    monkeypatch.setattr(label_review, "LABEL_CANDIDATES_PATH", candidates_path)
    monkeypatch.setattr(label_review, "TRAINER_LABELS_PATH", trainer_labels_path)
    monkeypatch.setattr(
        label_review,
        "_record_from_videomae_detection",
        lambda path, *_args: _pending_record(path.name)
        | {
            "raw_path": str(path),
            "edit_path": "",
            "edit_filename": "",
            "match": {"method": "videomae_current_checkpoint", "score": 0.8, "confidence": "high"},
        },
    )

    first = label_review.add_raw_file_to_review_queue(str(raw_file))

    saved_candidates = json.loads(candidates_path.read_text(encoding="utf-8"))
    assert first["created"] is True
    assert len(saved_candidates["records"]) == 1
    try:
        label_review.add_raw_file_to_review_queue(str(raw_file))
    except label_review.HTTPException as exc:
        assert exc.status_code == 409
        assert "already in the review or training set" in exc.detail
    else:
        raise AssertionError("Expected duplicate raw file to be rejected")


def test_add_raw_file_to_review_queue_rejects_tiktok_used_file(tmp_path, monkeypatch) -> None:
    raw_file = tmp_path / "posted.mp4"
    raw_file.write_bytes(b"raw")
    candidates_path = tmp_path / "fight_label_candidates.json"
    trainer_labels_path = tmp_path / "videomae_labels.json"
    payload = {
        "schema_version": 1,
        "raw_dirs": [str(tmp_path)],
        "records": [],
        "unmatched_edits": [],
        "duplicate_raw_stems": [],
    }
    candidates_path.write_text(json.dumps(payload), encoding="utf-8")
    trainer_labels_path.write_text("[]", encoding="utf-8")

    monkeypatch.setattr(label_review, "LABEL_CANDIDATES_PATH", candidates_path)
    monkeypatch.setattr(label_review, "TRAINER_LABELS_PATH", trainer_labels_path)

    try:
        label_review.add_raw_file_to_review_queue(str(raw_file), raw_file_usage={str(raw_file): "posted_to_tiktok"})
    except label_review.HTTPException as exc:
        assert exc.status_code == 409
        assert "already posted to TikTok" in exc.detail
    else:
        raise AssertionError("Expected TikTok-used raw file to be rejected")
