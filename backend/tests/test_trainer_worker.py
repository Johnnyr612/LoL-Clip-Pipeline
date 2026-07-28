from __future__ import annotations

from pathlib import Path

from backend import trainer_worker


def test_dataset_uses_raw_path_and_multiple_fight_segments(tmp_path) -> None:
    raw_clip = tmp_path / "split-root.mp4"
    raw_clip.write_bytes(b"")
    fallback_dir = tmp_path / "fallback"
    fallback_dir.mkdir()

    dataset = trainer_worker.LoLFightDataset(
        fallback_dir,
        [
            {
                "filename": "missing-from-fallback.mp4",
                "raw_path": str(raw_clip),
                "fight_segments": [[5.0, 14.0], [30.0, 42.0]],
                "duration": 60.0,
            }
        ],
    )

    assert dataset.missing_labels == []
    assert len(dataset) > 0
    assert {sample[2] for sample in dataset.sample_index} == {0, 1}
    assert {sample[0] for sample in dataset.sample_index} == {raw_clip}


def test_dataset_uses_posted_clip_bounds_for_strong_negatives(tmp_path) -> None:
    raw_clip = tmp_path / "posted-selection.mp4"
    raw_clip.write_bytes(b"")

    dataset = trainer_worker.LoLFightDataset(
        None,
        [
            {
                "filename": raw_clip.name,
                "raw_path": str(raw_clip),
                "clip_start": 24.516,
                "clip_end": 59.983,
                "fight_segments": [[30.0, 47.0]],
                "duration": 59.983,
            }
        ],
    )

    assert dataset.missing_labels == []
    assert len(dataset) > 0
    assert {sample[2] for sample in dataset.sample_index} == {0, 1}
    negative_starts = [start for _path, start, label in dataset.sample_index if label == 0]
    positive_starts = [start for _path, start, label in dataset.sample_index if label == 1]
    assert negative_starts
    assert positive_starts
    assert all(start + trainer_worker.WINDOW_SECONDS <= 24.516 for start in negative_starts)
    assert all(
        trainer_worker._window_overlap_pct(start, [(30.0, 47.0)]) >= trainer_worker.POSITIVE_OVERLAP_THRESHOLD
        for start in positive_starts
    )


def test_highlight_targets_encode_editor_phases() -> None:
    include, phase = trainer_worker._highlight_targets_for_label(
        {
            "clip_start": 10.0,
            "clip_end": 40.0,
            "fight_segments": [[16.0, 32.0]],
            "duration": 60.0,
        }
    )

    assert include[9].item() == 0
    assert include[10].item() == 1
    assert include[39].item() == 1
    assert include[40].item() == 0
    assert phase[9].item() == trainer_worker.PHASE_EXCLUDE
    assert phase[10].item() == trainer_worker.PHASE_BUILDUP
    assert phase[16].item() == trainer_worker.PHASE_FIGHT
    assert phase[32].item() == trainer_worker.PHASE_PAYOFF
    assert phase[40].item() == trainer_worker.PHASE_EXCLUDE


def test_highlight_dataset_uses_clip_bounds_as_training_samples(tmp_path) -> None:
    raw_clip = tmp_path / "style-clip.mp4"
    raw_clip.write_bytes(b"")

    dataset = trainer_worker.LoLHighlightDataset(
        None,
        [
            {
                "filename": raw_clip.name,
                "raw_path": str(raw_clip),
                "clip_start": 12.0,
                "clip_end": 44.0,
                "fight_segments": [[20.0, 36.0]],
                "duration": 60.0,
            }
        ],
    )

    assert dataset.missing_labels == []
    assert len(dataset) == 1
    assert dataset.sample_index[0][0] == raw_clip


def test_split_labels_holds_out_whole_source_groups() -> None:
    labels = [
        {"filename": f"clip-{index}.mp4", "raw_path": str(Path("D:/clips") / f"source-{index // 2}.mp4")}
        for index in range(10)
    ]

    train_labels, val_labels = trainer_worker._split_labels(labels, val_fraction=0.3)

    train_groups = {trainer_worker._label_group_key(label) for label in train_labels}
    val_groups = {trainer_worker._label_group_key(label) for label in val_labels}
    assert train_groups
    assert val_groups
    assert train_groups.isdisjoint(val_groups)
    assert len(train_labels) + len(val_labels) == len(labels)


def test_progress_line_includes_indicator() -> None:
    line = trainer_worker._progress_line("train", 3, 10, epoch=1, epochs=2, detail="loss=0.1234")

    assert "train epoch 1/2" in line
    assert "3/10" in line
    assert "%" in line
    assert "loss=0.1234" in line
    assert "[" in line and "]" in line
