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
