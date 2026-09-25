# Training Data Workflow

This folder is for trainer/labeler state. It is ignored by Git except for this README because the JSON files can contain private local paths such as `D:\Medal\Clips\League of Legends\...`.

## Expected Files

| File | Owner | Purpose |
| --- | --- | --- |
| `fight_label_candidates.json` | Trainer/labeler | Review queue with raw paths, optional edited references, proposed clip/fight boundaries, notes, skip state, and approval state |
| `videomae_labels.json` | Trainer/labeler | Approved training labels consumed by `backend/trainer_worker.py` |

## Medal Clip Intake

Target source folder:

```text
D:\Medal\Clips\League of Legends
```

To make this folder visible to the current label review system, `fight_label_candidates.json` should contain it in `raw_dirs`:

```json
{
  "schema_version": 1,
  "raw_dirs": [
    "D:\\Medal\\Clips\\League of Legends"
  ],
  "edits_dir": "",
  "defaults": {
    "pre_fight_context_sec": 2.0,
    "post_fight_context_sec": 0.0
  },
  "records": [],
  "unmatched_edits": [],
  "duplicate_raw_stems": []
}
```

The trainer/labeler scans only `.mp4` files directly inside each configured raw directory.

## File States

The backend computes raw file state in `backend/label_review.py`.

| State | Meaning | Next action |
| --- | --- | --- |
| `new_holdout_candidate` | MP4 exists in a raw folder and has no review/training record | Add to review, or leave as evaluation/holdout |
| `review_queue` | MP4 has a review candidate and is waiting for approval or adjustment | Open the record and review clip/fight boundaries |
| `used_for_training` | MP4 appears in approved `videomae_labels.json` by path, filename, or stem | Keep as historical training data |
| `skipped` | MP4 has a record that was intentionally skipped | Leave out of training unless revisited |

This is the exact distinction needed for "already used for training", "new files", and "new labeled files ready for future training".

## Review Lifecycle

1. Medal writes a new `.mp4` to `D:\Medal\Clips\League of Legends`.
2. In the trainer/labeler UI, click `Refresh Raw Files`.
3. The new MP4 appears as `new_holdout_candidate`.
4. Click `Add to Review`.
5. The backend runs the selected highlight editor (VideoMAE or V-JEPA 2.1) and creates a candidate record.
6. Review `clip_start`, `clip_end`, and one or more `fight_segments`.
7. Click `Save` to keep it in review, `Skip` to exclude it, or `Approve` to include it in `videomae_labels.json`.
8. Training consumes only approved, non-skipped labels.

## Approved Label Schema

`videomae_labels.json` is a list of records like:

```json
{
  "filename": "example.mp4",
  "raw_path": "D:\\Medal\\Clips\\League of Legends\\example.mp4",
  "edit_path": "",
  "clip_start": 12.5,
  "clip_end": 42.0,
  "fight_start": 15.0,
  "fight_end": 37.0,
  "fight_segments": [[15.0, 37.0]],
  "duration": 60.0,
  "source_edit": "",
  "label_confidence": "reviewed",
  "needs_review": false,
  "skipped": false
}
```

Important fields:

- `raw_path` lets one training run use clips from multiple folders.
- `filename` remains a fallback when `--clips_dir` is used.
- `clip_start` and `clip_end` define the final post-worthy output span.
- `fight_segments` supports multiple fights inside one highlight.
- `needs_review=false` and `skipped=false` are required for trainer-label regeneration.

## Training Command

Example:

```powershell
.\.venv\Scripts\python.exe backend\trainer.py `
  --labels "D:\Codex Projects\CS668 LoL Auto Clip Trimmer\data\training\videomae_labels.json" `
  --task highlight `
  --epochs 25 `
  --batch_size 4 `
  --output_dir "D:\Codex Projects\CS668 LoL Auto Clip Trimmer\checkpoints\videomae_YYYYMMDD" `
  --freeze_backbone `
  --unfreeze_last_n_layers 2 `
  --progress_interval 5
```

The trainer prefers `raw_path` from each label. Use `--clips_dir` only as a fallback for older labels.

## Suggested Future Improvement

Add a dedicated intake command, for example `tools/sync_medal_raw_inventory.py`, that:

- Scans `D:\Medal\Clips\League of Legends`.
- Computes stable file identity with path, size, modified time, and optionally hash.
- Updates `fight_label_candidates.json` without running VideoMAE immediately.
- Preserves current state for files already in review, skipped, or used for training.
- Emits a summary of new, reviewed, skipped, and used files.

The current backend scan is enough for manual workflow. A sync command would make future automation safer and easier to test.

