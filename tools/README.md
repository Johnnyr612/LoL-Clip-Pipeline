# Tools

This folder contains trainer/labeler support scripts. These are useful for dataset creation and audits, but they should not ship as official app runtime features unless a specific script is promoted.

## Scripts

| Script | Purpose | Split target |
| --- | --- | --- |
| `build_fight_training_labels.py` | Match raw Medal clips to edited trims, create `fight_label_candidates.json`, and create initial trainer labels | Trainer/labeler |
| `pixel_match_label_candidates.py` | Pixel/frame matching helper for label candidates | Trainer/labeler |
| `audit_fight_detection.py` | Evaluate fight detection behavior on known clips/labels | Trainer/labeler |

## Medal Raw Clip Workflow

Use `build_fight_training_labels.py` when you have a folder of manually edited MP4 references and want to seed review records from them:

```powershell
.\.venv\Scripts\python.exe tools\build_fight_training_labels.py `
  --raw-dir "D:\Medal\Clips\League of Legends" `
  --edits-dir "D:\Path\To\Edited\References" `
  --output "data\training\fight_label_candidates.json" `
  --trainer-labels-output "data\training\videomae_labels.json"
```

For new raw Medal clips without edited references, prefer the dashboard's `Label Review` flow:

1. Add `D:\Medal\Clips\League of Legends` to `raw_dirs` in `data/training/fight_label_candidates.json`.
2. Open the dashboard's `Label Review` tab.
3. Click `Refresh Raw Files`.
4. Add `new_holdout_candidate` files to review.
5. Approve reviewed examples when they are ready for future training.

## Split Policy

Move this folder to the trainer/labeler side during official app isolation. The official app should not require dataset-building scripts to process clips.

