# Checkpoints

This folder contains model weights and training-run metrics. `.pt` and `.pth` files are configured for Git LFS in `.gitattributes`.

## Runtime Checkpoints

| File | Purpose |
| --- | --- |
| `minimap_yolov8s_best.pt` | YOLOv8 minimap champion detector |
| `videomae_lol_best.pt` | Legacy VideoMAE fight-window classifier used by the optional `--task fight` trainer path |
| `videomae_lol_highlight_editor*.pt` | VideoMAE highlight editor checkpoints for final clip span and phase prediction |

The runtime app default in `backend/config.py` currently points at:

```text
checkpoints/videomae_lol_highlight_editor_10ep_3layers.pt
```

This is the promoted default highlight editor for the current app branch.

## Experimental Run Folders

Folders such as these are training outputs:

- `slice_0/`
- `videomae_20260723_champion_fights/`

They usually contain `metrics.json` and sometimes checkpoint outputs. Treat them as trainer/labeler artifacts unless a checkpoint is explicitly promoted to the official app.

## Promotion Rule

A checkpoint should move from trainer/labeler to official app only when:

- It has been evaluated on clips that were not used for training.
- It improves or preserves trim quality on known hard clips.
- Its filename and expected task are documented.
- The app can load it from a clean clone with `git lfs pull`.
- The default path in `backend/config.py` agrees with the committed file.

## Split Policy

Official app repo:

- Keep `minimap_yolov8s_best.pt`.
- Keep `videomae_lol_highlight_editor_10ep_3layers.pt` as the promoted highlight editor checkpoint unless a newer checkpoint replaces it.
- Keep legacy `videomae_lol_best.pt` only if the app or tests still need it.

Trainer/labeler repo or branch:

- Keep experiment folders.
- Keep alternative highlight checkpoints.
- Keep metrics and training logs.
