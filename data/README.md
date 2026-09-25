# Data Directory

This directory has two very different kinds of data:

- `minimap_icons/`: committed runtime assets used by the app.
- `training/`: local/generated trainer-labeler data, ignored by default except for its README.

Keep this distinction during the split. The official app needs champion icon assets, but it does not need private training labels or raw clip state.

## `minimap_icons/`

Contains:

- `champions_manifest.json`
- `champions_list.txt`
- `images/*.png`

Runtime usage:

- `backend/config.py` points `MINIMAP_ICONS_DIR` and `MANIFEST_PATH` here.
- `backend/minimap_detector.py` uses these assets for champion/HUD matching support.

Maintenance notes:

- Keep these assets current when Riot/Data Dragon adds or renames champions.
- If a champion is missing, runtime participant summaries can become noisier.
- The minimap YOLO checkpoint (`checkpoints/minimap_yolov8s_best.pt`, local only, not tracked by Git) is the only minimap champion detector. These assets are used for HUD portrait matching of the player's champion.

## `training/`

Contains generated local trainer/labeler state. See `data/training/README.md`.

This folder is intentionally ignored because it can contain private file paths and labels derived from personal Medal clips.

