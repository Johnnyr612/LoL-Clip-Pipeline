# Backend Guide

The backend is a FastAPI service that currently owns both the official clip processing app and the trainer/labeler workflow.

## Runtime App Responsibilities

The official app path starts in `backend/main.py` and `backend/pipeline.py`.

Core runtime flow:

1. `POST /process` receives a local `.mp4` path, per-job trim/crop/processing settings, and a `highlight_checkpoint` from `checkpoints/` (listed by `GET /checkpoints/highlight`, which returns `videomae_lol_highlight*.pt` and `vjepa21_highlight*.pt`).
2. `pipeline.validate_input()` confirms the file exists, is MP4, has video, and is long enough.
3. `models.create_job()` creates a queued SQLite job record.
4. `ClipPipeline.run()` executes one job at a time behind the process semaphore in `main.py`.
5. `stage1_decode`: if the selected checkpoint is V-JEPA (`vjepa21_highlight*`), `vjepa_detector.prepare_model()` loads it first. `frame_io.decode_video()` then decodes the MP4 once, keeping 1920x1080 RGB analysis frames at 2 fps, minimap crops at 4 fps (skipped when minimap detection is off), and dense 384x384 V-JEPA window frames when needed. Audio is not extracted here.
6. `stage2_minimap`: `minimap_detector.MinimapDetector` (YOLO) and `team_tracker.TeamTracker` identify champion/team context on every 4th minimap frame unless minimap detection is skipped.
7. `stage3_fight`: `FightDetector.predict_highlight_trim()` runs the VideoMAE highlight editor or hands off to `vjepa_detector.predict_trim()`, then `apply_highlight_trim_settings()` applies the dashboard trim preset. HUD portrait matching, health-bar enemy counting, and optional `vision_classifier` YOLO build the participant summary.
8. `stage4_crop`: `fight_detector.estimate_combat_screen_x_position_tracks()` reads health bars into player/enemy/ally x tracks; `cropper.AdaptiveCropper` turns them into keyframes and per-frame crop boxes (see `View Shifter` below).
9. `stage5_encode`: `encoder.VideoEncoder` turns the crop path into an FFmpeg `crop` x-expression and writes the final MP4 to `config.OUTPUT_DIR`.
10. `models.update_job()` stores status, output path, flags, and debug payloads (`trim`, `crop_debug`, `timings_seconds`, `vjepa_inference`, detection summary) for the frontend.

## Trainer/Labeler Responsibilities

Training routes also live in `backend/main.py` today:

- `POST /train`
- `GET /train/stream`
- `GET /training/label-review`
- `POST /training/label-review/refresh-files`
- `POST /training/label-review/raw-files`
- `POST /training/label-review/records/{record_index}`
- `POST /training/label-review/records/{record_index}/skip`
- `DELETE /training/label-review/records/{record_index}`
- `POST /training/label-review/records/{record_index}/match-start`
- `POST /training/label-review/records/{record_index}/detect-fight`
- `POST /training/label-review/regenerate`
- `GET /training/video`

These should move to the trainer/labeler side during official app isolation.

## Important Modules

| Module | Purpose | Split target |
| --- | --- | --- |
| `main.py` | FastAPI route registration and app startup | Split by route group |
| `models.py` | SQLite tables and persistence helpers | Split app job tables from trainer tables |
| `config.py` | Environment variables and paths | Split app settings from trainer settings over time |
| `pipeline.py` | Official clip processing orchestration | Official app |
| `fight_detector.py` | VideoMAE highlight inference, V-JEPA dispatch, trim adjustment, and health-bar detection (enemy counts, kill/death end events, crop steering tracks) | Shared inference core |
| `vjepa_detector.py` | Cached V-JEPA checkpoint loading and batched, lock-serialized highlight inference | Shared inference core |
| `vjepa_runtime.py` | V-JEPA 2.1 model definition, window sampling, and highlight decoding (inference only) | Shared inference core |
| `cropper.py` | Static/dynamic 3:4 crop planning | Official app |
| `logging_config.py` | Backend log setup | Shared core |
| `encoder.py` | FFmpeg trim/crop/scale/encode path | Official app |
| `ffmpeg_tools.py` | FFmpeg helper discovery and commands | Official app |
| `frame_io.py` | Single-pass OpenCV decode shared by crop analysis, minimap detection, and V-JEPA | Shared core |
| `media_probe.py` | ffprobe media metadata | Shared core |
| `minimap_detector.py` | Champion detection/aggregation | Shared core |
| `team_tracker.py` | Temporal team-color voting | Shared core |
| `vision_classifier.py` | Optional full-frame YOLO participant classification | Official app optional |
| `label_review.py` | Raw clip inventory and label JSON mutation | Trainer/labeler |
| `trainer.py` | Training process coordinator/API surface | Trainer/labeler |
| `trainer_worker.py` | VideoMAE fine-tuning worker | Trainer/labeler |
| `tiktok.py` | TikTok OAuth/upload/direct post | Official app |

## View Shifter

These are the only files that decide where the 810x1080 crop window sits:

| Step | File / function | What it does |
| --- | --- | --- |
| Frames | `frame_io.decode_video()` | Provides 1920x1080 frames at 2 fps (`full_step = fps / 2`). This is the view shifter's time resolution. |
| Signals | `fight_detector.estimate_combat_screen_x_position_tracks()` | Per frame: HSV masks for red/green/blue bars (`_combat_health_bar_groups`, `_mask_color`, `_health_bar_boxes`), HUD/chat/minimap regions dropped (`_is_non_gameplay_healthbar_region`), player = most central thick green bar (`_select_player_health_bar`), enemies = thick red bars with a level badge and without objective text (`_camera_threat_bars`, `_has_champion_level_badge`, `_exclude_objective_health_bars`), one threat x chosen by `_threat_bar_center_x`, one ally x by `_ally_bar_center_x`. Enemy/ally tracks are filtered by `_stabilize_sparse_threat_positions` (3 supporting samples within ±3 frames). |
| Wiring | `pipeline.ClipPipeline.run()` stage 4 | Passes the tracks, `trim.clip_start/clip_end`, and `trim.fight_start` (as `focus_start`) to the cropper, then builds `crop_debug`. |
| Planning | `cropper.AdaptiveCropper.compute_keyframes()` / `_dynamic_keyframes()` | 1-second keyframes (`_trajectory_times`). Tracks are outlier-filtered (`stabilize_screen_positions`) and median-sampled (`windowed_median`, ±0.75s). A side state machine (candidate side, hold streak, committed side, view-change budget) plus opening logic (`_opening_threat_hint`, `_opening_thirds_side`) chooses a side; `dynamic_rule_of_thirds_crop_x`, `opening_fight_focus_crop_x`, `combat_cluster_crop_x`, `maximize_champion_inclusion_crop_x`, and `avoid_minimap_ui` compute the x. `_hold_small_crop_changes` (cut) or `_limit_pan_speed` (pan) smooths the path. |
| Per-frame | `AdaptiveCropper.interpolate_to_frames()` | Step (cut) or linear (pan) interpolation onto the 2 fps clip timestamps. |
| Render | `encoder._crop_x_expression()` / `quantize_crop_trajectory()` | Builds the FFmpeg `crop=810:1080:x=<expr>` after scaling the source to 1920x1080. |
| Tunables | `config.py` (`CROP_*`, `PLAYER_*`, `THREAT_FRAME_MARGIN_PX`, `DYNAMIC_*`, `COMBAT_CAMERA_THREAT_*`, `KEYFRAME_INTERVAL_SEC`, `PAN_*`, `MAX_PAN_SPEED_PX_PER_SEC`) | All thresholds and `LOL_CLIP_*` overrides. |
| Tests | `tests/test_cropper.py`, `tests/test_fight_detector.py` | Unit coverage for the crop helpers and health-bar tracks. |

`blend_target()`, `include_threat_in_crop()`, and the `BLEND_*`/`LOW_FLOW_THRESHOLD` settings are only referenced by tests, not by the live crop path. `compute_keyframes()` still accepts `player_positions`/`enemies` arguments, but the pipeline passes empty lists.

## Runtime Paths

Most runtime outputs are outside the repository:

- SQLite DB: `%APPDATA%\LoLClipApp\lol_clip_app.sqlite3`
- Logs: `%APPDATA%\LoLClipApp\logs`
- Temp files: `%APPDATA%\LoLClipApp\temp`
- Encoded output clips: `D:\LoLClipOutputVids` by default
- Detection debug images: `D:\LoLClipOutputVids\debug\<job_id>`

The app can process source clips in place. It does not need to copy `D:\Medal\Clips\League of Legends` into the repo.

## Label Review Clip States

`backend/label_review.py` already supports the clip-state distinction needed for Medal intake:

- `used_for_training`: the raw path, filename, or stem appears in `data/training/videomae_labels.json`.
- `review_queue`: the file already has a candidate record and is not skipped.
- `skipped`: the file has a candidate record marked skipped.
- `new_holdout_candidate`: the MP4 is present in a configured raw folder but has no reviewed/training record yet.

This status is returned under `raw_file_inventory` from `GET /training/label-review`.

## App Isolation Notes

When isolating the official app, remove or gate all training routes and imports. The biggest coupling points are:

- `main.py` imports `label_review` and `TrainingCoordinator`.
- `models.py` creates `training_runs` even when training is not used.
- `config.py` mixes app paths, model settings, crop settings, TikTok settings, and trainer settings.
- `fight_detector.py` and `trainer_worker.py` both rely on VideoMAE concepts but should not import each other in the official app.

The official app can keep runtime VideoMAE inference without keeping trainer workers.

