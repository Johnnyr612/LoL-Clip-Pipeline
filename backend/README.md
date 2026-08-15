# Backend Guide

The backend is a FastAPI service that currently owns both the official clip processing app and the trainer/labeler workflow.

## Runtime App Responsibilities

The official app path starts in `backend/main.py` and `backend/pipeline.py`.

Core runtime flow:

1. `POST /process` receives a local `.mp4` path and per-job trim/crop settings.
2. `pipeline.validate_input()` confirms the file exists, is MP4, has video, and is long enough.
3. `models.create_job()` creates a queued SQLite job record.
4. `ClipPipeline.run()` executes one job at a time behind the process semaphore in `main.py`.
5. `frame_io.decode_video()` extracts full-frame and minimap frame arrays.
6. `minimap_detector.MinimapDetector` and `team_tracker.TeamTracker` identify champion/team context unless minimap detection is skipped.
7. `fight_detector.FightDetector` loads VideoMAE highlight weights and predicts the trim.
8. `cropper.AdaptiveCropper` builds a vertical crop path.
9. `encoder.VideoEncoder` writes the final MP4 to `config.OUTPUT_DIR`.
10. `models.update_job()` stores status, output path, flags, and debug payloads for the frontend.

## Trainer/Labeler Responsibilities

Training routes also live in `backend/main.py` today:

- `POST /train`
- `GET /train/stream`
- `GET /training/label-review`
- `POST /training/label-review/refresh-files`
- `POST /training/label-review/raw-files`
- `POST /training/label-review/records/{record_index}`
- `POST /training/label-review/records/{record_index}/skip`
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
| `fight_detector.py` | VideoMAE highlight inference and trim adjustment | Shared inference core |
| `cropper.py` | Static/dynamic 3:4 crop planning | Official app |
| `encoder.py` | FFmpeg trim/crop/scale/encode path | Official app |
| `ffmpeg_tools.py` | FFmpeg helper discovery and commands | Official app |
| `frame_io.py` | OpenCV frame decode helpers | Shared core |
| `media_probe.py` | ffprobe media metadata | Shared core |
| `minimap_detector.py` | Champion detection/aggregation | Shared core |
| `team_tracker.py` | Temporal team-color voting | Shared core |
| `vision_classifier.py` | Optional full-frame YOLO participant classification | Official app optional |
| `label_review.py` | Raw clip inventory and label JSON mutation | Trainer/labeler |
| `trainer.py` | Training process coordinator/API surface | Trainer/labeler |
| `trainer_worker.py` | VideoMAE fine-tuning worker | Trainer/labeler |
| `tiktok.py` | TikTok OAuth/upload/direct post | Official app |

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

