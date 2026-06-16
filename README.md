# LoL Clip Pipeline

Local pipeline for turning League of Legends source clips into vertical short-form videos with fight detection and adaptive cropping.

## What It Does

- Accepts an existing `.mp4` clip path through the dashboard or `/process` API. The backend also has a `/jobs` upload endpoint, but the current dashboard uses local paths.
- Extracts full-frame and minimap frames with OpenCV.
- Detects likely fight timing with a fine-tuned VideoMAE checkpoint, falling back to a heuristic score when the checkpoint is missing or inference fails.
- Detects player/enemy context from YOLO minimap champion detections, HUD portraits, health bars, and optional full-frame YOLO classification.
- Computes a smooth 3:4 vertical crop focused on the fight.
- Encodes a 1080x1440 MP4 with FFmpeg.
- Can send completed clips to TikTok through the Content Posting API after TikTok OAuth connection.
- Stores job state, progress, flags, output paths, and detection debug data in SQLite.

## Current Limitations

- Champion recognition now uses the YOLOv8 minimap champion detector weights only for minimap champion detection.
- TikTok direct posting requires TikTok app review and the `video.publish` scope. Upload-to-inbox with `video.upload` is the recommended first review path.

## Local YOLO Participant Classification

Participant classification can use an optional local full-frame YOLO model. Point the app at your weights before starting the backend:

```powershell
$env:LOL_CLIP_YOLO_WEIGHTS = "D:\path\to\your\weights.pt"
```

Optional tuning:

```powershell
$env:LOL_CLIP_YOLO_CONFIDENCE = "0.35"
$env:LOL_CLIP_YOLO_DEVICE = "0"
```

The full-frame YOLO classifier is optional. If weights are not configured, the app falls back to minimap YOLO detections, HUD portrait matching, and health-bar detection.

## Minimap YOLO Champion Detection

Minimap champion detection loads `checkpoints/minimap_yolov8s_best.pt` by default. This checkpoint was copied from the newest detector run in `LoL Minimap Champion Detector/runs/run_003_yolov8s_export_20260610_104747/weights/best.pt`.

Override the checkpoint or inference settings before starting the backend:

```powershell
$env:LOL_CLIP_MINIMAP_YOLO_WEIGHTS = "D:\path\to\best.pt"
$env:LOL_CLIP_MINIMAP_YOLO_CONFIDENCE = "0.35"
$env:LOL_CLIP_MINIMAP_YOLO_DEVICE = "0"
```

If the minimap YOLO model cannot load or produces no detections for a frame, that frame contributes no minimap champion detections. The app no longer falls back to the older Hough-circle plus icon/template detector for minimap champion detection.

## TikTok Posting

TikTok integration uses Login Kit plus the Content Posting API:

- `video.upload`: sends the completed MP4 to the creator's TikTok inbox so they can finish editing/posting in TikTok.
- `video.publish`: initializes Direct Post for the completed MP4. TikTok requires app approval for this scope, and unaudited clients may be limited to private visibility.

Set TikTok credentials before starting the backend. The easiest local setup is to create a `.env` file in the project root:

```env
TIKTOK_CLIENT_KEY=...
TIKTOK_CLIENT_SECRET=...
TIKTOK_REDIRECT_URI=https://your-domain.example/tiktok/callback
TIKTOK_AUTH_SUCCESS_URL=https://your-domain.example
```

`.env` and `.env.*` are ignored by Git. Values set directly in PowerShell still override `.env` for that session.

For local development, the defaults are:

```text
TIKTOK_REDIRECT_URI=http://127.0.0.1:8000/tiktok/callback
TIKTOK_AUTH_SUCCESS_URL=http://127.0.0.1:5173
```

TikTok's production web Login Kit requires registered `https` redirect URIs. Use the dashboard's TikTok section on a completed job to connect an account, upload to inbox, or start Direct Post.

## Fight Detection And VideoMAE

The current fight detector loads `checkpoints/videomae_lol_best.pt` when present. The model is based on `MCG-NJU/videomae-base` with a small binary classifier head that predicts fight versus non-fight for 16-second windows.

Fine-tuning is handled by `backend/trainer_worker.py`:

- Labels provide `fight_start` and `fight_end` for each training clip.
- The dataset samples 16-second windows from each clip.
- A window is positive when at least half of its seconds overlap the labeled fight.
- A window is negative when it has minimal overlap with the labeled fight.
- Positive and negative windows are balanced before training.
- Frames are resized to 224x224, normalized with ImageNet stats, and passed through VideoMAE.
- The classifier is trained with AdamW, cosine warmup scheduling, gradient accumulation, validation loss tracking, and early stopping.

The first training pass overfit because the dataset was too small and too easy: many negative windows came from the same source clips and did not represent enough real non-fight gameplay. That produced a checkpoint that could memorize the training distribution better than it generalized to new clips.

## Fight Training Status

The current branch includes the VideoMAE trainer in `backend/trainer.py` and `backend/trainer_worker.py`. It trains from a directory of `.mp4` clips plus a labels JSON file containing `filename`, `fight_start`, `fight_end`, and optional duration fields. It can also reuse precomputed frame arrays from `precomputed/` when matching `.npy` files exist.

Example direct training command:

```powershell
.\.venv\Scripts\python.exe backend\trainer.py --clips_dir "D:\path\to\training_clips" --labels "D:\path\to\labels.json" --epochs 25 --batch_size 4 --output_dir checkpoints
```

The backend also exposes `POST /train` and `GET /train/stream` for starting a run and streaming metrics. The Vite dev server proxies those routes, but the current frontend does not expose training controls.

Future fight-boundary retraining should still focus on better negative samples:

- Add varied non-fight windows from full gameplay clips.
- Save precomputed frame arrays under `precomputed/` or a branch-specific precomputed directory.
- Keep train/validation clips separated by source video where possible.

This branch does not include a `backend/prepare_negatives.py` helper. If that workflow is revived, add the helper before documenting commands for it.

## Champion Detection Notes

The minimap detector now uses the supervised YOLOv8 checkpoint as the champion detector. The icon assets are still used by HUD portrait matching:

- `data/minimap_icons/images`
- `data/minimap_icons/champions_manifest.json`

Useful follow-up work:

- Keep Riot/Data Dragon assets current so new champions are not missing.
- Use a match champion whitelist when available, ideally the 10 champions from Riot's local Live Client Data API during recording.
- Keep evaluating YOLO detections against real failed clips and retrain on hard examples.
- Use temporal voting across frames instead of trusting a single crop.

## Requirements

- Python 3.12
- Node.js and npm
- FFmpeg and ffprobe on `PATH`
- Git LFS for large checkpoint and sample video files.
- PyTorch for VideoMAE inference/training and Ultralytics YOLO. It is intentionally not pinned in `requirements.txt`; install a CPU or CUDA build appropriate for your machine.
- Optional: CUDA-enabled PyTorch for faster VideoMAE training.
- Optional: `LOL_CLIP_YOLO_WEIGHTS` for local YOLO participant classification.
- Optional: TikTok developer credentials for Upload/Direct Post.

## Included Large Files

This project includes trained weights and a sample clip through Git LFS:

- `checkpoints/videomae_lol_best.pt`: fine-tuned VideoMAE fight detector.
- `checkpoints/minimap_yolov8s_best.pt`: YOLOv8 minimap champion detector.
- `TestClip.mp4`: sample input clip for testing the pipeline.

`TestClip.mp4` was not part of the training set. It is included only as a reproducible test clip so a new user can run the pipeline end to end after setup.

After cloning, make sure Git LFS has downloaded the real files:

```powershell
git lfs install
git lfs pull
```

## Setup

From the project root:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Install frontend dependencies:

```powershell
cd frontend
npm install
cd ..
```

## Run The App

Start the backend:

```powershell
.\.venv\Scripts\python.exe -m uvicorn backend.main:app --reload --host 127.0.0.1 --port 8000
```

Start the frontend in a second terminal:

```powershell
cd frontend
npm run dev
```

Open:

```text
http://127.0.0.1:5173
```

Paste a full local `.mp4` path into the dashboard and start a job.

To test with the included sample clip, paste the full path to:

```text
TestClip.mp4
```

For example, from this project directory:

```text
C:\path\to\CS668 LoL Auto Clip Trimmer\TestClip.mp4
```

Runtime files are written outside the repo:

- Database, uploads, temp files, and logs: `%APPDATA%\LoLClipApp`
- Encoded clips and minimap detection debug images: `%USERPROFILE%\Videos\LoLClipApp`

## Local Checkpoint Files

The trained checkpoint files are intentionally tracked with Git LFS so users can run the pipeline without retraining:

- `checkpoints/videomae_lol_best.pt`
- `checkpoints/minimap_yolov8s_best.pt`

If the VideoMAE checkpoint is missing, fight detection falls back to heuristics.

## Secrets

Do not commit real API keys. Local environment files are ignored by Git:

- `.env`
- `.env.*`

Use `.env.example` as a template for local model and TikTok settings.

## Useful Commands

Run backend tests:

```powershell
.\.venv\Scripts\python.exe -m pytest
```

## Project Layout

- `backend/`: FastAPI app, clip pipeline, detection, cropping, encoding, TikTok posting, and training coordinator.
- `frontend/`: React/Vite dashboard.
- `data/minimap_icons/`: champion icon source data used by HUD portrait matching.
- `checkpoints/`: model checkpoints tracked through Git LFS.
