# LoL Clip Pipeline

Local pipeline for turning League of Legends source clips into vertical short-form videos with fight detection, adaptive cropping, and generated descriptions.

## What It Does

- Accepts an existing `.mp4` clip path through the dashboard or API.
- Extracts full-frame and minimap frames with OpenCV.
- Detects likely fight timing with a fine-tuned VideoMAE checkpoint, falling back to a heuristic score when the checkpoint is missing or inference fails.
- Detects player/enemy context from YOLO minimap champion detections, HUD portraits, health bars, and optional full-frame YOLO classification.
- Computes a smooth 3:4 vertical crop focused on the fight.
- Encodes a 1080x1440 MP4 with FFmpeg.
- Generates a social-ready description from detected fight context.
- Stores job state, progress, flags, output paths, and descriptions in SQLite.

## Current Limitations

- Champion recognition now uses the YOLOv8 minimap champion detector weights only for minimap champion detection.
- Description quality depends on upstream detection quality. The OpenAI description request receives fight metadata and dialog text; it does not inspect video frames directly.
- The current minimap GAN is an augmentation experiment. It can generate minimap-style feature samples, but it is not accurately detecting the correct champions yet.
- Social publishing is future work; the current app focuses on local clip generation and description drafting.

## Description Generation

Descriptions are generated with the OpenAI API using `OPENAI_API_KEY`. Set it in your environment before starting the backend:

```powershell
$env:OPENAI_API_KEY = "sk-..."
```

By default the app uses `gpt-4o-mini` for descriptions. You can override that without code changes:

```powershell
$env:LOL_CLIP_CAPTION_MODEL = "gpt-4o-mini"
```

When `OPENAI_API_KEY` is missing, the OpenAI request fails, or the model returns invalid JSON, the app uses a deterministic fallback description from `backend/caption_gen.py`. The fallback builds a payload from the detected player champion, enemy champions, fight type, and minimap context. It returns:

- `caption`: a short hook plus body text.
- `hashtags`: fixed gaming and League hashtags.
- `hook_line`: the first-line hook, for example a duel hook when one enemy is known.

Fallback flags are stored as `caption_api_key_missing` or `caption_fallback`.

## Local YOLO Participant Classification

Participant classification can use a local YOLO model instead of the OpenAI vision API. Point the app at your weights before starting the backend:

```powershell
$env:LOL_CLIP_YOLO_WEIGHTS = "D:\path\to\your\weights.pt"
```

Optional tuning:

```powershell
$env:LOL_CLIP_YOLO_CONFIDENCE = "0.35"
$env:LOL_CLIP_YOLO_DEVICE = "0"
```

The YOLO classifier is optional. If weights are not configured, the app falls back to minimap, HUD, and health-bar detection.

## Minimap YOLO Champion Detection

Minimap champion detection loads `checkpoints/minimap_yolov8s_best.pt` by default. This checkpoint was copied from the newest detector run in `LoL Minimap Champion Detector/runs/run_003_yolov8s_export_20260610_104747/weights/best.pt`.

Override the checkpoint or inference settings before starting the backend:

```powershell
$env:LOL_CLIP_MINIMAP_YOLO_WEIGHTS = "D:\path\to\best.pt"
$env:LOL_CLIP_MINIMAP_YOLO_CONFIDENCE = "0.35"
$env:LOL_CLIP_MINIMAP_YOLO_DEVICE = "0"
```

If the minimap YOLO model cannot load or produces no detections for a frame, that frame contributes no minimap champion detections. The app no longer falls back to the older Hough-circle plus icon/template detector for minimap champion detection.

## Future Social Integration

TikTok and Instagram publishing are intentionally not wired into the current app. They should be revisited after clip quality, champion detection, and description generation are stable.

Future work should include:

- TikTok OAuth account connection.
- TikTok draft upload or direct post using the generated description.
- Instagram/Reels publishing once a public video URL flow is available.
- Safe token storage, refresh handling, and clear publishing status in the dashboard.

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

## Training V2 Plan

Future fight-boundary retraining should use the `training-v2` branch. That branch adds a better negative-sample workflow:

1. Build additional negatives from full clips with `backend/prepare_negatives.py`.
2. Save precomputed frame arrays under `precomputed_v2/`.
3. Write expanded labels to `data/trainer_labels_v2.json`.
4. Train from the `training-v2` branch using those labels and `precomputed_v2`.

Example v2 preparation command:

```powershell
.\.venv\Scripts\python.exe backend\prepare_negatives.py --clips-dir "D:\Medal\Clips\League of Legends" --labels data\trainer_labels_all.json --output-labels data\trainer_labels_v2.json --precomputed-dir precomputed_v2 --max-negatives-per-clip 3
```

The goal for v2 is to reduce overfitting by giving VideoMAE more varied non-fight windows and a cleaner train/validation split before replacing `checkpoints/videomae_lol_best.pt`.

## Champion Detection Notes

The minimap detector now uses the supervised YOLOv8 checkpoint as the champion detector. The older icon/template assets are still used by HUD portrait matching and related tooling:

- `data/minimap_icons/images`
- `data/minimap_icons/champions_manifest.json`
- synthetic minimap-style augmentation
- optional low-confidence real minimap crops from actual clips

Useful follow-up work:

- Keep Riot/Data Dragon assets current so new champions are not missing.
- Use a match champion whitelist when available, ideally the 10 champions from Riot's local Live Client Data API during recording.
- Keep evaluating YOLO detections against real failed clips and retrain on hard examples.
- Use temporal voting across frames instead of trusting a single crop.

The existing minimap GAN can stay as an experiment for augmentation, but the practical path is labeled synthetic data plus real low-confidence crops.

## Requirements

- Python 3.12
- Node.js and npm
- FFmpeg and ffprobe on `PATH`
- Git LFS for large checkpoint and sample video files.
- Optional: CUDA-enabled PyTorch for faster VideoMAE/GAN training
- `OPENAI_API_KEY` for generated descriptions.
- Optional: `LOL_CLIP_CAPTION_MODEL` to override the caption model.
- Optional: `LOL_CLIP_YOLO_WEIGHTS` for local YOLO participant classification.

## Included Large Files

This project includes trained weights and a sample clip through Git LFS:

- `checkpoints/videomae_lol_best.pt`: fine-tuned VideoMAE fight detector.
- `checkpoints/minimap_yolov8s_best.pt`: YOLOv8 minimap champion detector.
- `checkpoints/minimap_mask_gan.pt`: current minimap mask GAN experiment.
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
C:\path\to\New project\TestClip.mp4
```

## Local Checkpoint Files

The trained checkpoint files are intentionally committed with Git LFS so users can run the pipeline without retraining:

- `checkpoints/videomae_lol_best.pt`
- `checkpoints/minimap_yolov8s_best.pt`
- `checkpoints/minimap_mask_gan.pt`

If the VideoMAE checkpoint is missing, fight detection falls back to heuristics. If the OpenAI API key is missing, descriptions use the fallback generator.

## Secrets

Do not commit real API keys. Local environment files are ignored by Git:

- `.env`
- `.env.*`

Use `.env.example` as a template or set `OPENAI_API_KEY` directly in your shell before starting the backend.

## Useful Commands

Run backend tests:

```powershell
.\.venv\Scripts\python.exe -m pytest
```

Build the minimap classifier cache:

```powershell
.\.venv\Scripts\python.exe tools\build_minimap_classifier_cache.py
```

Build the cache with GAN samples:

```powershell
.\.venv\Scripts\python.exe tools\build_minimap_classifier_cache.py --gan-checkpoint checkpoints\minimap_mask_gan.pt --gan-samples-per-icon 12
```

Collect real minimap GAN crops:

```powershell
.\.venv\Scripts\python.exe tools\collect_minimap_gan_crops.py --clips "D:\Medal\Clips\League of Legends"
```

Train the minimap mask GAN:

```powershell
.\.venv\Scripts\python.exe tools\train_minimap_mask_gan.py
```

## Project Layout

- `backend/`: FastAPI app, clip pipeline, detection, cropping, encoding, descriptions, and training coordinator.
- `frontend/`: React/Vite dashboard.
- `data/minimap_icons/`: champion icon source data used by minimap detection.
- `tools/`: minimap classifier and GAN data tools.
- `checkpoints/`: local model checkpoints, ignored by Git.
