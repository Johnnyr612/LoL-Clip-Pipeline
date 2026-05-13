# LoL Clip Pipeline

Local pipeline for turning League of Legends source clips into vertical short-form videos with fight detection, adaptive cropping, descriptions, and optional publishing.

## What It Does

- Accepts an existing `.mp4` clip path through the dashboard or API.
- Extracts full-frame and minimap frames with OpenCV.
- Detects likely fight timing with a fine-tuned VideoMAE checkpoint, falling back to a heuristic score when the checkpoint is missing or inference fails.
- Detects player/enemy context from minimap icons, HUD portraits, health bars, and optional vision-model classification.
- Computes a smooth 3:4 vertical crop focused on the fight.
- Encodes a 1080x1440 MP4 with FFmpeg.
- Generates TikTok and Instagram descriptions from detected fight context.
- Stores job state, progress, flags, output paths, and descriptions in SQLite.

## Current Limitations

- Champion recognition is still the main weak spot. The current minimap classifier uses champion icons, synthetic augmentation, pHash/template matching, and an optional classifier cache.
- Description quality depends on upstream detection quality. The OpenAI description request receives fight metadata and dialog text; it does not inspect video frames directly.
- The current minimap GAN is an augmentation experiment. It can generate minimap-style feature samples, but it is not accurately detecting the correct champions yet.
- TikTok publishing requires a configured TikTok developer app and the Content Posting API scopes.

## Description Generation

Descriptions are generated with the OpenAI API using `OPENAI_API_KEY`. Set it in your environment before starting the backend:

```powershell
$env:OPENAI_API_KEY = "sk-..."
```

By default the app uses `gpt-4o-mini` for descriptions. You can override that without code changes:

```powershell
$env:LOL_CLIP_CAPTION_MODEL = "gpt-4o-mini"
```

The same `OPENAI_API_KEY` also enables the optional OpenAI vision fallback for participant classification. The vision model can be overridden separately:

```powershell
$env:LOL_CLIP_VISION_MODEL = "gpt-4o-mini"
```

When `OPENAI_API_KEY` is missing, the OpenAI request fails, or the model returns invalid JSON, the app uses deterministic fallback descriptions from `backend/caption_gen.py`. The fallback builds TikTok and Instagram payloads from the detected player champion, enemy champions, fight type, and minimap context. It returns:

- `caption`: a short hook plus body text.
- `hashtags`: fixed gaming and League hashtags.
- `hook_line`: the first-line hook, for example a duel hook when one enemy is known.

Fallback flags are stored as `caption_api_key_missing` or `caption_fallback`.

## TikTok Publishing

The dashboard can connect a TikTok account through TikTok OAuth when these environment variables are set before the backend starts:

```powershell
$env:TIKTOK_CLIENT_KEY = "..."
$env:TIKTOK_CLIENT_SECRET = "..."
$env:TIKTOK_REDIRECT_URI = "http://127.0.0.1:8000/auth/tiktok/callback"
```

The requested default scopes are `user.info.basic,video.upload,video.publish`. You can override them with `TIKTOK_SCOPES`.

TikTok actions in the Description panel:

- `Post`: uses TikTok Direct Post with the generated description.
- `Save draft`: uploads the video to the TikTok inbox so it can be finished in the TikTok app. TikTok's inbox upload flow may not prefill the description, so the backend returns it in the response for copying if needed.

Direct posting uses `TIKTOK_PRIVACY_LEVEL`, defaulting to `SELF_ONLY`. TikTok may restrict unaudited apps to private posting.

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

## Champion Detection Future Work

The next major improvement should be a supervised minimap champion detector/classifier. The current approach works from:

- `data/minimap_icons/images`
- `data/minimap_icons/champions_manifest.json`
- synthetic minimap-style augmentation
- optional low-confidence real minimap crops from actual clips

Planned direction:

- Keep Riot/Data Dragon assets current so new champions are not missing.
- Use a match champion whitelist when available, ideally the 10 champions from Riot's local Live Client Data API during recording.
- Separate icon localization from champion identity: first find minimap champion bubbles, then classify cropped icons.
- Use temporal voting across frames instead of trusting a single crop.
- Collect real hard examples from failed clips and retrain on them.
- Take inspiration from Maknee's `LeagueMinimapDetectionCNN`, especially its synthetic minimap generation and Faster R-CNN-style detector, while retraining on the current champion set instead of relying on old patch weights.

The existing minimap GAN can stay as an experiment for augmentation, but the practical path is labeled synthetic data plus real low-confidence crops.

## Requirements

- Python 3.12
- Node.js and npm
- FFmpeg and ffprobe on `PATH`
- Optional: CUDA-enabled PyTorch for faster VideoMAE/GAN training
- `OPENAI_API_KEY` for generated descriptions.
- Optional: `LOL_CLIP_CAPTION_MODEL` to override the caption model.
- Optional: `LOL_CLIP_VISION_MODEL` for vision-based participant classification.

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

## Local Checkpoint Files

These files are intentionally not committed:

- `checkpoints/videomae_lol_best.pt`
- `checkpoints/minimap_mask_gan.pt`

If the VideoMAE checkpoint is missing, fight detection falls back to heuristics. If the OpenAI API key is missing, descriptions use the fallback generator.

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

- `backend/`: FastAPI app, clip pipeline, detection, cropping, encoding, descriptions, upload helpers, and training coordinator.
- `frontend/`: React/Vite dashboard.
- `data/minimap_icons/`: champion icon source data used by minimap detection.
- `tools/`: minimap classifier and GAN data tools.
- `checkpoints/`: local model checkpoints, ignored by Git.
