# LoL Clip Pipeline

Local pipeline for turning League of Legends source clips into vertical short-form videos with fight detection, adaptive cropping, captions, and optional publishing.

## What It Does

- Accepts an existing `.mp4` clip path through the dashboard or API.
- Extracts full-frame and minimap frames.
- Detects likely fight timing with a trained VideoMAE checkpoint, falling back to heuristics if needed.
- Detects player/enemy context from minimap icons, HUD portraits, health bars, and optional vision-model classification.
- Computes a smooth 3:4 vertical crop focused on the fight.
- Encodes a 1080x1440 MP4 with FFmpeg.
- Generates TikTok and Instagram captions from detected fight context.
- Stores job state, progress, flags, output paths, and captions in SQLite.

## Current Limitations

- Caption generation depends on upstream detection quality. The caption model does not inspect video frames directly.
- The local caption model file is not included. Without `models/llama-3-8b-instruct.Q4_K_M.gguf`, the app uses fallback captions.
- Champion recognition is still the main weak spot. The current minimap classifier uses champion icons, synthetic augmentation, pHash/template matching, and an optional classifier cache.
- The minimap GAN is not used directly by captions. It only helps if you explicitly rebuild the minimap classifier cache with GAN samples.
- The native `frame_decoder.dll` is not present by default. The app works through the OpenCV fallback, but native decoding would be faster.
- Upload endpoints exist, but social auth/token setup is currently environment-variable based.

## Active Improvement Direction

The best next improvement is a supervised minimap champion classifier using:

- `data/minimap_icons/images`
- `data/minimap_icons/champions_manifest.json`
- low-confidence real minimap crops collected from actual clips

The GAN can remain an augmentation experiment, but labeled synthetic and real examples should become the main source of champion identity training.

## Requirements

- Python 3.12
- Node.js and npm
- FFmpeg and ffprobe on `PATH`
- Optional: CUDA-enabled PyTorch for faster VideoMAE/GAN training
- Optional: `OPENAI_API_KEY` for vision-based participant classification

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

## Optional Model Files

These files are intentionally not committed:

- `checkpoints/videomae_lol_best.pt`
- `checkpoints/minimap_mask_gan.pt`
- `models/llama-3-8b-instruct.Q4_K_M.gguf`

If the VideoMAE checkpoint is missing, fight detection falls back to heuristics. If the Llama model is missing, captions use the fallback generator.

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

- `backend/`: FastAPI app, clip pipeline, detection, cropping, encoding, captions, upload helpers, training coordinator.
- `frontend/`: React/Vite dashboard.
- `data/minimap_icons/`: champion icon source data used by minimap detection.
- `tools/`: minimap classifier/GAN data tools.
- `frame_decoder/`: optional native decoder source.
- `checkpoints/`: local model checkpoints, ignored by Git.
- `models/`: local caption model files, ignored by Git.

