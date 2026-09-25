# LoL Clip Pipeline

Local pipeline for turning League of Legends source clips into vertical short-form videos with fight detection and dynamic cropping.

## What It Does

- Accepts an existing `.mp4` clip path through the dashboard or `/process` API.
- Decodes the source once with OpenCV: 1920x1080 analysis frames at 2 fps, minimap crops at 4 fps (skipped when minimap detection is off), and, for V-JEPA checkpoints, dense 384x384 window frames.
- Detects post-worthy trim timing with the selected highlight editor checkpoint: the fine-tuned VideoMAE editor (default) or the V-JEPA 2.1 editor. Jobs fail if the selected checkpoint is missing or cannot produce a trim.
- Detects player/enemy context from YOLO minimap champion detections, temporal team-color tracking, HUD portraits, health bars, and optional full-frame YOLO classification.
- Computes a 3:4 vertical crop path that steers toward the fight using full-frame health bars (green = player, red = enemies, blue = allies).
- Encodes a 1080x1440 MP4 with FFmpeg.
- Records per-stage timings (`timings_seconds`) and V-JEPA inference diagnostics (`vjepa_inference`) in each job's debug payload.
- Can send completed clips to TikTok through the Content Posting API after TikTok OAuth connection.
- Stores job state, queue/progress, flags, output paths, trim/crop settings, and detection debug data in SQLite.

## Current Limitations

- Champion recognition uses the YOLOv8 minimap champion detector weights only for minimap champion detection. Those weights are no longer tracked in the repo; without them, minimap detection produces no detections.
- V-JEPA inference supports clips up to 120 seconds and produces one continuous highlight span (no multi-fight montage).
- Dynamic crop steering uses full-frame health-bar detections only. Minimap detections still help identify champions and teams, but the minimap has no input on the view cropper.
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

To focus jobs on the raw clip timing/crop path while leaving minimap champion/team detection for later, enable the dashboard's `Skip minimap detection` option or set:

```powershell
$env:LOL_CLIP_SKIP_MINIMAP_DETECTION = "1"
```

Skipped minimap jobs also skip minimap frame extraction. They still use the selected highlight editor, main-frame HUD matching, health-bar crop signals, and optional local full-frame YOLO classification.

## Team Color Tracking

Champion identity and team color are handled as separate signals. YOLO is used to identify champion icons, while `backend/team_tracker.py` classifies team color from the thin outer ring around each minimap portrait.

The tracker is intentionally conservative:

- Samples only the icon border annulus so champion portrait art does not pollute the team-color decision.
- Skips border sampling for overlapping detections because clustered champion icons can contaminate each other's rings.
- Votes across all sampled minimap frames because a champion's team does not change mid-game.
- Finalizes a team only after enough clean votes meet the confidence threshold. Otherwise the champion is marked as needing review instead of guessing.

The pipeline stores the tracker summary in each job's `detection_debug.team_tracker` payload. This includes the finalized team when available, confidence, vote counts, per-team tallies, and a `needs_review` flag.

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

## Highlight Editing And VideoMAE

The dashboard's highlight-weights picker lists every `checkpoints/videomae_lol_highlight*.pt` and `checkpoints/vjepa21_highlight*.pt` file (served by `GET /checkpoints/highlight`). The selected file is sent with each `/process` request and is also used when a processed clip is added to Label Review. If nothing is selected, the backend falls back to `config.VIDEOMAE_HIGHLIGHT_CHECKPOINT`.

The VideoMAE path expects `checkpoints/videomae_lol_highlight_editor_10ep_3layers.pt` by default. That model predicts the post-worthy trim directly from the raw one-minute clip: an include/exclude timeline plus phase labels (`exclude`, `buildup`, `fight`, `payoff`). If the highlight editor is missing, cannot load, or does not select an include span, the job fails so the model issue is visible.

Fine-tuning is handled by `backend/trainer_worker.py`:

- Highlight labels use `clip_start` and `clip_end` as the final post-worthy span.
- Phase labels are derived from `clip_start -> fight_start -> fight_end -> clip_end`.
- The default `highlight` task samples 16 frames across the raw 60-second context and predicts per-second include/exclude plus phase classes.
- The legacy `fight` task still samples 16-second windows, marks positives by fight overlap, and balances positive/negative windows.
- Frames are resized to 224x224, normalized with ImageNet stats, and passed through VideoMAE.
- The heads are trained with AdamW, cosine warmup scheduling, gradient accumulation, validation loss tracking, and early stopping.

The first training pass overfit because the dataset was too small and too easy: many negative windows came from the same source clips and did not represent enough real non-fight gameplay. That produced a checkpoint that could memorize the training distribution better than it generalized to new clips.

## V-JEPA 2.1 Highlight Editor

Checkpoints whose filename starts with `vjepa21_highlight` (currently `checkpoints/vjepa21_highlight_best_10ep.pt`) are routed to `backend/vjepa_detector.py` instead of VideoMAE:

- `vjepa_detector.prepare_model()` loads and caches the checkpoint once (keyed by path, size, and mtime) and validates the saved architecture, 384px preprocessing, and normalization.
- `frame_io.decode_video()` decodes the dense V-JEPA window frames in the same pass as the crop-analysis frames, resizing the original frame (not the 1080p copy) exactly as in training.
- `vjepa_runtime.py` holds the inference-only model: a V-JEPA 2.1 ViT-B encoder over 2-second, 16-frame windows at a 1-second step, followed by a temporal head. `decode_highlight()` turns the per-second logits into one highlight span; an auxiliary combat head sets `fight_start`/`fight_end`.
- Windows are batched on CUDA (`LOL_CLIP_VJEPA_BATCH_SIZE`, default `2`, range 1-16) with fp16 autocast. Out-of-memory errors halve the batch; CPU uses one window at a time.

Local requirements (ignored by Git): the checkpoint file, the official source tree at `external/vjepa2-main/` (override with `LOL_CLIP_VJEPA_SOURCE_DIR`), and `pip install -r requirements-vjepa.txt`. Use the `Model Only` trim preset to see the raw model timing without the heuristic trim adjustments. See `docs/vjepa-local-preview.md` for details and timing inspection commands.

## Highlight Training Status

The current branch includes the VideoMAE trainer in `backend/trainer.py` and `backend/trainer_worker.py`. The default `--task highlight` trains from a labels JSON file containing `filename`, `raw_path`, `clip_start`, `clip_end`, `fight_start`, `fight_end`, `fight_segments`, and duration fields. `raw_path` is preferred so one training run can use source clips from multiple folders; `--clips_dir` remains a fallback for older labels that only contain filenames. It can also reuse precomputed frame arrays from `precomputed/` when matching `.npy` files exist.

By default, fine-tuning freezes most of VideoMAE and trains the highlight heads plus the final two encoder layers. Use `--task fight` for the legacy binary fight-window trainer. Use `--no-freeze_backbone` for a full-backbone run, or adjust `--unfreeze_last_n_layers`, `--classifier_lr`, and `--backbone_lr` for a narrower or wider fine-tune. The trainer prints per-batch progress bars and writes live metrics to `slice_0/metrics.json` under the output directory.

Example direct training command:

```powershell
.\.venv\Scripts\python.exe backend\trainer.py `
  --labels "D:\Codex Projects\CS668 LoL Auto Clip Trimmer\data\training\videomae_labels.json" `
  --task highlight `
  --epochs 25 `
  --batch_size 4 `
  --output_dir "D:\Codex Projects\CS668 LoL Auto Clip Trimmer\checkpoints\videomae_20260716" `
  --freeze_backbone `
  --unfreeze_last_n_layers 2 `
  --progress_interval 5
```

The backend also exposes `POST /train` and `GET /train/stream` for starting a run and streaming metrics. The Vite dev server proxies those routes, but the current frontend does not expose training controls.

Future highlight-editor retraining should focus on reviewed examples that represent your posting style:

- Keep `clip_start` and `clip_end` aligned to the actual posted edit.
- Review noisy `fight_start`/`fight_end` values because they define buildup/fight/payoff phases.
- Save precomputed frame arrays under `precomputed/` or a branch-specific precomputed directory.
- Keep train/validation clips separated by source video where possible.

This branch does not include a `backend/prepare_negatives.py` helper. If that workflow is revived, add the helper before documenting commands for it.

## Champion Detection Notes

The minimap detector now uses the supervised YOLOv8 checkpoint as the champion detector. Team color is refined by temporal border-ring voting in `backend/team_tracker.py`, and the finalized team can override noisy per-frame team readings before participant aggregation. The icon assets are still used by HUD portrait matching:

- `data/minimap_icons/images`
- `data/minimap_icons/champions_manifest.json`

Useful follow-up work:

- Keep Riot/Data Dragon assets current so new champions are not missing.
- Use a match champion whitelist when available, ideally the 10 champions from Riot's local Live Client Data API during recording.
- Keep evaluating YOLO detections against real failed clips and retrain on hard examples.
- Continue evaluating team-color tracker summaries from `detection_debug.team_tracker` on hard clips, especially crowded fights where many detections are skipped.

## Requirements

- Python 3.12
- Node.js and npm
- FFmpeg and ffprobe on `PATH`
- Git LFS for large checkpoint and sample video files.
- PyTorch for VideoMAE/V-JEPA inference, VideoMAE training, and Ultralytics YOLO. It is intentionally not pinned in `requirements.txt`; install a CPU or CUDA build appropriate for your machine.
- Optional: CUDA-enabled PyTorch for faster VideoMAE training and V-JEPA inference.
- Optional: `requirements-vjepa.txt` (`timm`, `einops`, `tqdm`) plus `external/vjepa2-main/` for V-JEPA checkpoints.
- Optional: `LOL_CLIP_YOLO_WEIGHTS` for local YOLO participant classification.
- Optional: TikTok developer credentials for Upload/Direct Post.

## Included Large Files

The repo includes one large file through Git LFS:

- `TestClip.mp4`: sample input clip for testing the pipeline.

Model checkpoints are **not** tracked in the repo anymore (the `checkpoints/` directory was removed from Git). Place them in `checkpoints/` locally; see `Local Checkpoint Files` below.

`TestClip.mp4` was not part of the training set. It is included only as a reproducible test clip so a new user can run the pipeline end to end after setup.

After cloning, make sure Git LFS has downloaded the real files:

```powershell
git lfs install
git lfs pull
```

## Setup

From the project root:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

If `py -3.12` is not available, install Python 3.12 first. The command `python -3.12` is not valid when `python.exe` points directly at one interpreter; use the Windows launcher `py -3.12`.

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
- Encoded clips and minimap detection debug images: `D:\LoLClipOutputVids` by default

Override the output folder with:

```env
LOL_CLIP_OUTPUT_DIR=D:\LoLClipOutputVids
```

## Dashboard State

The dashboard has three main tabs:

- `Jobs`: start local-path jobs, pick a highlight checkpoint (VideoMAE or V-JEPA 2.1), choose trim presets, choose crop view settings, monitor the active queue, browse previous outputs, preview the final vertical clip, inspect media/crop debug, and post completed clips to TikTok.
- `Label Review`: review raw files and model-generated trim boundaries for future VideoMAE training data.
- `Settings`: view/open local folders and save source clip folders.

The backend intentionally runs one processing job at a time with a semaphore. If you start multiple clips back to back, the newest jobs enter `queued` status and the `Active Queue` panel shows which job is running and which jobs are waiting.

## Crop Composition

The vertical crop stays at 3:4 (`810x1080` in the normalized 1920x1080 space, then scaled to `1080x1440`). The dashboard sends crop settings per job (`mode`, `transition`).

- `static` uses one fixed centered crop (`LOL_CLIP_STATIC_CROP_X`).
- `dynamic` (default) is tuned for locked-camera clips. It opens with a rule-of-thirds composition (player on a third, look-room toward `LOL_CLIP_DYNAMIC_DEFAULT_THIRDS_SIDE`, default `right`), uses an enemy found near the model's `fight_start` to frame the opening of the fight, reframes toward a persistent visible enemy side, centers the player/enemy cluster when they are close together, widens to include a nearby ally when everyone fits, avoids the minimap corner, and falls back to the default composition when no enemy remains.

Dynamic steering only uses full-frame health bars (`fight_detector.estimate_combat_screen_x_position_tracks`):

- The most central, thick green bar anchors the player. If no player bar is found in a frame, that frame contributes no player, enemy, or ally position.
- Thick red bars with a champion level badge next to them define enemy position; objective bars (with text over them) and thin minion/ward bars are filtered out. When several enemies are visible, the one pushing farthest past the centered crop's edge is chosen.
- Thick blue bars define an ally position.
- Enemy/ally samples need support from nearby samples before they count, and the cropper also discards single-frame jumps and takes a 0.75s median around each 1-second keyframe.
- Minimap champion detections and minimap player positions do not move the crop.

The output panel's `Crop plan` section records the chosen mode, transition, movement range, keyframe count, position changes, sample crop positions, and the number of health-bar samples that steered the crop. If movement is `0px`, Jump and Smooth will look identical for that output because the crop path resolved to a fixed view.

See `backend/README.md` (`View Shifter`) for the file-by-file flow.

Useful `.env` controls:

- `LOL_CLIP_CROP_MODE=dynamic` follows persistent visible enemy direction in locked-camera clips.
- `LOL_CLIP_CROP_MODE=static` uses one fixed crop.
- `LOL_CLIP_CROP_TRANSITION=cut` jumps between chosen crop positions. Set `pan` for a speed-limited slide (240px/s).
- `LOL_CLIP_PLAYER_COMPOSITION=thirds` enables look-room framing. Set `center` to keep the player centered.
- `LOL_CLIP_DYNAMIC_DEFAULT_THIRDS_SIDE=right` sets the look-room side used before an enemy side is known (`left`, `right`, or `center`).
- `LOL_CLIP_THIRDS_LOOK_ROOM_PX=20` moves the player 20px farther from the fight-side third, giving the crop more room toward visible enemies.
- `LOL_CLIP_DYNAMIC_THREAT_SIDE_TRIGGER_PX=60` controls how far left/right an enemy must be from the player before it counts as a crop direction.
- `LOL_CLIP_DYNAMIC_CLUSTER_CENTER_TRIGGER_PX=320` centers the player/enemy pair instead of using thirds when they are this close.
- `LOL_CLIP_DYNAMIC_THREAT_HOLD_SEC=1.0` controls how long that direction must persist before the crop reframes.
- `LOL_CLIP_DYNAMIC_MAX_VIEW_CHANGES=3` caps non-center enemy reframes per clip (a reframe that is needed to keep the enemy visible is still allowed).
- `LOL_CLIP_DYNAMIC_OPENING_LOOKAHEAD_SEC=3.0` controls how far into the clip the opening framing looks for the first enemy.
- `LOL_CLIP_DYNAMIC_OPENING_FOCUS_HOLD_SEC=2.0` keeps the fight-start crop framed around the player/enemy pair for the first seconds of combat before normal dynamic framing resumes.
- `LOL_CLIP_DYNAMIC_PAIR_MIN_PADDING_PX=24` lets the crop use smaller edge padding when a red health bar is approaching the 3:4 view but strict padding would keep the camera centered.
- `LOL_CLIP_CAMERA_THREAT_SUPPORT_TOLERANCE_PX=170` controls how loosely nearby red health-bar samples are grouped as the same threat.
- `LOL_CLIP_CAMERA_THREAT_BAR_MIN_WIDTH` / `_MIN_HEIGHT` / `_MIN_AREA` (defaults 25 / 8 / 300) set the red-bar size filter for crop steering.

## Output Encoding Quality

The pipeline crops the source into a vertical frame and re-encodes it, so the output bitrate controls how much of the original Medal quality is preserved. A simple Medal trim can often copy the original stream without re-encoding, but any crop or scale step needs a new encode. The default encoder uses high-quality H.264 settings:

- `LOL_CLIP_VIDEO_ENCODER=libx264`
- `LOL_CLIP_MATCH_SOURCE_ENCODING=1`
- `LOL_CLIP_FFMPEG_CRF=18`
- `LOL_CLIP_FFMPEG_PRESET=slow`
- `LOL_CLIP_AUDIO_BITRATE=320k`

When source matching is enabled, the backend probes the input MP4 with ffprobe and uses its detected frame rate for the output encode when available. Because the vertical crop and scale filters modify every video frame, the video stream cannot be copied bit-for-bit like a simple trim; it must be re-encoded. To reduce generation loss, source matching gives the output bitrate headroom above the detected source bitrate:

```text
LOL_CLIP_SOURCE_BITRATE_MULTIPLIER=1.15
```

For example, a `24.9M` source encodes around `28.6M` by default. If the input audio is already AAC, the backend copies the audio stream instead of re-encoding it. The completed job stores the input/output media labels in the debug payload shown by the dashboard.

For larger files that stay closer to Medal's QHD 60 FPS H.264 settings, set fallback output settings in `.env`. If FFmpeg lists `h264_nvenc`, the NVIDIA encoder path is closest to Medal's GPU/H.264 recording setup:

```text
LOL_CLIP_MATCH_SOURCE_ENCODING=1
LOL_CLIP_VIDEO_ENCODER=h264_nvenc
LOL_CLIP_NVENC_PRESET=p5
LOL_CLIP_NVENC_RC=vbr
LOL_CLIP_SOURCE_BITRATE_MULTIPLIER=1.15
LOL_CLIP_VIDEO_BITRATE=25M
LOL_CLIP_VIDEO_MAXRATE=25M
LOL_CLIP_VIDEO_BUFSIZE=50M
```

Leave `LOL_CLIP_VIDEO_BITRATE` empty to use CRF mode instead. Lower CRF values increase quality and file size; `18` is visually high quality, while `16` is a heavier near-source option.

## Local Checkpoint Files

The code expects these files in `checkpoints/` (not tracked by Git; `.gitattributes` still routes any committed `checkpoints/*.pt` through LFS):

- `checkpoints/vjepa21_highlight_best_10ep.pt`: V-JEPA 2.1 highlight editor.
- `checkpoints/videomae_lol_highlight_editor_10ep_3layers.pt`: default VideoMAE highlight editor (`config.VIDEOMAE_HIGHLIGHT_CHECKPOINT`).
- `checkpoints/videomae_lol_best.pt`: legacy VideoMAE fight detector for the `--task fight` trainer path.
- `checkpoints/minimap_yolov8s_best.pt`: YOLOv8 minimap champion detector (or set `LOL_CLIP_MINIMAP_YOLO_WEIGHTS`).

The dashboard picker only lists highlight checkpoints that actually exist, so with only the V-JEPA file present, V-JEPA is what gets selected. If the selected highlight editor checkpoint is missing or fails, processing stops with an error. This is intentional: the trimming decision should come from the trained highlight editor, not a hardcoded padding or heuristic path.

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
- `backend/team_tracker.py`: conservative minimap team-color tracker based on border-ring sampling, cluster skipping, and temporal voting.
- `backend/vjepa_detector.py`, `backend/vjepa_runtime.py`: V-JEPA 2.1 highlight inference.
- `docs/`: split plan, branching notes, and V-JEPA local preview notes.
- `tools/`: trainer/labeler dataset and audit scripts.
- `external/vjepa2-main/`: local, Git-ignored copy of the official V-JEPA 2 source (needed for V-JEPA checkpoints).
- `frontend/`: React/Vite dashboard.
- `data/minimap_icons/`: champion icon source data used by HUD portrait matching.
- `checkpoints/`: local model checkpoints (not tracked by Git).

## Split-Readiness Docs

Additional architecture and migration notes are available in:

- `docs/PROJECT_SPLIT_PLAN.md`: official app versus trainer/labeler ownership map.
- `docs/BRANCHING_AND_MIGRATION.md`: branch names, V2 migration order, and checkpoint promotion notes.
- `docs/vjepa-local-preview.md`: V-JEPA setup, batch-size tuning, and timing inspection.
- `backend/README.md`: backend route/module ownership and app isolation notes.
- `frontend/README.md`: dashboard view ownership and label-review split notes.
- `data/README.md` and `data/training/README.md`: runtime assets versus private trainer-labeler data.
- `tools/README.md`: training/data-audit script ownership.
