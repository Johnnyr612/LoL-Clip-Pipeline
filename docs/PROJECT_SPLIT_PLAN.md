# Project Split Plan

This repository currently combines two products that share models and video-domain logic:

- The official clipper app: local dashboard, FastAPI API, processing queue, fight trim inference, crop planning, encoding, output browsing, and TikTok posting.
- The trainer/labeler: raw clip inventory, label review, VideoMAE label generation, retraining scripts, checkpoints, and review-only dataset state.

The next repository, `https://github.com/Johnnyr612/LoL_Auto_Clipper_V2`, should make that boundary explicit before feature work continues. The safest split is to keep the current repository as the historical monolith until V2 has a clean app core, then move trainer/labeler work into either a sibling repo or a long-lived branch that does not ship with the official app.

## Current Ownership Map

| Path | Current responsibility | Split target |
| --- | --- | --- |
| `backend/main.py` | API routes for processing, TikTok, settings, training, and label review | Split. App keeps jobs/settings/TikTok/process; trainer keeps `/train` and `/training/*` routes |
| `backend/pipeline.py` | Official processing pipeline: validate, decode, minimap, highlight trim, crop, encode, persist job | Official app |
| `backend/fight_detector.py` | Runtime VideoMAE highlight inference and legacy fight helpers | Shared core, but app should keep inference-only surface |
| `backend/cropper.py` | Vertical crop planning | Official app |
| `backend/encoder.py` and `backend/ffmpeg_tools.py` | FFmpeg output generation | Official app |
| `backend/minimap_detector.py`, `backend/team_tracker.py`, `backend/vision_classifier.py` | Champion/team/participant signals | Shared core used by app, evaluated by trainer |
| `backend/models.py` | SQLite schema for jobs, training runs, TikTok tokens | Split. App DB should not need training tables unless trainer stays embedded |
| `backend/label_review.py` | Review inventory and label JSON lifecycle | Trainer/labeler |
| `backend/trainer.py`, `backend/trainer_worker.py` | VideoMAE training coordinator and worker | Trainer/labeler |
| `frontend/src/JobDashboard.tsx`, `OutputPanel.tsx`, `SettingsPanel.tsx` | Official app dashboard | Official app |
| `frontend/src/LabelReview.tsx` | Trainer/labeler UI | Trainer/labeler |
| `tools/*.py` | Dataset audits, candidate generation, pixel matching | Trainer/labeler |
| `data/minimap_icons/` | Champion icon assets used by runtime detection | Official app shared asset |
| `data/training/` | Private generated review candidates and training labels | Trainer/labeler local data |
| `checkpoints/*.pt` | Runtime and experimental weights | Split by release quality |
| `TestClip.mp4` | Reproducible smoke-test input | Official app, optional in V2 through Git LFS |

## Recommended V2 Shape

Use one of these shapes depending on how independently you want the trainer to evolve.

### Option A: Two Repositories

- `LoL_Auto_Clipper_V2`: official app only.
- `LoL_Auto_Clipper_Trainer`: trainer, labeler, dataset scripts, experiments, and non-release checkpoints.

This is the cleanest product boundary. The app repo can stay small, less private, and easier to package. The trainer repo can contain rough workflow tools without increasing app risk.

### Option B: One Repository, Two Packages

- `apps/clipper/`: official FastAPI and React app.
- `apps/trainer/`: label review and training UI/API.
- `packages/lol_clip_core/`: shared video validation, frame decode, VideoMAE inference wrappers, minimap/team utilities, data models.

This works if you want one repo for now but still want import boundaries. It is more convenient during active development, but it requires discipline so trainer routes and data files do not leak into the app build.

## Official App Isolation Boundary

The official app should include only what is needed to turn an existing MP4 into a finished vertical clip and optionally publish it.

Keep:

- `/health`
- `/settings/folders`
- `/settings/open-folder`
- `/jobs`
- `/jobs/{job_id}`
- `/process`
- `/output-files`
- `/outputs/{relative_path}`
- `/checkpoints/highlight`
- `/tiktok/*`
- Runtime pipeline modules: `pipeline`, `frame_io`, `media_probe`, `fight_detector` inference, `cropper`, `encoder`, `ffmpeg_tools`, `minimap_detector`, `team_tracker`, `vision_classifier`, `tiktok`, `logging_config`, `config`
- Runtime assets: stable release checkpoints, minimap icon manifest/images, sample smoke-test clip if desired

Remove from official app:

- `/train`
- `/train/stream`
- `/training/label-review`
- `/training/video`
- `backend/trainer.py`
- `backend/trainer_worker.py`
- `backend/label_review.py`
- `frontend/src/LabelReview.tsx`
- `tools/*` except runtime support tools
- `data/training/*`
- experimental checkpoint directories such as `checkpoints/slice_0/`

The app can still expose a "model version" and checkpoint picker, but it should not expose training controls in the official distribution.

## Trainer/Labeler Boundary

The trainer/labeler should own the private and evolving dataset workflow:

- Raw Medal clip intake from folders such as `D:\Medal\Clips\League of Legends`.
- Review queue state in `data/training/fight_label_candidates.json`.
- Approved trainer labels in `data/training/videomae_labels.json`.
- Skip decisions and review notes.
- Edited-reference matching through `tools/build_fight_training_labels.py`.
- New-file inventory and "used for training" checks.
- Training runs and metrics under `checkpoints/<run_name>/slice_0/metrics.json`.
- Promotion decisions for checkpoints that are good enough to become app defaults.

## Shared-Core Candidates

If you split into packages, these modules are good candidates for a shared importable core:

- `backend/config.py`, after separating app-only settings from training-only settings.
- `backend/media_probe.py`
- `backend/frame_io.py`
- The inference parts of `backend/fight_detector.py`
- Champion/team structures from `backend/minimap_detector.py`
- `backend/team_tracker.py`

Avoid sharing the whole current `backend` package unchanged. It carries app routes, trainer workers, TikTok persistence, and runtime side effects together.

## Data Privacy And Large Files

Do not move private raw clips or generated label JSON into the official app repo unless you intentionally want them in Git history.

Recommended policy:

- Keep source Medal clips outside Git, for example `D:\Medal\Clips\League of Legends`.
- Keep generated training data ignored by default.
- Commit only README files that explain expected schemas and local setup.
- Use Git LFS only for release-quality checkpoints and sample assets.
- Keep experimental checkpoints in ignored local folders or in the trainer repo.

## Split-Time Cleanup Checklist

- Decide whether V2 uses two repositories or one repo with `apps/` and `packages/`.
- Pick one canonical default highlight checkpoint filename and make code, README, and files agree.
- Move label review routes out of the official API or protect them behind a trainer-only build flag.
- Move `LabelReview.tsx` out of the app's default dashboard for official builds.
- Separate SQLite schema into app tables and trainer tables.
- Define a checkpoint promotion process from trainer output to app release checkpoint.
- Add a Medal clip inventory sync command or scheduled UI refresh in the trainer/labeler.
- Write migration notes before force-moving files so old branch history stays understandable.

