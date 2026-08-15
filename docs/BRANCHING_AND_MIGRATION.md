# Branching And Migration

The current branch is `master` and its `origin` remote points at `https://github.com/Johnnyr612/LoL-Clip-Pipeline.git`. The new target repository has already been created at `https://github.com/Johnnyr612/LoL_Auto_Clipper_V2`.

This document describes branch names and migration order. It does not assume the remote has been rewired yet.

## Recommended Branch Names

Use clear branch names that describe the split boundary:

- `codex/docs-split-readiness`: documentation-only branch for the files added here.
- `codex/app-isolation`: remove trainer/labeler routes and UI from the official app surface.
- `codex/trainer-labeler-isolation`: preserve and improve the training workflow after the official app boundary is clean.
- `codex/v2-repo-bootstrap`: first branch in `LoL_Auto_Clipper_V2` after copying or importing the app subset.
- `codex/checkpoint-promotion`: checkpoint naming, LFS rules, and release-weight cleanup.

## Migration Order

1. Freeze the current repo as the reference monolith.
2. Commit this documentation so the split intent is captured before moving files.
3. Create `codex/app-isolation` from the current branch.
4. Remove trainer/labeler API routes from the official app branch.
5. Remove `Label Review` from the official frontend navigation or gate it behind a dev-only flag.
6. Keep runtime inference and output processing tests green.
7. Bootstrap `LoL_Auto_Clipper_V2` from the isolated official app branch.
8. Create a trainer/labeler branch or repo from the original monolith and keep dataset tooling there.
9. Promote only validated checkpoints from trainer/labeler into the V2 app repo.

## Remote Setup Options

If this working tree will push to the new V2 repo, add a second remote first:

```powershell
git remote add v2 https://github.com/Johnnyr612/LoL_Auto_Clipper_V2.git
git fetch v2
```

Then push only an intentional branch:

```powershell
git push v2 codex/app-isolation:codex/app-isolation
```

Avoid changing `origin` until you are sure this checkout should stop tracking `LoL-Clip-Pipeline`.

## What Should Land In V2

V2 should begin with:

- App README and setup docs.
- Backend runtime routes and modules.
- Frontend job/output/settings views.
- Runtime icon data in `data/minimap_icons/`.
- Release-quality checkpoint files through Git LFS.
- `.env.example`, `requirements.txt`, `pyproject.toml`, frontend package files.
- Smoke tests and `TestClip.mp4` if you want a reproducible local test.

V2 should not begin with:

- Private raw Medal clips.
- Generated `data/training/*.json`.
- Training logs.
- `precomputed/` or `precomputed_v2/`.
- Experimental checkpoint run folders.
- Trainer worker code unless the official app intentionally includes trainer controls.

## Checkpoint Naming Decision

The code default currently points at the promoted 10-epoch, 3-layer highlight editor:

```text
checkpoints/videomae_lol_highlight_editor_10ep_3layers.pt
```

This checkout also contains alternate highlight editor files such as:

```text
checkpoints/videomae_lol_highlight_editor_epoch3.pt
checkpoints/videomae_lol_highlight_editor_10ep_5layers.pt
```

For V2, keep `backend/config.py`, the README, and the committed checkpoint file aligned. If a future checkpoint is promoted, update all three together.

## Split Validation

Before pushing a branch as V2-ready:

```powershell
.\.venv\Scripts\python.exe -m pytest
cd frontend
npm run build
```

Then run the app manually:

```powershell
.\.venv\Scripts\python.exe -m uvicorn backend.main:app --reload --host 127.0.0.1 --port 8000
cd frontend
npm run dev
```

Use `TestClip.mp4` or a known Medal clip to verify that the dashboard can process a local path end to end.
