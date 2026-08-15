# Frontend Guide

The frontend is a Vite React dashboard with three current views:

- `Jobs`: process MP4s, monitor queue/progress, select checkpoint/settings, preview outputs, and post to TikTok.
- `Label Review`: inspect raw files, review trim/fight labels, approve/skip records, and regenerate trainer labels.
- `Settings`: show local folders and maintain client-side source folder shortcuts.

## Files By Responsibility

| File | Purpose | Split target |
| --- | --- | --- |
| `src/App.tsx` | Top-level navigation and view selection | Split nav for official app |
| `src/JobDashboard.tsx` | Job creation, queue, trim/crop settings, checkpoint choice | Official app |
| `src/OutputPanel.tsx` | Output preview, media/debug/crop/TikTok details | Official app |
| `src/SettingsPanel.tsx` | Folder visibility and local source-folder shortcuts | Official app |
| `src/LabelReview.tsx` | Trainer/labeler raw inventory and label editing | Trainer/labeler |
| `src/types.ts` | Shared frontend job types | Official app, with trainer types moved later |
| `src/index.css` | Tailwind and dashboard styling | Shared until split |

## Official App Isolation

For the official app, remove the `Label Review` navigation button from `App.tsx` or hide it behind a development/trainer flag. The official app should open directly into the job/output workflow.

Official app views should keep:

- Process local MP4 by full path.
- Queue/progress display.
- Checkpoint picker for release highlight checkpoints.
- Trim and crop settings.
- Output video preview.
- Debug summaries useful to users.
- Settings folders.
- TikTok connection and posting controls.

Trainer-only views should move out:

- Raw video inventory.
- New/eval file list.
- Add-to-review action.
- Clip/fight boundary editor.
- Approve/skip label actions.
- Regenerate training labels.

## Medal Folder UX

The frontend already displays raw inventory returned by the backend label review payload. To support `D:\Medal\Clips\League of Legends` cleanly in the trainer/labeler:

- Show the configured raw directories near the raw inventory summary.
- Keep `Refresh Raw Files` as the explicit scan action.
- Let `new_holdout_candidate` files be added to review.
- Keep `used_for_training` read-only so old training examples are not accidentally re-labeled.
- Keep `skipped` visible but visually quiet.
- Preserve `record_index` links so a file can jump to its review record.

For the official app, source-folder shortcuts can stay local-only in `localStorage`; they are conveniences for selecting/remembering source folders, not training state.

## API Dependencies

`LabelReview.tsx` depends on trainer routes:

- `GET /training/label-review`
- `POST /training/label-review/refresh-files`
- `POST /training/label-review/raw-files`
- `POST /training/label-review/records/{index}`
- `POST /training/label-review/records/{index}/skip`
- `POST /training/label-review/records/{index}/detect-fight`
- `GET /training/video`

If these routes leave the official backend, `LabelReview.tsx` should leave the official frontend at the same time.

