# Local V-JEPA preview

The Highlight weights dropdown includes `vjepa21_highlight_best_10ep.pt`.
Choose it and the **Model Only** trim preset to inspect the trained timing without
the normal heuristic padding/trim adjustments. Other presets still work.

The model consumes dense 16-frame, 384-pixel, two-second windows directly from the
raw MP4, then predicts one continuous highlight using the saved temporal head and
decoder configuration. It does not support long-video montage yet (120s maximum).
The selected model is also honored when adding a processed clip to label review.

Local assets (ignored by Git): `checkpoints/vjepa21_highlight_best_10ep.pt` and
`external/vjepa2-main/` (official source including its MIT license).
Optional inference dependencies: `pip install -r requirements-vjepa.txt`.
No downloads or training occur on inference requests. Existing VideoMAE choices
remain available; the default is unchanged.

Run backend: `.venv/Scripts/python.exe -m uvicorn backend.main:app --host 127.0.0.1 --port 8000`
Run frontend from `frontend`: `npm run dev -- --host 127.0.0.1`
Open http://127.0.0.1:5173. Processing/export is local; posting requires a separate
explicit action in the dashboard.

## Inference speed settings

Restart the backend after pulling these changes. V-JEPA defaults to two windows
per GPU batch. Set `LOL_CLIP_VJEPA_BATCH_SIZE=1` for the single-window baseline,
or `4` to try a larger batch (accepted range 1–16). CPU uses one window.
CUDA out-of-memory retries halve the batch; a single-window failure is surfaced.
The encoder remains loaded between jobs, so compare warmed-up runs separately
from the first request. Small floating-point differences from batching are possible.

The pipeline decodes shared RGB frames once for V-JEPA and crop analysis. Dense
V-JEPA samples use the same raw-frame indices and direct bilinear resize as training;
they are not reconstructed from sparse 1080p crop-analysis frames. Their buffers
are released after prediction. The one-second step and saved decoder are unchanged.
FFmpeg still reads the input separately for final export.

Enable **Skip minimap detection** to skip both minimap extraction and detection.
Unused WAV extraction is skipped by the highlight pipeline; export still keeps audio.
Mixed-precision CUDA inference and the configured GPU video encoder remain available.

`GET /jobs/{job_id}` returns `detection_debug` as a JSON string. Parse it to inspect
`timings_seconds` (decode/model preparation, minimap, fight analysis, crop, export,
and total) and `vjepa_inference` (model preparation, cache hit, forward time,
window count, shared decoding, device, batch sizes, OOM retries). Stage timings
are also printed in backend logs. Forward timing includes GPU synchronization;
on the standalone fallback it also includes its own frame decoding.

Example PowerShell commands for the user (not executed during this update):

```powershell
$env:LOL_CLIP_VJEPA_BATCH_SIZE = "2"
.\.venv\Scripts\python.exe -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
```

After processing a clip, inspect its timings:

```powershell
$job = Invoke-RestMethod "http://127.0.0.1:8000/jobs/YOUR_JOB_ID"
$debug = $job.detection_debug | ConvertFrom-Json
$debug.timings_seconds
$debug.vjepa_inference
```

These speed changes have not been tested or benchmarked, at the user's request.
