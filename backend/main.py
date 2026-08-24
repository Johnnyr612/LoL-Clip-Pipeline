from __future__ import annotations

import asyncio
import json
import logging
import os
import platform
import subprocess
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal, Optional
from urllib.parse import urlencode

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, RedirectResponse, StreamingResponse
from pydantic import BaseModel

from . import config, models
from .logging_config import setup_logging
from .label_review import (
    AddRawFileRequest,
    LabelReviewUpdate,
    add_raw_file_to_review_queue,
    delete_label_review_record,
    detect_label_review_record_with_videomae,
    get_label_review_payload,
    match_label_review_record,
    regenerate_trainer_labels,
    save_label_review_record,
    skip_label_review_record,
    validated_video_path,
)
from .pipeline import ClipPipeline, ProcessingSettings
from .cropper import CropSettings
from .fight_detector import TrimSettings
from .tiktok import TikTokError, TikTokPostOptions
from . import tiktok
from .trainer import TrainingCoordinator

logging.basicConfig(level=logging.INFO)
setup_logging()

app = FastAPI(title="LoL Clip Pipeline")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

job_semaphore = asyncio.Semaphore(1)
pipeline: ClipPipeline | None = None
trainer = TrainingCoordinator()


class TrainRequest(BaseModel):
    task: Literal["highlight", "fight"] = "highlight"
    clips_dir: str = ""
    labels: str
    epochs: int = 25
    batch_size: Optional[int] = None
    freeze_backbone: bool = True
    unfreeze_last_n_layers: int = 2
    classifier_lr: float = 1e-4
    backbone_lr: float = 1e-5
    val_fraction: float = 0.15
    progress_interval: int = 5


class TikTokPostRequest(BaseModel):
    mode: Literal["inbox", "direct"] = "inbox"
    title: str = ""
    privacy_level: str = "SELF_ONLY"
    disable_duet: bool = False
    disable_comment: bool = False
    disable_stitch: bool = False
    video_cover_timestamp_ms: int = 1000
    brand_content_toggle: bool = False
    brand_organic_toggle: bool = False
    is_aigc: bool = False


class TrimSettingsRequest(BaseModel):
    fight_start_preroll_sec: float = config.FIGHT_START_PREROLL_SEC
    output_context_padding_sec: float = config.OUTPUT_CONTEXT_PADDING_SEC
    combat_event_end_padding_sec: float = config.COMBAT_EVENT_END_PADDING_SEC
    max_pre_fight_lead_sec: float = config.MAX_PRE_FIGHT_LEAD_SEC
    min_clip_duration_sec: float = config.COMBAT_EVENT_MIN_CLIP_DURATION_SEC
    model_only: bool = False

    def to_trim_settings(self) -> TrimSettings:
        return TrimSettings(
            fight_start_preroll_sec=_clamp_float(self.fight_start_preroll_sec, 0.0, 6.0),
            output_context_padding_sec=_clamp_float(self.output_context_padding_sec, 0.0, 5.0),
            combat_event_end_padding_sec=_clamp_float(self.combat_event_end_padding_sec, 0.0, 8.0),
            max_pre_fight_lead_sec=_clamp_float(self.max_pre_fight_lead_sec, 0.0, 8.0),
            min_clip_duration_sec=_clamp_float(self.min_clip_duration_sec, 5.0, 45.0),
            model_only=bool(self.model_only),
        )


class CropSettingsRequest(BaseModel):
    mode: str = config.CROP_MODE
    transition: str = config.CROP_TRANSITION

    def to_crop_settings(self) -> CropSettings:
        mode = str(self.mode or config.CROP_MODE).strip().lower()
        transition = str(self.transition or config.CROP_TRANSITION).strip().lower()
        if mode not in {"static", "dynamic"}:
            mode = "dynamic"
        if transition not in {"cut", "pan"}:
            transition = config.CROP_TRANSITION
        return CropSettings(mode=mode, transition=transition)


class ProcessingSettingsRequest(BaseModel):
    skip_minimap_detection: bool = config.SKIP_MINIMAP_DETECTION

    def to_processing_settings(self) -> ProcessingSettings:
        return ProcessingSettings(skip_minimap_detection=bool(self.skip_minimap_detection))


class OpenFolderRequest(BaseModel):
    path: str


def _clamp_float(value: object, minimum: float, maximum: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        number = minimum
    return max(minimum, min(maximum, number))


def _normalize_source_path(value: object) -> Path:
    raw = str(value or "").strip()
    quote_pairs = {('"', '"'), ("'", "'")}
    while len(raw) >= 2 and (raw[0], raw[-1]) in quote_pairs:
        raw = raw[1:-1].strip()
    return Path(raw)


def _normalize_folder_path(value: object) -> Path:
    raw = str(value or "").strip()
    quote_pairs = {('"', '"'), ("'", "'")}
    while len(raw) >= 2 and (raw[0], raw[-1]) in quote_pairs:
        raw = raw[1:-1].strip()
    return Path(raw).expanduser().resolve()


def _folder_payload(key: str, label: str, path: Path, kind: str) -> dict:
    resolved = path.expanduser().resolve()
    return {
        "key": key,
        "label": label,
        "path": str(resolved),
        "kind": kind,
        "exists": resolved.is_dir(),
    }


def _open_folder_in_file_manager(path: Path) -> None:
    system = platform.system().lower()
    if system == "windows":
        os.startfile(str(path))  # type: ignore[attr-defined]
    elif system == "darwin":
        subprocess.Popen(["open", str(path)])
    else:
        subprocess.Popen(["xdg-open", str(path)])


async def _open_existing_folder(raw_path: object) -> dict:
    path = _normalize_folder_path(raw_path)
    if not path.is_dir():
        raise HTTPException(status_code=404, detail=f"Folder not found: {path}")
    await asyncio.to_thread(_open_folder_in_file_manager, path)
    return {"opened": True, "path": str(path)}


def _checkpoints_root() -> Path:
    return (config.PROJECT_ROOT / "checkpoints").resolve()


def _checkpoint_payload(path: Path) -> dict:
    stat = path.stat()
    return {
        "filename": path.name,
        "path": str(path),
        "size": stat.st_size,
        "modified_at": datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat(),
        "active": path.resolve() == config.VIDEOMAE_HIGHLIGHT_CHECKPOINT.resolve(),
    }


def _resolve_highlight_checkpoint(value: object) -> Path:
    raw = str(value or "").strip()
    if not raw:
        return config.VIDEOMAE_HIGHLIGHT_CHECKPOINT.resolve()
    quote_pairs = {('"', '"'), ("'", "'")}
    while len(raw) >= 2 and (raw[0], raw[-1]) in quote_pairs:
        raw = raw[1:-1].strip()

    root = _checkpoints_root()
    candidate = Path(raw)
    if not candidate.is_absolute():
        candidate = root / candidate
    resolved = candidate.resolve()
    if not resolved.is_relative_to(root):
        raise HTTPException(status_code=422, detail="Checkpoint must be inside the project checkpoints folder")
    if resolved.suffix.lower() != ".pt":
        raise HTTPException(status_code=422, detail="Checkpoint must be a .pt file")
    if not resolved.is_file():
        raise HTTPException(status_code=404, detail=f"Checkpoint not found: {resolved}")
    return resolved


def _resolve_output_path(relative_path: str) -> Path:
    root = config.OUTPUT_DIR.resolve()
    candidate = (root / relative_path).resolve()
    if not candidate.is_relative_to(root) or not candidate.is_file():
        raise HTTPException(status_code=404, detail="Output file not found")
    return candidate


async def _mark_job_label_review_result(job_id: str, result: dict) -> None:
    job = await models.get_job(config.DB_PATH, job_id)
    if not job:
        return
    try:
        flags = json.loads(job.get("flags") or "[]")
    except json.JSONDecodeError:
        flags = []
    if not isinstance(flags, list):
        flags = []
    label_flag = "label_review_autolabeled" if result.get("created") else "label_review_existing_record"
    next_flags = [*flags]
    if label_flag not in next_flags:
        next_flags.append(label_flag)

    try:
        detection_debug = json.loads(job.get("detection_debug") or "{}")
    except json.JSONDecodeError:
        detection_debug = {}
    if not isinstance(detection_debug, dict):
        detection_debug = {}
    detection_debug["label_review"] = {
        "record_index": result.get("record_index"),
        "created": bool(result.get("created")),
        "status": result.get("record", {}).get("review_status"),
    }
    await models.update_job(config.DB_PATH, job_id, flags=next_flags, detection_debug=detection_debug)


async def _add_source_to_label_review(job_id: str, source_path: Path, highlight_checkpoint_path: Path) -> None:
    try:
        raw_file_usage = await models.list_tiktok_source_usages(config.DB_PATH)
        result = await asyncio.to_thread(add_raw_file_to_review_queue, str(source_path), highlight_checkpoint_path, raw_file_usage)
    except Exception as exc:  # noqa: BLE001 - background annotation should not overwrite pipeline success.
        job = await models.get_job(config.DB_PATH, job_id)
        if not job:
            return
        try:
            flags = json.loads(job.get("flags") or "[]")
        except json.JSONDecodeError:
            flags = []
        if not isinstance(flags, list):
            flags = []
        next_flags = [*flags]
        if "label_review_autolabel_failed" not in next_flags:
            next_flags.append("label_review_autolabel_failed")
        try:
            detection_debug = json.loads(job.get("detection_debug") or "{}")
        except json.JSONDecodeError:
            detection_debug = {}
        if not isinstance(detection_debug, dict):
            detection_debug = {}
        detection_debug["label_review"] = {"error": str(exc)}
        await models.update_job(config.DB_PATH, job_id, flags=next_flags, detection_debug=detection_debug)
        return
    await _mark_job_label_review_result(job_id, result)


async def _label_review_payload_with_usage() -> dict:
    return get_label_review_payload(await models.list_tiktok_source_usages(config.DB_PATH))


def _stream_video_with_range(video_path: Path, request: Request) -> StreamingResponse:
    file_size = video_path.stat().st_size
    range_header = request.headers.get("range")
    start = 0
    end = file_size - 1
    status_code = 200

    if range_header:
        units, _, raw_range = range_header.partition("=")
        if units.strip().lower() != "bytes" or "-" not in raw_range:
            raise HTTPException(status_code=416, detail="Invalid range header")
        raw_start, raw_end = raw_range.split("-", 1)
        try:
            if raw_start:
                start = int(raw_start)
                end = int(raw_end) if raw_end else file_size - 1
            elif raw_end:
                suffix_length = int(raw_end)
                start = max(0, file_size - suffix_length)
        except ValueError as exc:
            raise HTTPException(status_code=416, detail="Invalid range header") from exc
        if start > end or start >= file_size:
            raise HTTPException(status_code=416, detail="Requested range not satisfiable")
        end = min(end, file_size - 1)
        status_code = 206

    content_length = end - start + 1

    def iter_file():
        with video_path.open("rb") as handle:
            handle.seek(start)
            remaining = content_length
            while remaining > 0:
                chunk = handle.read(min(1024 * 1024, remaining))
                if not chunk:
                    break
                remaining -= len(chunk)
                yield chunk

    headers = {
        "Accept-Ranges": "bytes",
        "Content-Length": str(content_length),
        "Content-Type": "video/mp4",
    }
    if status_code == 206:
        headers["Content-Range"] = f"bytes {start}-{end}/{file_size}"
    return StreamingResponse(iter_file(), status_code=status_code, media_type="video/mp4", headers=headers)


@app.on_event("startup")
async def startup() -> None:
    global pipeline
    config.APPDATA_DIR.mkdir(parents=True, exist_ok=True)
    config.TEMP_DIR.mkdir(parents=True, exist_ok=True)
    config.LOG_DIR.mkdir(parents=True, exist_ok=True)
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    await models.init_db(config.DB_PATH)
    await models.mark_interrupted_jobs_failed(config.DB_PATH)
    pipeline = ClipPipeline(config.DB_PATH)


@app.get("/health")
async def health() -> dict:
    return {"ok": True}


@app.get("/settings/folders")
async def settings_folders() -> dict:
    upload_dir = config.APPDATA_DIR / "uploads"
    folders = [
        _folder_payload("output", "Output clips", config.OUTPUT_DIR, "output"),
        _folder_payload("uploads", "Uploaded input clips", upload_dir, "input"),
        _folder_payload("project", "Project folder", config.PROJECT_ROOT, "project"),
        _folder_payload("checkpoints", "Checkpoint weights", config.PROJECT_ROOT / "checkpoints", "weights"),
        _folder_payload("logs", "Logs", config.LOG_DIR, "logs"),
        _folder_payload("temp", "Temp files", config.TEMP_DIR, "temp"),
    ]
    return {"folders": folders}


@app.post("/settings/open-folder")
async def open_settings_folder(req: OpenFolderRequest) -> dict:
    return await _open_existing_folder(req.path)


@app.get("/tiktok/status")
async def tiktok_status() -> dict:
    return await tiktok.connection_status(config.DB_PATH)


@app.get("/tiktok/auth")
async def tiktok_auth(mode: Literal["inbox", "direct"] = "inbox") -> RedirectResponse:
    try:
        url = await tiktok.build_authorization_url(config.DB_PATH, mode)
    except TikTokError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return RedirectResponse(url)


@app.get("/tiktok/callback")
async def tiktok_callback(code: str = "", state: str = "", error: str = "") -> RedirectResponse:
    if error:
        return RedirectResponse(f"{config.TIKTOK_AUTH_SUCCESS_URL}?{urlencode({'tiktok': 'error', 'detail': error})}")
    try:
        await tiktok.exchange_code(config.DB_PATH, code, state)
    except TikTokError as exc:
        return RedirectResponse(f"{config.TIKTOK_AUTH_SUCCESS_URL}?{urlencode({'tiktok': 'error', 'detail': str(exc)})}")
    return RedirectResponse(f"{config.TIKTOK_AUTH_SUCCESS_URL}?{urlencode({'tiktok': 'connected'})}")


@app.post("/tiktok/disconnect")
async def tiktok_disconnect() -> dict:
    await tiktok.disconnect(config.DB_PATH)
    return {"connected": False}


@app.get("/tiktok/creator-info")
async def tiktok_creator_info() -> dict:
    try:
        return await tiktok.query_creator_info(config.DB_PATH)
    except TikTokError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/tiktok/jobs/{job_id}/publish")
async def tiktok_publish_job(job_id: str, req: TikTokPostRequest) -> dict:
    job = await models.get_job(config.DB_PATH, job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if job.get("status") != "complete" or not job.get("output_path"):
        raise HTTPException(status_code=422, detail="Job must be complete before it can be sent to TikTok")

    try:
        request_payload = req.model_dump() if hasattr(req, "model_dump") else req.dict()
        return await tiktok.publish_video(
            config.DB_PATH,
            job_id,
            Path(str(job["output_path"])),
            TikTokPostOptions(**request_payload),
        )
    except TikTokError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/tiktok/publish/{publish_id}/status")
async def tiktok_publish_status(publish_id: str) -> dict:
    try:
        return await tiktok.fetch_publish_status(config.DB_PATH, publish_id)
    except TikTokError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/jobs")
async def list_jobs(limit: int = 50) -> dict:
    return {"jobs": await models.list_jobs(config.DB_PATH, limit=limit)}


@app.get("/output-files")
async def list_output_files(limit: int = 50) -> dict:
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    safe_limit = max(1, min(limit, 200))
    files = []
    for path in config.OUTPUT_DIR.glob("*.mp4"):
        try:
            stat = path.stat()
        except OSError:
            continue
        files.append(
            {
                "filename": path.name,
                "path": str(path),
                "url": f"/outputs/{path.name}",
                "size": stat.st_size,
                "modified_at": datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat(),
            }
        )
    files.sort(key=lambda item: item["modified_at"], reverse=True)
    return {"output_dir": str(config.OUTPUT_DIR), "files": files[:safe_limit]}


@app.get("/checkpoints/highlight")
async def list_highlight_checkpoints() -> dict:
    root = _checkpoints_root()
    root.mkdir(parents=True, exist_ok=True)
    files = []
    for path in root.glob("videomae_lol_highlight*.pt"):
        try:
            files.append(_checkpoint_payload(path.resolve()))
        except OSError:
            continue
    files.sort(key=lambda item: item["modified_at"], reverse=True)
    return {
        "checkpoint_dir": str(root),
        "default_checkpoint": str(config.VIDEOMAE_HIGHLIGHT_CHECKPOINT.resolve()),
        "checkpoints": files,
    }


@app.get("/outputs/{relative_path:path}")
async def output_file(relative_path: str, request: Request):
    output_path = _resolve_output_path(relative_path)
    if output_path.suffix.lower() != ".mp4":
        return FileResponse(output_path)
    return _stream_video_with_range(output_path, request)


@app.get("/jobs/{job_id}")
async def get_job(job_id: str) -> dict:
    job = await models.get_job(config.DB_PATH, job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return job


@app.post("/process")
async def process_existing(payload: dict) -> dict:
    if pipeline is None:
        raise HTTPException(
            status_code=503, detail="Pipeline not ready"
        )
    source_path = _normalize_source_path(payload.get("source_path", ""))
    trim_settings = TrimSettingsRequest(**(payload.get("trim_settings") or {})).to_trim_settings()
    crop_settings = CropSettingsRequest(**(payload.get("crop_settings") or {})).to_crop_settings()
    processing_settings = ProcessingSettingsRequest(**(payload.get("processing_settings") or {})).to_processing_settings()
    highlight_checkpoint_path = _resolve_highlight_checkpoint(payload.get("highlight_checkpoint", ""))
    add_to_label_review = bool(payload.get("add_to_label_review", False))

    # Validate input before starting background task
    try:
        from .pipeline import validate_input, InputValidationError
        validate_input(source_path)
    except InputValidationError as exc:
        raise HTTPException(
            status_code=422, detail=str(exc)
        ) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=422, detail=str(exc)
        ) from exc

    # Create job record immediately
    job_id = str(uuid.uuid4())
    await models.create_job(config.DB_PATH, job_id, str(source_path))

    # Run pipeline in background - do not await
    async def run_background() -> None:
        async with job_semaphore:
            await asyncio.to_thread(
                lambda: asyncio.run(pipeline.run(source_path, job_id, trim_settings, highlight_checkpoint_path, crop_settings, processing_settings))
            )
            if add_to_label_review:
                completed_job = await models.get_job(config.DB_PATH, job_id)
                if completed_job and completed_job.get("status") == "complete":
                    await _add_source_to_label_review(job_id, source_path, highlight_checkpoint_path)

    asyncio.create_task(run_background())

    # Return job_id immediately - frontend polls for progress
    return {"job_id": job_id}


@app.post("/train")
async def start_training(req: TrainRequest) -> dict:
    run_id = await trainer.start(
        clips_dir=Path(req.clips_dir) if req.clips_dir else None,
        labels=Path(req.labels),
        epochs=req.epochs,
        batch_size=req.batch_size,
        freeze_backbone=req.freeze_backbone,
        unfreeze_last_n_layers=req.unfreeze_last_n_layers,
        classifier_lr=req.classifier_lr,
        backbone_lr=req.backbone_lr,
        val_fraction=req.val_fraction,
        progress_interval=req.progress_interval,
        task=req.task,
    )
    return {"run_id": run_id}


@app.get("/train/stream")
async def train_stream() -> StreamingResponse:
    async def events():
        async for metric in trainer.stream():
            yield f"data: {json.dumps(metric)}\n\n"

    return StreamingResponse(events(), media_type="text/event-stream")

@app.get("/training/label-review")
async def training_label_review() -> dict:
    return await _label_review_payload_with_usage()


@app.post("/training/label-review/refresh-files")
async def refresh_training_label_files() -> dict:
    return await _label_review_payload_with_usage()


@app.post("/training/label-review/raw-files")
async def add_training_raw_file(req: AddRawFileRequest) -> dict:
    raw_file_usage = await models.list_tiktok_source_usages(config.DB_PATH)
    result = await asyncio.to_thread(add_raw_file_to_review_queue, req.path, None, raw_file_usage)
    return {**result, "payload": await _label_review_payload_with_usage()}


@app.post("/training/label-review/records/{record_index}")
async def update_training_label_review(record_index: int, update: LabelReviewUpdate) -> dict:
    return save_label_review_record(record_index, update)




@app.post("/training/label-review/records/{record_index}/skip")
async def skip_training_label_record(record_index: int) -> dict:
    return skip_label_review_record(record_index)


@app.delete("/training/label-review/records/{record_index}")
async def delete_training_label_record(record_index: int) -> dict:
    result = delete_label_review_record(record_index)
    return {**result, "payload": await _label_review_payload_with_usage()}


@app.post("/training/label-review/records/{record_index}/match-start")
async def match_training_label_start(record_index: int) -> dict:
    return match_label_review_record(record_index)


@app.post("/training/label-review/records/{record_index}/detect-fight")
async def detect_training_label_fight(record_index: int) -> dict:
    return await asyncio.to_thread(detect_label_review_record_with_videomae, record_index)

@app.post("/training/label-review/regenerate")
async def regenerate_training_labels() -> dict:
    return regenerate_trainer_labels()


@app.get("/training/video")
async def training_review_video(path: str, request: Request) -> StreamingResponse:
    video_path = validated_video_path(path)
    return _stream_video_with_range(video_path, request)

