from __future__ import annotations

import asyncio
import json
import logging
import uuid
from pathlib import Path
from typing import Literal, Optional
from urllib.parse import urlencode

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import config, models
from .logging_config import setup_logging
from .pipeline import ClipPipeline
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
app.mount("/outputs", StaticFiles(directory=str(config.OUTPUT_DIR)), name="outputs")

job_semaphore = asyncio.Semaphore(2)
pipeline: ClipPipeline | None = None
trainer = TrainingCoordinator()


class TrainRequest(BaseModel):
    clips_dir: str
    labels: str
    epochs: int = 25
    batch_size: Optional[int] = None


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


def _normalize_source_path(value: object) -> Path:
    raw = str(value or "").strip()
    quote_pairs = {('"', '"'), ("'", "'")}
    while len(raw) >= 2 and (raw[0], raw[-1]) in quote_pairs:
        raw = raw[1:-1].strip()
    return Path(raw)


@app.on_event("startup")
async def startup() -> None:
    global pipeline
    config.APPDATA_DIR.mkdir(parents=True, exist_ok=True)
    config.TEMP_DIR.mkdir(parents=True, exist_ok=True)
    config.LOG_DIR.mkdir(parents=True, exist_ok=True)
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    await models.init_db(config.DB_PATH)
    pipeline = ClipPipeline(config.DB_PATH)


@app.get("/health")
async def health() -> dict:
    return {"ok": True}


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


@app.post("/jobs")
async def create_job(file: UploadFile = File(...)) -> dict:
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not ready")
    if not file.filename or not file.filename.lower().endswith(".mp4"):
        raise HTTPException(status_code=422, detail="Only .mp4 uploads are supported")
    upload_dir = config.APPDATA_DIR / "uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)
    source_path = upload_dir / Path(file.filename).name
    with source_path.open("wb") as handle:
        while chunk := await file.read(1024 * 1024):
            handle.write(chunk)

    async def run_background() -> None:
        async with job_semaphore:
            await asyncio.to_thread(
                lambda: asyncio.run(pipeline.run(source_path))
            )

    asyncio.create_task(run_background())
    return {"accepted": True, "source_path": str(source_path)}


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
                lambda: asyncio.run(pipeline.run(source_path, job_id))
            )

    asyncio.create_task(run_background())

    # Return job_id immediately - frontend polls for progress
    return {"job_id": job_id}


@app.post("/train")
async def start_training(req: TrainRequest) -> dict:
    run_id = await trainer.start(
        clips_dir=Path(req.clips_dir),
        labels=Path(req.labels),
        epochs=req.epochs,
        batch_size=req.batch_size,
    )
    return {"run_id": run_id}


@app.get("/train/stream")
async def train_stream() -> StreamingResponse:
    async def events():
        async for metric in trainer.stream():
            yield f"data: {json.dumps(metric)}\n\n"

    return StreamingResponse(events(), media_type="text/event-stream")
