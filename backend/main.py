from __future__ import annotations

import asyncio
import json
import logging
import uuid
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import config, models
from .logging_config import setup_logging
from .pipeline import ClipPipeline
from .trainer import TrainingCoordinator
from .uploader import UploadResult, publish_instagram_job, publish_tiktok_job

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


class UploadRequest(BaseModel):
    job_id: str


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
    source_path = Path(payload.get("source_path", ""))

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


@app.post("/upload/tiktok")
async def upload_tiktok(req: UploadRequest) -> dict:
    job = await models.get_job(config.DB_PATH, req.job_id)
    if not job:
        return {"status": "error", "detail": "Job not found"}
    try:
        result = await publish_tiktok_job(job)
        return _upload_response(result)
    except Exception as exc:  # noqa: BLE001 - upload routes always return structured responses.
        return {"status": "error", "detail": str(exc)}


@app.post("/upload/instagram")
async def upload_instagram(req: UploadRequest) -> dict:
    job = await models.get_job(config.DB_PATH, req.job_id)
    if not job:
        return {"status": "error", "detail": "Job not found"}
    try:
        result = await publish_instagram_job(job)
        return _upload_response(result)
    except Exception as exc:  # noqa: BLE001 - upload routes always return structured responses.
        return {"status": "error", "detail": str(exc)}


def _upload_response(result: UploadResult) -> dict:
    if result.ok:
        return {"status": "ok", "post_url": result.public_url or ""}
    return {"status": "error", "detail": json.dumps(result.raw_response)}
