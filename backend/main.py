from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles

from . import config, models
from .logging_config import setup_logging
from .pipeline import ClipPipeline, InputValidationError
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
            await pipeline.run(source_path)

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
        raise HTTPException(status_code=503, detail="Pipeline not ready")
    source_path = Path(payload.get("source_path", ""))
    try:
        async with job_semaphore:
            job_id = await pipeline.run(source_path, payload.get("job_id"))
    except InputValidationError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return {"job_id": job_id}


@app.post("/train")
async def start_training() -> dict:
    run_id = await trainer.start()
    return {"run_id": run_id}


@app.get("/train/stream")
async def train_stream() -> StreamingResponse:
    async def events():
        async for metric in trainer.stream():
            yield f"data: {json.dumps(metric)}\n\n"

    return StreamingResponse(events(), media_type="text/event-stream")
