from __future__ import annotations

import asyncio
import json
import logging
import secrets
import uuid
from pathlib import Path
from typing import Optional
from urllib.parse import urlencode

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import config, models
from .logging_config import setup_logging
from .pipeline import ClipPipeline
from .trainer import TrainingCoordinator
from .uploader import UploadResult, publish_instagram_job, publish_tiktok_job, save_tiktok_tokens, tiktok_connected

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
    mode: str = "post"
    description: Optional[str] = None


def _normalize_source_path(value: object) -> Path:
    raw = str(value or "").strip()
    quote_pairs = {('"', '"'), ("'", "'")}
    while len(raw) >= 2 and (raw[0], raw[-1]) in quote_pairs:
        raw = raw[1:-1].strip()
    return Path(raw)


def _tiktok_state_path() -> Path:
    return config.APPDATA_DIR / "tiktok_oauth_state.txt"


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


@app.get("/auth/tiktok/status")
async def tiktok_status() -> dict:
    return {
        "connected": tiktok_connected(),
        "oauth_configured": bool(config.TIKTOK_CLIENT_KEY and config.TIKTOK_CLIENT_SECRET),
    }


@app.get("/auth/tiktok/start")
async def start_tiktok_auth() -> RedirectResponse:
    if not config.TIKTOK_CLIENT_KEY or not config.TIKTOK_CLIENT_SECRET:
        raise HTTPException(
            status_code=503,
            detail="Set TIKTOK_CLIENT_KEY and TIKTOK_CLIENT_SECRET before connecting TikTok.",
        )
    config.APPDATA_DIR.mkdir(parents=True, exist_ok=True)
    state = secrets.token_urlsafe(24)
    _tiktok_state_path().write_text(state, encoding="utf-8")
    query = urlencode(
        {
            "client_key": config.TIKTOK_CLIENT_KEY,
            "scope": config.TIKTOK_SCOPES,
            "response_type": "code",
            "redirect_uri": config.TIKTOK_REDIRECT_URI,
            "state": state,
        }
    )
    return RedirectResponse(f"https://www.tiktok.com/v2/auth/authorize/?{query}")


@app.get("/auth/tiktok/callback")
async def tiktok_callback(code: str = "", state: str = "", error: str = "") -> HTMLResponse:
    if error:
        raise HTTPException(status_code=400, detail=error)
    if not code:
        raise HTTPException(status_code=400, detail="Missing TikTok authorization code")
    expected_state = _tiktok_state_path().read_text(encoding="utf-8") if _tiktok_state_path().exists() else ""
    if expected_state and state != expected_state:
        raise HTTPException(status_code=400, detail="Invalid TikTok OAuth state")
    import httpx

    async with httpx.AsyncClient(timeout=30) as client:
        response = await client.post(
            "https://open.tiktokapis.com/v2/oauth/token/",
            data={
                "client_key": config.TIKTOK_CLIENT_KEY,
                "client_secret": config.TIKTOK_CLIENT_SECRET,
                "code": code,
                "grant_type": "authorization_code",
                "redirect_uri": config.TIKTOK_REDIRECT_URI,
            },
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )
    if response.is_error:
        raise HTTPException(status_code=400, detail=response.text)
    save_tiktok_tokens(response.json())
    return HTMLResponse("<h1>TikTok connected</h1><p>You can close this tab and return to LoL Clip Pipeline.</p>")


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


@app.get("/jobs/{job_id}/source")
async def get_job_source(job_id: str) -> FileResponse:
    job = await models.get_job(config.DB_PATH, job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    source_path = Path(job.get("source_path") or "")
    if not source_path.exists() or not source_path.is_file():
        raise HTTPException(status_code=404, detail="Source video not found")
    if source_path.suffix.lower() != ".mp4":
        raise HTTPException(status_code=415, detail="Only .mp4 previews are supported")
    return FileResponse(source_path, media_type="video/mp4", filename=source_path.name)


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


@app.post("/upload/tiktok")
async def upload_tiktok(req: UploadRequest) -> dict:
    job = await models.get_job(config.DB_PATH, req.job_id)
    if not job:
        return {"status": "error", "detail": "Job not found"}
    try:
        mode = req.mode if req.mode in {"post", "draft"} else "post"
        result = await publish_tiktok_job(job, mode, req.description)
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
        return {"status": "ok", "post_url": result.public_url or "", "detail": result.raw_response}
    return {"status": "error", "detail": json.dumps(result.raw_response)}
