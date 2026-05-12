from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import aiosqlite

SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    id TEXT PRIMARY KEY,
    status TEXT NOT NULL,
    stage TEXT,
    stage_failed TEXT,
    progress INTEGER DEFAULT 0,
    status_message TEXT DEFAULT '',
    flags TEXT NOT NULL DEFAULT '[]',
    error_detail TEXT,
    source_path TEXT,
    output_path TEXT,
    captions TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS training_runs (
    id TEXT,
    slice INTEGER,
    epoch INTEGER,
    train_loss REAL,
    val_loss REAL,
    accuracy REAL,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS oauth_tokens (
    platform TEXT PRIMARY KEY,
    encrypted_token BLOB NOT NULL,
    token_iv BLOB NOT NULL,
    expires_at INTEGER,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
"""


async def init_db(db_path: Path) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    async with aiosqlite.connect(db_path) as db:
        await db.executescript(SCHEMA)
        await _ensure_job_columns(db)
        await db.commit()


async def _ensure_job_columns(db: aiosqlite.Connection) -> None:
    cursor = await db.execute("PRAGMA table_info(jobs)")
    rows = await cursor.fetchall()
    columns = {row[1] for row in rows}
    if "progress" not in columns:
        await db.execute("ALTER TABLE jobs ADD COLUMN progress INTEGER DEFAULT 0")
    if "status_message" not in columns:
        await db.execute("ALTER TABLE jobs ADD COLUMN status_message TEXT DEFAULT ''")


async def create_job(db_path: Path, job_id: str, source_path: Path | str) -> None:
    await init_db(db_path)
    async with aiosqlite.connect(db_path) as db:
        await db.execute(
            """
            INSERT OR REPLACE INTO jobs
                (id, status, stage, progress, status_message, source_path)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (job_id, "queued", "queued", 0, "", str(source_path)),
        )
        await db.commit()


async def update_job(db_path: Path, job_id: str, **fields: Any) -> None:
    await init_db(db_path)
    if not fields:
        return
    normalized = {
        key: json.dumps(value) if key in {"flags", "captions"} and not isinstance(value, str) else value
        for key, value in fields.items()
    }
    normalized["updated_at"] = "CURRENT_TIMESTAMP"
    assignments = []
    values: list[Any] = []
    for key, value in normalized.items():
        if value == "CURRENT_TIMESTAMP":
            assignments.append(f"{key}=CURRENT_TIMESTAMP")
        else:
            assignments.append(f"{key}=?")
            values.append(value)
    values.append(job_id)
    async with aiosqlite.connect(db_path) as db:
        await db.execute(f"UPDATE jobs SET {', '.join(assignments)} WHERE id=?", values)
        await db.commit()


async def update_job_progress(
    db_path: Path,
    job_id: str,
    stage: str,
    progress: int,
    status_message: str = "",
) -> None:
    await init_db(db_path)
    async with aiosqlite.connect(str(db_path)) as db:
        await db.execute(
            """UPDATE jobs
               SET stage = ?, progress = ?, status_message = ?, updated_at = CURRENT_TIMESTAMP
               WHERE id = ?""",
            (stage, progress, status_message, job_id),
        )
        await db.commit()


async def get_job(db_path: Path, job_id: str) -> dict[str, Any] | None:
    await init_db(db_path)
    async with aiosqlite.connect(db_path) as db:
        db.row_factory = aiosqlite.Row
        cursor = await db.execute("SELECT * FROM jobs WHERE id=?", (job_id,))
        row = await cursor.fetchone()
    return dict(row) if row else None
