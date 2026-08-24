from __future__ import annotations

import json
import time
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
    tiktok_publish_id TEXT,
    tiktok_publish_mode TEXT,
    tiktok_publish_status TEXT,
    tiktok_publish_fail_reason TEXT,
    detection_debug TEXT NOT NULL DEFAULT '{}',
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

CREATE TABLE IF NOT EXISTS tiktok_tokens (
    platform TEXT PRIMARY KEY,
    open_id TEXT,
    scope TEXT,
    access_token TEXT NOT NULL,
    refresh_token TEXT NOT NULL,
    token_type TEXT DEFAULT 'Bearer',
    expires_at INTEGER NOT NULL,
    refresh_expires_at INTEGER,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS tiktok_oauth_states (
    state TEXT PRIMARY KEY,
    created_at INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS tiktok_publish_jobs (
    publish_id TEXT PRIMARY KEY,
    job_id TEXT NOT NULL,
    mode TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'initialized',
    response TEXT NOT NULL DEFAULT '{}',
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
    if "detection_debug" not in columns:
        await db.execute("ALTER TABLE jobs ADD COLUMN detection_debug TEXT NOT NULL DEFAULT '{}'")
    if "tiktok_publish_id" not in columns:
        await db.execute("ALTER TABLE jobs ADD COLUMN tiktok_publish_id TEXT")
    if "tiktok_publish_mode" not in columns:
        await db.execute("ALTER TABLE jobs ADD COLUMN tiktok_publish_mode TEXT")
    if "tiktok_publish_status" not in columns:
        await db.execute("ALTER TABLE jobs ADD COLUMN tiktok_publish_status TEXT")
    if "tiktok_publish_fail_reason" not in columns:
        await db.execute("ALTER TABLE jobs ADD COLUMN tiktok_publish_fail_reason TEXT")


async def save_tiktok_oauth_state(db_path: Path, state: str) -> None:
    await init_db(db_path)
    async with aiosqlite.connect(db_path) as db:
        await db.execute(
            "INSERT OR REPLACE INTO tiktok_oauth_states (state, created_at) VALUES (?, ?)",
            (state, int(time.time())),
        )
        await db.commit()


async def consume_tiktok_oauth_state(db_path: Path, state: str, max_age_sec: int = 600) -> bool:
    await init_db(db_path)
    now = int(time.time())
    async with aiosqlite.connect(db_path) as db:
        cursor = await db.execute("SELECT created_at FROM tiktok_oauth_states WHERE state=?", (state,))
        row = await cursor.fetchone()
        await db.execute("DELETE FROM tiktok_oauth_states WHERE state=?", (state,))
        await db.commit()
    if row is None:
        return False
    return now - int(row[0]) <= max_age_sec


async def save_tiktok_token(db_path: Path, token: dict[str, Any]) -> None:
    await init_db(db_path)
    now = int(time.time())
    expires_at = now + int(token.get("expires_in", 0))
    refresh_expires_at = now + int(token.get("refresh_expires_in", 0)) if token.get("refresh_expires_in") else None
    async with aiosqlite.connect(db_path) as db:
        # This is local desktop storage. Encrypt or move these tokens to a managed
        # secret store before exposing the app as a hosted multi-user service.
        await db.execute(
            """
            INSERT OR REPLACE INTO tiktok_tokens
                (platform, open_id, scope, access_token, refresh_token, token_type, expires_at, refresh_expires_at, updated_at)
            VALUES ('tiktok', ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            """,
            (
                token.get("open_id"),
                token.get("scope", ""),
                token["access_token"],
                token["refresh_token"],
                token.get("token_type", "Bearer"),
                expires_at,
                refresh_expires_at,
            ),
        )
        await db.commit()


async def get_tiktok_token(db_path: Path) -> dict[str, Any] | None:
    await init_db(db_path)
    async with aiosqlite.connect(db_path) as db:
        db.row_factory = aiosqlite.Row
        cursor = await db.execute("SELECT * FROM tiktok_tokens WHERE platform='tiktok'")
        row = await cursor.fetchone()
    return dict(row) if row else None


async def delete_tiktok_token(db_path: Path) -> None:
    await init_db(db_path)
    async with aiosqlite.connect(db_path) as db:
        await db.execute("DELETE FROM tiktok_tokens WHERE platform='tiktok'")
        await db.commit()


async def save_tiktok_publish_job(db_path: Path, publish_id: str, job_id: str, mode: str, response: dict[str, Any]) -> None:
    await init_db(db_path)
    async with aiosqlite.connect(db_path) as db:
        await db.execute(
            """
            INSERT OR REPLACE INTO tiktok_publish_jobs
                (publish_id, job_id, mode, status, response, updated_at)
            VALUES (?, ?, ?, 'initialized', ?, CURRENT_TIMESTAMP)
            """,
            (publish_id, job_id, mode, json.dumps(response)),
        )
        if mode == "inbox":
            await db.execute(
                """
                UPDATE jobs
                SET tiktok_publish_id=?,
                    tiktok_publish_mode=?,
                    tiktok_publish_status='initialized',
                    tiktok_publish_fail_reason=NULL,
                    updated_at=CURRENT_TIMESTAMP
                WHERE id=?
                """,
                (publish_id, mode, job_id),
            )
        await db.commit()


async def get_tiktok_publish_job(db_path: Path, publish_id: str) -> dict[str, Any] | None:
    await init_db(db_path)
    async with aiosqlite.connect(db_path) as db:
        db.row_factory = aiosqlite.Row
        cursor = await db.execute("SELECT * FROM tiktok_publish_jobs WHERE publish_id=?", (publish_id,))
        row = await cursor.fetchone()
    return dict(row) if row else None


async def update_tiktok_publish_status(
    db_path: Path,
    publish_id: str,
    status: str,
    response: dict[str, Any],
    fail_reason: str | None = None,
) -> dict[str, Any] | None:
    await init_db(db_path)
    async with aiosqlite.connect(db_path) as db:
        db.row_factory = aiosqlite.Row
        cursor = await db.execute("SELECT job_id, mode FROM tiktok_publish_jobs WHERE publish_id=?", (publish_id,))
        row = await cursor.fetchone()
        if row is None:
            return None
        await db.execute(
            """
            UPDATE tiktok_publish_jobs
            SET status=?, response=?, updated_at=CURRENT_TIMESTAMP
            WHERE publish_id=?
            """,
            (status, json.dumps(response), publish_id),
        )
        await db.execute(
            """
            UPDATE jobs
            SET tiktok_publish_id=?,
                tiktok_publish_mode=?,
                tiktok_publish_status=?,
                tiktok_publish_fail_reason=?,
                updated_at=CURRENT_TIMESTAMP
            WHERE id=?
            """,
            (publish_id, row["mode"], status, fail_reason, row["job_id"]),
        )
        await db.commit()
    return {"job_id": row["job_id"], "mode": row["mode"], "status": status, "fail_reason": fail_reason}


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
    normalized = {}
    for key, value in fields.items():
        if key in {"flags", "detection_debug"} and not isinstance(value, str):
            normalized[key] = json.dumps(value)
        elif isinstance(value, Path):
            normalized[key] = str(value)
        else:
            normalized[key] = value
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


async def mark_interrupted_jobs_failed(db_path: Path) -> None:
    await init_db(db_path)
    async with aiosqlite.connect(db_path) as db:
        await db.execute(
            """
            UPDATE jobs
            SET status = 'failed',
                stage_failed = COALESCE(stage, 'queued'),
                status_message = 'Interrupted by server restart',
                error_detail = 'This job was queued or running when the backend restarted. Start it again to process the clip.',
                updated_at = CURRENT_TIMESTAMP
            WHERE status IN ('queued', 'running')
            """
        )
        await db.commit()


async def list_jobs(db_path: Path, limit: int = 50) -> list[dict[str, Any]]:
    await init_db(db_path)
    safe_limit = max(1, min(limit, 200))
    async with aiosqlite.connect(db_path) as db:
        db.row_factory = aiosqlite.Row
        cursor = await db.execute(
            """
            SELECT * FROM jobs
            ORDER BY datetime(created_at) DESC, rowid DESC
            LIMIT ?
            """,
            (safe_limit,),
        )
        rows = await cursor.fetchall()
    return [dict(row) for row in rows]


async def list_tiktok_source_usages(db_path: Path) -> dict[str, str]:
    await init_db(db_path)
    async with aiosqlite.connect(db_path) as db:
        db.row_factory = aiosqlite.Row
        cursor = await db.execute(
            """
            SELECT
                jobs.source_path,
                COALESCE(tiktok_publish_jobs.mode, jobs.tiktok_publish_mode) AS mode,
                COALESCE(tiktok_publish_jobs.status, jobs.tiktok_publish_status) AS publish_status
            FROM jobs
            LEFT JOIN tiktok_publish_jobs ON tiktok_publish_jobs.job_id = jobs.id
            WHERE jobs.source_path IS NOT NULL
              AND (
                tiktok_publish_jobs.publish_id IS NOT NULL
                OR jobs.tiktok_publish_id IS NOT NULL
              )
            """
        )
        rows = await cursor.fetchall()

    usages: dict[str, str] = {}
    for row in rows:
        source_path = str(row["source_path"] or "").strip()
        if not source_path:
            continue
        publish_status = str(row["publish_status"] or "").strip().upper()
        if publish_status == "FAILED":
            continue
        mode = str(row["mode"] or "").strip().lower()
        usage = "sent_to_inbox" if mode == "inbox" else "posted_to_tiktok"
        usages[source_path] = usage
    return usages
