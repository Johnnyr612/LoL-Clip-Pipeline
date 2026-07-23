from __future__ import annotations

import asyncio

from backend import models


def test_list_jobs_returns_newest_first(tmp_path) -> None:
    asyncio.run(_run_list_jobs_returns_newest_first(tmp_path))


def test_mark_interrupted_jobs_failed_leaves_completed_jobs_alone(tmp_path) -> None:
    asyncio.run(_run_mark_interrupted_jobs_failed_leaves_completed_jobs_alone(tmp_path))


async def _run_list_jobs_returns_newest_first(tmp_path) -> None:
    db_path = tmp_path / "jobs.sqlite3"
    await models.create_job(db_path, "old-job", "old.mp4")
    await models.create_job(db_path, "new-job", "new.mp4")

    jobs = await models.list_jobs(db_path, limit=10)

    assert [job["id"] for job in jobs] == ["new-job", "old-job"]


async def _run_mark_interrupted_jobs_failed_leaves_completed_jobs_alone(tmp_path) -> None:
    db_path = tmp_path / "jobs.sqlite3"
    await models.create_job(db_path, "queued-job", "queued.mp4")
    await models.create_job(db_path, "complete-job", "complete.mp4")
    await models.update_job(db_path, "complete-job", status="complete", stage="complete", progress=100)

    await models.mark_interrupted_jobs_failed(db_path)

    queued = await models.get_job(db_path, "queued-job")
    complete = await models.get_job(db_path, "complete-job")
    assert queued is not None
    assert queued["status"] == "failed"
    assert queued["stage_failed"] == "queued"
    assert "backend restarted" in queued["error_detail"]
    assert complete is not None
    assert complete["status"] == "complete"
