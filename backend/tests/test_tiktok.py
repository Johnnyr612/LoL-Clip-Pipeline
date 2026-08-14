from __future__ import annotations

import asyncio
import time

import aiosqlite
import pytest

from backend import config, models, tiktok


class _FakeResponse:
    def __init__(self, payload: dict, status_code: int = 200):
        self._payload = payload
        self.status_code = status_code
        self.text = str(payload)

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise RuntimeError(self.text)

    def json(self) -> dict:
        return self._payload


class _FakeTikTokClient:
    posts: list[tuple[str, dict | None, dict | None]] = []
    puts: list[tuple[str, dict | None, bytes]] = []
    status_payloads: list[dict] = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_args):
        return None

    async def post(self, url: str, data=None, json=None, headers=None):
        self.posts.append((url, data, json))
        if url == tiktok.TIKTOK_TOKEN_URL:
            return _FakeResponse(
                {
                    "access_token": "access",
                    "refresh_token": "refresh",
                    "expires_in": 86400,
                    "refresh_expires_in": 31536000,
                    "open_id": "creator-open-id",
                    "scope": "user.info.basic,video.upload",
                    "token_type": "Bearer",
                }
            )
        if url == tiktok.TIKTOK_INBOX_UPLOAD_URL:
            return _FakeResponse(
                {
                    "data": {"publish_id": "pub123", "upload_url": "https://upload.example/video"},
                    "error": {"code": "ok", "message": ""},
                }
            )
        if url == tiktok.TIKTOK_STATUS_URL:
            if self.status_payloads:
                return _FakeResponse(self.status_payloads.pop(0))
            return _FakeResponse(
                {
                    "data": {"status": "SEND_TO_USER_INBOX"},
                    "error": {"code": "ok", "message": ""},
                }
            )
        return _FakeResponse({"data": {}, "error": {"code": "ok", "message": ""}})

    async def put(self, url: str, content=None, headers=None):
        self.puts.append((url, headers, content))
        return _FakeResponse({}, 201)


@pytest.fixture(autouse=True)
def tiktok_env(monkeypatch):
    monkeypatch.setattr(config, "TIKTOK_CLIENT_KEY", "client-key")
    monkeypatch.setattr(config, "TIKTOK_CLIENT_SECRET", "client-secret")
    monkeypatch.setattr(config, "TIKTOK_REDIRECT_URI", "https://example.test/tiktok/callback")
    _FakeTikTokClient.posts = []
    _FakeTikTokClient.puts = []
    _FakeTikTokClient.status_payloads = []
    monkeypatch.setattr(tiktok.httpx, "AsyncClient", lambda *_, **__: _FakeTikTokClient())


def test_tiktok_oauth_state_and_token_exchange(tmp_path):
    asyncio.run(_run_tiktok_oauth_state_and_token_exchange(tmp_path))


async def _run_tiktok_oauth_state_and_token_exchange(tmp_path):
    db_path = tmp_path / "tokens.sqlite3"
    auth_url = await tiktok.build_authorization_url(db_path)
    state = auth_url.split("state=", 1)[1].split("&", 1)[0]

    token = await tiktok.exchange_code(db_path, "auth-code", state)
    stored = await models.get_tiktok_token(db_path)

    assert token["access_token"] == "access"
    assert stored is not None
    assert stored["open_id"] == "creator-open-id"
    assert _FakeTikTokClient.posts[0][1]["grant_type"] == "authorization_code"


def test_tiktok_publish_uploads_completed_mp4(tmp_path):
    asyncio.run(_run_tiktok_publish_uploads_completed_mp4(tmp_path))


async def _run_tiktok_publish_uploads_completed_mp4(tmp_path):
    db_path = tmp_path / "publish.sqlite3"
    video_path = tmp_path / "clip.mp4"
    video_path.write_bytes(b"1234")
    await models.create_job(db_path, "job-1", "source.mp4")
    await models.save_tiktok_token(
        db_path,
        {
            "access_token": "access",
            "refresh_token": "refresh",
            "expires_in": 86400,
            "refresh_expires_in": 31536000,
            "open_id": "creator-open-id",
            "scope": "user.info.basic,video.upload",
            "token_type": "Bearer",
        },
    )

    result = await tiktok.publish_video(db_path, "job-1", video_path, tiktok.TikTokPostOptions(mode="inbox"))

    assert result["publish_id"] == "pub123"
    assert result["status"] == "initialized"
    assert _FakeTikTokClient.posts[0][0] == tiktok.TIKTOK_INBOX_UPLOAD_URL
    assert _FakeTikTokClient.posts[0][2]["source_info"]["video_size"] == 4
    assert _FakeTikTokClient.puts[0][1]["Content-Range"] == "bytes 0-3/4"
    assert _FakeTikTokClient.puts[0][2] == b"1234"
    job = await models.get_job(db_path, "job-1")
    assert job is not None
    assert job["tiktok_publish_id"] == "pub123"
    assert job["tiktok_publish_mode"] == "inbox"
    assert job["tiktok_publish_status"] == "initialized"


def test_tiktok_status_fetch_updates_job_record(tmp_path):
    asyncio.run(_run_tiktok_status_fetch_updates_job_record(tmp_path))


async def _run_tiktok_status_fetch_updates_job_record(tmp_path):
    db_path = tmp_path / "status.sqlite3"
    await models.create_job(db_path, "job-1", "source.mp4")
    await models.save_tiktok_token(
        db_path,
        {
            "access_token": "access",
            "refresh_token": "refresh",
            "expires_in": 86400,
            "refresh_expires_in": 31536000,
            "open_id": "creator-open-id",
            "scope": "user.info.basic,video.upload",
            "token_type": "Bearer",
        },
    )
    await models.save_tiktok_publish_job(
        db_path,
        "pub123",
        "job-1",
        "inbox",
        {"data": {"publish_id": "pub123"}, "error": {"code": "ok", "message": ""}},
    )
    _FakeTikTokClient.status_payloads = [
        {"data": {"status": "SEND_TO_USER_INBOX"}, "error": {"code": "ok", "message": ""}},
    ]

    result = await tiktok.fetch_publish_status(db_path, "pub123")

    assert result["status"] == "SEND_TO_USER_INBOX"
    assert result["terminal"] is True
    assert result["success"] is True
    job = await models.get_job(db_path, "job-1")
    assert job is not None
    assert job["tiktok_publish_status"] == "SEND_TO_USER_INBOX"
    assert job["tiktok_publish_fail_reason"] is None


def test_tiktok_status_fetch_surfaces_fail_reason(tmp_path):
    asyncio.run(_run_tiktok_status_fetch_surfaces_fail_reason(tmp_path))


async def _run_tiktok_status_fetch_surfaces_fail_reason(tmp_path):
    db_path = tmp_path / "failed.sqlite3"
    await models.create_job(db_path, "job-1", "source.mp4")
    await models.save_tiktok_token(
        db_path,
        {
            "access_token": "access",
            "refresh_token": "refresh",
            "expires_in": 86400,
            "refresh_expires_in": 31536000,
            "open_id": "creator-open-id",
            "scope": "user.info.basic,video.upload",
            "token_type": "Bearer",
        },
    )
    await models.save_tiktok_publish_job(
        db_path,
        "pub123",
        "job-1",
        "inbox",
        {"data": {"publish_id": "pub123"}, "error": {"code": "ok", "message": ""}},
    )
    _FakeTikTokClient.status_payloads = [
        {"data": {"status": "FAILED", "fail_reason": "bad file"}, "error": {"code": "ok", "message": ""}},
    ]

    result = await tiktok.fetch_publish_status(db_path, "pub123")

    assert result["status"] == "FAILED"
    assert result["terminal"] is True
    assert result["fail_reason"] == "bad file"
    job = await models.get_job(db_path, "job-1")
    assert job is not None
    assert job["tiktok_publish_status"] == "FAILED"
    assert job["tiktok_publish_fail_reason"] == "bad file"


def test_tiktok_status_poll_stops_on_terminal(tmp_path, monkeypatch):
    asyncio.run(_run_tiktok_status_poll_stops_on_terminal(tmp_path, monkeypatch))


async def _run_tiktok_status_poll_stops_on_terminal(tmp_path, monkeypatch):
    db_path = tmp_path / "poll.sqlite3"
    sleeps: list[float] = []

    async def fake_sleep(duration: float) -> None:
        sleeps.append(duration)

    monkeypatch.setattr(tiktok.asyncio, "sleep", fake_sleep)
    await models.create_job(db_path, "job-1", "source.mp4")
    await models.save_tiktok_token(
        db_path,
        {
            "access_token": "access",
            "refresh_token": "refresh",
            "expires_in": 86400,
            "refresh_expires_in": 31536000,
            "open_id": "creator-open-id",
            "scope": "user.info.basic,video.upload",
            "token_type": "Bearer",
        },
    )
    await models.save_tiktok_publish_job(
        db_path,
        "pub123",
        "job-1",
        "inbox",
        {"data": {"publish_id": "pub123"}, "error": {"code": "ok", "message": ""}},
    )
    _FakeTikTokClient.status_payloads = [
        {"data": {"status": "PROCESSING_UPLOAD"}, "error": {"code": "ok", "message": ""}},
        {"data": {"status": "SEND_TO_USER_INBOX"}, "error": {"code": "ok", "message": ""}},
    ]

    result = await tiktok.poll_publish_status_until_terminal(db_path, "pub123")

    assert result["status"] == "SEND_TO_USER_INBOX"
    assert sleeps == [tiktok.TIKTOK_STATUS_MIN_POLL_INTERVAL_SEC]


def test_tiktok_refreshes_expired_token(tmp_path):
    asyncio.run(_run_tiktok_refreshes_expired_token(tmp_path))


async def _run_tiktok_refreshes_expired_token(tmp_path):
    db_path = tmp_path / "refresh.sqlite3"
    await models.init_db(db_path)
    async with aiosqlite.connect(db_path) as db:
        await db.execute(
            """
            INSERT INTO tiktok_tokens
                (platform, access_token, refresh_token, expires_at)
            VALUES ('tiktok', 'old-access', 'old-refresh', ?)
            """,
            (int(time.time()) - 10,),
        )
        await db.commit()

    token = await tiktok.get_valid_token(db_path)

    assert token["access_token"] == "access"
    assert _FakeTikTokClient.posts[0][1]["grant_type"] == "refresh_token"
