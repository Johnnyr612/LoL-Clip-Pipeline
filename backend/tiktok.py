from __future__ import annotations

import asyncio
import math
import secrets
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urlencode

import httpx

from . import config, models

TIKTOK_AUTH_URL = "https://www.tiktok.com/v2/auth/authorize/"
TIKTOK_TOKEN_URL = "https://open.tiktokapis.com/v2/oauth/token/"
TIKTOK_CREATOR_INFO_URL = "https://open.tiktokapis.com/v2/post/publish/creator_info/query/"
TIKTOK_DIRECT_POST_URL = "https://open.tiktokapis.com/v2/post/publish/video/init/"
TIKTOK_INBOX_UPLOAD_URL = "https://open.tiktokapis.com/v2/post/publish/inbox/video/init/"
TIKTOK_STATUS_URL = "https://open.tiktokapis.com/v2/post/publish/status/fetch/"
TIKTOK_STATUS_SUCCESS_INBOX = "SEND_TO_USER_INBOX"
TIKTOK_STATUS_FAILED = "FAILED"
TIKTOK_STATUS_MIN_POLL_INTERVAL_SEC = 10.0
TIKTOK_STATUS_MAX_POLL_SECONDS = 120.0

PostMode = Literal["inbox", "direct"]


class TikTokError(RuntimeError):
    pass


@dataclass(frozen=True)
class TikTokPostOptions:
    mode: PostMode = "inbox"
    title: str = ""
    privacy_level: str = "SELF_ONLY"
    disable_duet: bool = False
    disable_comment: bool = False
    disable_stitch: bool = False
    video_cover_timestamp_ms: int = 1000
    brand_content_toggle: bool = False
    brand_organic_toggle: bool = False
    is_aigc: bool = False


def _configured() -> bool:
    return bool(config.TIKTOK_CLIENT_KEY and config.TIKTOK_CLIENT_SECRET and config.TIKTOK_REDIRECT_URI)


def _require_configured() -> None:
    if not _configured():
        raise TikTokError("TikTok client key, client secret, and redirect URI must be configured")


def _scope_for_mode(mode: PostMode) -> str:
    return config.TIKTOK_DIRECT_SCOPES if mode == "direct" else config.TIKTOK_DEFAULT_SCOPES


async def build_authorization_url(db_path: Path, mode: PostMode = "inbox") -> str:
    _require_configured()
    state = secrets.token_urlsafe(32)
    await models.save_tiktok_oauth_state(db_path, state)
    query = urlencode(
        {
            "client_key": config.TIKTOK_CLIENT_KEY,
            "response_type": "code",
            "scope": _scope_for_mode(mode),
            "redirect_uri": config.TIKTOK_REDIRECT_URI,
            # TikTok recommends a state value so the callback cannot be forged.
            "state": state,
        }
    )
    return f"{TIKTOK_AUTH_URL}?{query}"


async def exchange_code(db_path: Path, code: str, state: str) -> dict[str, Any]:
    _require_configured()
    if not await models.consume_tiktok_oauth_state(db_path, state):
        raise TikTokError("TikTok OAuth state is invalid or expired")
    async with httpx.AsyncClient(timeout=30) as client:
        response = await client.post(
            TIKTOK_TOKEN_URL,
            data={
                "client_key": config.TIKTOK_CLIENT_KEY,
                "client_secret": config.TIKTOK_CLIENT_SECRET,
                "code": code,
                "grant_type": "authorization_code",
                "redirect_uri": config.TIKTOK_REDIRECT_URI,
            },
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )
    payload = _checked_json(response)
    await models.save_tiktok_token(db_path, payload)
    return payload


async def refresh_access_token(db_path: Path, token: dict[str, Any]) -> dict[str, Any]:
    _require_configured()
    async with httpx.AsyncClient(timeout=30) as client:
        response = await client.post(
            TIKTOK_TOKEN_URL,
            data={
                "client_key": config.TIKTOK_CLIENT_KEY,
                "client_secret": config.TIKTOK_CLIENT_SECRET,
                "grant_type": "refresh_token",
                "refresh_token": token["refresh_token"],
            },
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )
    payload = _checked_json(response)
    await models.save_tiktok_token(db_path, payload)
    stored = await models.get_tiktok_token(db_path)
    if stored is None:
        raise TikTokError("TikTok token refresh succeeded but token storage failed")
    return stored


async def get_valid_token(db_path: Path) -> dict[str, Any]:
    token = await models.get_tiktok_token(db_path)
    if token is None:
        raise TikTokError("TikTok is not connected")
    # Refresh a little early so a long upload does not start with a token that is
    # about to expire during the initialization request.
    if int(token["expires_at"]) <= int(time.time()) + 120:
        return await refresh_access_token(db_path, token)
    return token


async def connection_status(db_path: Path) -> dict[str, Any]:
    token = await models.get_tiktok_token(db_path)
    return {
        "configured": _configured(),
        "connected": token is not None,
        "open_id": token.get("open_id") if token else None,
        "scope": token.get("scope") if token else "",
        "expires_at": token.get("expires_at") if token else None,
        "refresh_expires_at": token.get("refresh_expires_at") if token else None,
    }


async def query_creator_info(db_path: Path) -> dict[str, Any]:
    token = await get_valid_token(db_path)
    async with httpx.AsyncClient(timeout=30) as client:
        response = await client.post(
            TIKTOK_CREATOR_INFO_URL,
            headers={
                "Authorization": f"Bearer {token['access_token']}",
                "Content-Type": "application/json; charset=UTF-8",
            },
        )
    return _checked_tiktok_response(response)


async def publish_video(db_path: Path, job_id: str, output_path: Path, options: TikTokPostOptions) -> dict[str, Any]:
    token = await get_valid_token(db_path)
    required_scope = "video.publish" if options.mode == "direct" else "video.upload"
    if not _has_scope(token.get("scope", ""), required_scope):
        raise TikTokError(f"TikTok account must be reconnected with the {required_scope} scope")
    if not output_path.exists() or output_path.suffix.lower() != ".mp4":
        raise TikTokError("TikTok posting requires a completed local .mp4 output")

    size = output_path.stat().st_size
    chunk_size, total_chunk_count = _upload_shape(size)
    body: dict[str, Any] = {
        "source_info": {
            "source": "FILE_UPLOAD",
            "video_size": size,
            "chunk_size": chunk_size,
            "total_chunk_count": total_chunk_count,
        }
    }
    url = TIKTOK_INBOX_UPLOAD_URL
    if options.mode == "direct":
        url = TIKTOK_DIRECT_POST_URL
        body["post_info"] = {
            "title": options.title,
            "privacy_level": options.privacy_level,
            "disable_duet": options.disable_duet,
            "disable_comment": options.disable_comment,
            "disable_stitch": options.disable_stitch,
            "video_cover_timestamp_ms": options.video_cover_timestamp_ms,
            "brand_content_toggle": options.brand_content_toggle,
            "brand_organic_toggle": options.brand_organic_toggle,
            "is_aigc": options.is_aigc,
        }

    async with httpx.AsyncClient(timeout=60) as client:
        init_response = await client.post(
            url,
            headers={
                "Authorization": f"Bearer {token['access_token']}",
                "Content-Type": "application/json; charset=UTF-8",
            },
            json=body,
        )
        init_payload = _checked_tiktok_response(init_response)
        data = init_payload.get("data", {})
        upload_url = data.get("upload_url")
        publish_id = data.get("publish_id")
        if not upload_url or not publish_id:
            raise TikTokError("TikTok did not return an upload_url and publish_id")

        # TikTok separates the init request from the binary transfer. The upload
        # URL is short-lived, so chunks are sent immediately and sequentially.
        await _upload_file(client, upload_url, output_path, size, chunk_size, total_chunk_count)

    await models.save_tiktok_publish_job(db_path, publish_id, job_id, options.mode, init_payload)
    result = {"publish_id": publish_id, "mode": options.mode, "init": init_payload}
    if options.mode == "inbox":
        result["status"] = "initialized"
    return result


async def fetch_publish_status(db_path: Path, publish_id: str) -> dict[str, Any]:
    token = await get_valid_token(db_path)
    async with httpx.AsyncClient(timeout=30) as client:
        response = await client.post(
            TIKTOK_STATUS_URL,
            headers={
                "Authorization": f"Bearer {token['access_token']}",
                "Content-Type": "application/json; charset=UTF-8",
            },
            json={"publish_id": publish_id},
        )
    payload = _checked_tiktok_response(response)
    status = _publish_status_value(payload)
    fail_reason = _publish_fail_reason(payload)
    await models.update_tiktok_publish_status(db_path, publish_id, status, payload, fail_reason)
    return _publish_status_response(publish_id, payload)


async def poll_publish_status_until_terminal(
    db_path: Path,
    publish_id: str,
    timeout_sec: float = TIKTOK_STATUS_MAX_POLL_SECONDS,
) -> dict[str, Any]:
    start = time.monotonic()
    interval = TIKTOK_STATUS_MIN_POLL_INTERVAL_SEC
    last_status: dict[str, Any] | None = None
    # TODO: Replace polling with a TikTok webhook receiver once this local app has a stable public callback URL.
    while time.monotonic() - start <= timeout_sec:
        last_status = await fetch_publish_status(db_path, publish_id)
        if last_status.get("terminal"):
            return last_status
        remaining = timeout_sec - (time.monotonic() - start)
        if remaining <= 0:
            break
        await asyncio.sleep(min(interval, remaining))
        interval = min(30.0, interval * 1.25)
    timed_out = last_status or {"publish_id": publish_id, "status": "UNKNOWN", "terminal": False, "success": False}
    return {**timed_out, "status": timed_out.get("status") or "UNKNOWN", "timed_out": True}


async def disconnect(db_path: Path) -> None:
    await models.delete_tiktok_token(db_path)


def _upload_shape(size: int) -> tuple[int, int]:
    if size <= 0:
        raise TikTokError("Cannot upload an empty video file")
    if size <= 64 * 1024 * 1024:
        return size, 1
    chunk_size = 10 * 1024 * 1024
    return chunk_size, max(1, math.floor(size / chunk_size))


def _has_scope(scope_text: str, required_scope: str) -> bool:
    scopes = {scope.strip() for scope in scope_text.split(",") if scope.strip()}
    return required_scope in scopes


def _publish_status_value(payload: dict[str, Any]) -> str:
    status = payload.get("data", {}).get("status")
    return str(status or "UNKNOWN")


def _publish_fail_reason(payload: dict[str, Any]) -> str | None:
    reason = payload.get("data", {}).get("fail_reason")
    text = str(reason or "").strip()
    return text or None


def _publish_status_response(publish_id: str, payload: dict[str, Any]) -> dict[str, Any]:
    status = _publish_status_value(payload)
    fail_reason = _publish_fail_reason(payload)
    return {
        "publish_id": publish_id,
        "status": status,
        "fail_reason": fail_reason,
        "terminal": status in {TIKTOK_STATUS_SUCCESS_INBOX, TIKTOK_STATUS_FAILED},
        "success": status == TIKTOK_STATUS_SUCCESS_INBOX,
        "response": payload,
    }


async def _upload_file(client: httpx.AsyncClient, upload_url: str, path: Path, size: int, chunk_size: int, total_chunks: int) -> None:
    with path.open("rb") as handle:
        for index in range(total_chunks):
            first = index * chunk_size
            last = size - 1 if index == total_chunks - 1 else min(size - 1, first + chunk_size - 1)
            handle.seek(first)
            data = handle.read(last - first + 1)
            response = await client.put(
                upload_url,
                content=data,
                headers={
                    "Content-Type": "video/mp4",
                    "Content-Length": str(len(data)),
                    "Content-Range": f"bytes {first}-{last}/{size}",
                },
            )
            if response.status_code not in {200, 201, 206}:
                raise TikTokError(f"TikTok upload failed: {response.status_code} {response.text}")


def _checked_json(response: httpx.Response) -> dict[str, Any]:
    try:
        response.raise_for_status()
        payload = response.json()
    except Exception as exc:  # noqa: BLE001 - convert library exceptions into UI-safe messages.
        raise TikTokError(f"TikTok request failed: {exc}") from exc
    if "error" in payload and payload["error"].get("code") not in {None, "ok"}:
        raise TikTokError(payload["error"].get("message") or payload["error"].get("code") or "TikTok API error")
    return payload


def _checked_tiktok_response(response: httpx.Response) -> dict[str, Any]:
    payload = _checked_json(response)
    error = payload.get("error", {})
    if error and error.get("code") != "ok":
        raise TikTokError(error.get("message") or error.get("code") or "TikTok API error")
    return payload
