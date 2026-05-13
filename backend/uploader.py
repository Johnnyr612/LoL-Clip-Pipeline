from __future__ import annotations

import asyncio
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import httpx

from . import config


@dataclass(frozen=True)
class UploadResult:
    platform: str
    ok: bool
    public_url: str | None
    raw_response: dict[str, Any]
    auth_required: bool = False


def _tiktok_token_path() -> Path:
    return config.APPDATA_DIR / "tiktok_tokens.json"


def load_tiktok_tokens() -> dict[str, Any]:
    token_path = _tiktok_token_path()
    if not token_path.exists():
        return {}
    try:
        return json.loads(token_path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def save_tiktok_tokens(payload: dict[str, Any]) -> None:
    token_path = _tiktok_token_path()
    token_path.parent.mkdir(parents=True, exist_ok=True)
    now = int(time.time())
    normalized = dict(payload)
    if "expires_in" in normalized:
        normalized["expires_at"] = now + int(normalized.get("expires_in") or 0)
    if "refresh_expires_in" in normalized:
        normalized["refresh_expires_at"] = now + int(normalized.get("refresh_expires_in") or 0)
    token_path.write_text(json.dumps(normalized), encoding="utf-8")


async def get_tiktok_access_token() -> str | None:
    env_token = os.environ.get("TIKTOK_ACCESS_TOKEN")
    if env_token:
        return env_token
    tokens = load_tiktok_tokens()
    access_token = tokens.get("access_token")
    expires_at = int(tokens.get("expires_at") or 0)
    if isinstance(access_token, str) and access_token and expires_at > int(time.time()) + 120:
        return access_token
    refresh_token = tokens.get("refresh_token")
    if not refresh_token or not config.TIKTOK_CLIENT_KEY or not config.TIKTOK_CLIENT_SECRET:
        return access_token if isinstance(access_token, str) and access_token else None
    async with httpx.AsyncClient(timeout=30) as client:
        response = await client.post(
            "https://open.tiktokapis.com/v2/oauth/token/",
            data={
                "client_key": config.TIKTOK_CLIENT_KEY,
                "client_secret": config.TIKTOK_CLIENT_SECRET,
                "grant_type": "refresh_token",
                "refresh_token": refresh_token,
            },
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )
    if response.is_error:
        return access_token if isinstance(access_token, str) and access_token else None
    refreshed = response.json()
    save_tiktok_tokens(refreshed)
    next_token = refreshed.get("access_token")
    return next_token if isinstance(next_token, str) else None


def tiktok_connected() -> bool:
    if os.environ.get("TIKTOK_ACCESS_TOKEN"):
        return True
    tokens = load_tiktok_tokens()
    return bool(tokens.get("access_token") or tokens.get("refresh_token"))


def caption_for_platform(captions_value: Any, platform: str) -> str:
    if isinstance(captions_value, str):
        try:
            captions = json.loads(captions_value)
        except json.JSONDecodeError:
            return captions_value
    elif isinstance(captions_value, dict):
        captions = captions_value
    else:
        captions = {}

    platform_payload = captions.get(platform, {})
    if isinstance(platform_payload, dict):
        caption = platform_payload.get("caption")
        hashtags = platform_payload.get("hashtags", [])
        if isinstance(caption, str) and isinstance(hashtags, list):
            tags = " ".join(str(tag) for tag in hashtags)
            return f"{caption}\n\n{tags}".strip()
        if isinstance(caption, str):
            return caption
    return ""


async def publish_tiktok_job(job: dict[str, Any], mode: str = "post", description: str | None = None) -> UploadResult:
    output_path = _job_output_path(job)
    if output_path is None:
        return UploadResult("tiktok", False, None, {"error": "missing_output_path"})
    access_token = await get_tiktok_access_token()
    if not access_token:
        return UploadResult(
            "tiktok",
            False,
            None,
            {"error": "missing_tiktok_access_token"},
            auth_required=True,
        )
    caption = description if description is not None else caption_for_platform(job.get("captions"), "tiktok")
    if mode == "draft":
        return await TikTokClient(access_token).upload_to_inbox(output_path, caption)
    return await TikTokClient(access_token).upload_and_publish(output_path, caption)


async def publish_instagram_job(job: dict[str, Any]) -> UploadResult:
    output_path = _job_output_path(job)
    if output_path is None:
        return UploadResult(
            "instagram",
            False,
            None,
            {"error": "missing_output_path"},
        )
    access_token = os.environ.get("INSTAGRAM_ACCESS_TOKEN")
    ig_user_id = os.environ.get("INSTAGRAM_USER_ID")
    public_base_url = os.environ.get("PUBLIC_OUTPUT_BASE_URL")
    if not access_token:
        return UploadResult(
            "instagram",
            False,
            None,
            {"error": "missing_instagram_access_token"},
            auth_required=True,
        )
    if not ig_user_id:
        return UploadResult(
            "instagram",
            False,
            None,
            {"error": "missing_instagram_user_id"},
            auth_required=True,
        )
    if not public_base_url:
        return UploadResult(
            "instagram",
            False,
            None,
            {"error": "missing_public_output_base_url"},
        )
    video_url = f"{public_base_url.rstrip('/')}/{output_path.name}"
    caption = caption_for_platform(job.get("captions"), "instagram")
    return await InstagramClient(access_token, ig_user_id).upload_and_publish(video_url, caption)


class TikTokClient:
    def __init__(self, access_token: str):
        self.access_token = access_token

    async def upload_and_publish(self, video_path: Path, caption: str) -> UploadResult:
        async with httpx.AsyncClient(timeout=120) as client:
            headers = {"Authorization": f"Bearer {self.access_token}"}
            video_size = video_path.stat().st_size
            creator = await client.post(
                "https://open.tiktokapis.com/v2/post/publish/creator_info/query/",
                headers={**headers, "Content-Type": "application/json; charset=UTF-8"},
                json={},
            )
            if creator.status_code == 401:
                return UploadResult("tiktok", False, None, _safe_json(creator), auth_required=True)
            if creator.is_error:
                return UploadResult("tiktok", False, None, _safe_json(creator))
            creator_data = creator.json().get("data", {})
            privacy_options = creator_data.get("privacy_level_options") or []
            privacy_level = config.TIKTOK_PRIVACY_LEVEL
            if privacy_options and privacy_level not in privacy_options:
                privacy_level = "SELF_ONLY" if "SELF_ONLY" in privacy_options else privacy_options[0]
            init = await client.post(
                "https://open.tiktokapis.com/v2/post/publish/video/init/",
                headers={**headers, "Content-Type": "application/json; charset=UTF-8"},
                json={
                    "post_info": {
                        "title": caption,
                        "privacy_level": privacy_level,
                        "disable_duet": False,
                        "disable_comment": False,
                        "disable_stitch": False,
                        "brand_content_toggle": False,
                        "brand_organic_toggle": False,
                    },
                    "source_info": {
                        "source": "FILE_UPLOAD",
                        "video_size": video_size,
                        "chunk_size": video_size,
                        "total_chunk_count": 1,
                    },
                },
            )
            if init.status_code == 401:
                return UploadResult("tiktok", False, None, init.json(), auth_required=True)
            if init.is_error:
                return UploadResult("tiktok", False, None, _safe_json(init))
            upload_url = init.json().get("data", {}).get("upload_url")
            publish_id = init.json().get("data", {}).get("publish_id")
            if not upload_url:
                return UploadResult("tiktok", False, None, init.json())
            with video_path.open("rb") as handle:
                upload = await client.put(
                    upload_url,
                    content=handle.read(),
                    headers={
                        "Content-Type": "video/mp4",
                        "Content-Length": str(video_size),
                        "Content-Range": f"bytes 0-{video_size - 1}/{video_size}",
                    },
                )
            if upload.is_error:
                return UploadResult("tiktok", False, None, _safe_json(upload))
            return UploadResult("tiktok", True, None, {"publish_id": publish_id, "creator": creator.json(), "init": init.json(), "upload": _safe_json(upload)})

    async def upload_to_inbox(self, video_path: Path, description: str) -> UploadResult:
        async with httpx.AsyncClient(timeout=120) as client:
            headers = {"Authorization": f"Bearer {self.access_token}"}
            video_size = video_path.stat().st_size
            init = await client.post(
                "https://open.tiktokapis.com/v2/post/publish/inbox/video/init/",
                headers={**headers, "Content-Type": "application/json; charset=UTF-8"},
                json={
                    "source_info": {
                        "source": "FILE_UPLOAD",
                        "video_size": video_size,
                        "chunk_size": video_size,
                        "total_chunk_count": 1,
                    },
                },
            )
            if init.status_code == 401:
                return UploadResult("tiktok", False, None, init.json(), auth_required=True)
            if init.is_error:
                return UploadResult("tiktok", False, None, _safe_json(init))
            upload_url = init.json().get("data", {}).get("upload_url")
            publish_id = init.json().get("data", {}).get("publish_id")
            if not upload_url:
                return UploadResult("tiktok", False, None, init.json())
            with video_path.open("rb") as handle:
                upload = await client.put(
                    upload_url,
                    content=handle.read(),
                    headers={
                        "Content-Type": "video/mp4",
                        "Content-Length": str(video_size),
                        "Content-Range": f"bytes 0-{video_size - 1}/{video_size}",
                    },
                )
            if upload.is_error:
                return UploadResult("tiktok", False, None, _safe_json(upload))
            return UploadResult(
                "tiktok",
                True,
                None,
                {
                    "publish_id": publish_id,
                    "mode": "draft",
                    "description": description,
                    "note": "TikTok inbox uploads require finishing the post in the TikTok app. Copy the returned description if TikTok does not prefill it.",
                    "init": init.json(),
                    "upload": _safe_json(upload),
                },
            )


class InstagramClient:
    def __init__(self, access_token: str, ig_user_id: str):
        self.access_token = access_token
        self.ig_user_id = ig_user_id

    async def upload_and_publish(self, video_url: str, caption: str) -> UploadResult:
        async with httpx.AsyncClient(timeout=120) as client:
            base = f"https://graph.facebook.com/v19.0/{self.ig_user_id}"
            create = await client.post(
                f"{base}/media",
                data={"media_type": "REELS", "video_url": video_url, "caption": caption, "access_token": self.access_token},
            )
            if create.status_code == 401:
                return UploadResult("instagram", False, None, _safe_json(create), auth_required=True)
            if create.is_error:
                return UploadResult("instagram", False, None, _safe_json(create))
            creation_id = create.json().get("id")
            for _ in range(24):
                status = await client.get(f"https://graph.facebook.com/v19.0/{creation_id}", params={"fields": "status_code", "access_token": self.access_token})
                payload = _safe_json(status)
                status_code = payload.get("status_code")
                if status_code == "FINISHED":
                    publish = await client.post(f"{base}/media_publish", data={"creation_id": creation_id, "access_token": self.access_token})
                    raw = _safe_json(publish)
                    return UploadResult("instagram", not publish.is_error, raw.get("permalink"), raw)
                if status_code == "ERROR":
                    return UploadResult("instagram", False, None, payload)
                await asyncio.sleep(5)
            return UploadResult("instagram", False, None, {"error": "container_status_timeout", "creation_id": creation_id})


def _safe_json(response: httpx.Response) -> dict[str, Any]:
    try:
        return response.json()
    except Exception:  # noqa: BLE001
        return {"status_code": response.status_code, "text": response.text}


def _job_output_path(job: dict[str, Any]) -> Optional[Path]:
    raw_path = job.get("output_path")
    if not raw_path:
        return None
    output_path = Path(str(raw_path))
    if not output_path.exists():
        return None
    return output_path
