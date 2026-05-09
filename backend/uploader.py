from __future__ import annotations

import asyncio
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import httpx


@dataclass(frozen=True)
class UploadResult:
    platform: str
    ok: bool
    public_url: str | None
    raw_response: dict[str, Any]
    auth_required: bool = False


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


async def publish_tiktok_job(job: dict[str, Any]) -> UploadResult:
    output_path = _job_output_path(job)
    if output_path is None:
        return UploadResult("tiktok", False, None, {"error": "missing_output_path"})
    access_token = os.environ.get("TIKTOK_ACCESS_TOKEN")
    if not access_token:
        return UploadResult(
            "tiktok",
            False,
            None,
            {"error": "missing_tiktok_access_token"},
            auth_required=True,
        )
    caption = caption_for_platform(job.get("captions"), "tiktok")
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
        # Endpoint choreography is intentionally explicit; callers surface raw errors.
        async with httpx.AsyncClient(timeout=120) as client:
            headers = {"Authorization": f"Bearer {self.access_token}"}
            init = await client.post("https://open.tiktokapis.com/v2/post/publish/inbox/video/init/", headers=headers, json={"post_info": {"title": caption}})
            if init.status_code == 401:
                return UploadResult("tiktok", False, None, init.json(), auth_required=True)
            if init.is_error:
                return UploadResult("tiktok", False, None, _safe_json(init))
            upload_url = init.json().get("data", {}).get("upload_url")
            publish_id = init.json().get("data", {}).get("publish_id")
            if not upload_url:
                return UploadResult("tiktok", False, None, init.json())
            with video_path.open("rb") as handle:
                upload = await client.put(upload_url, content=handle.read(), headers={"Content-Type": "video/mp4"})
            if upload.is_error:
                return UploadResult("tiktok", False, None, _safe_json(upload))
            return UploadResult("tiktok", True, None, {"publish_id": publish_id, "init": init.json(), "upload": _safe_json(upload)})


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
