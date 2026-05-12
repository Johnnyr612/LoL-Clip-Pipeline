from __future__ import annotations

import base64
import json
from dataclasses import dataclass
from typing import Sequence

import cv2
import httpx
import numpy as np

from . import config


@dataclass(frozen=True)
class VisionFightResult:
    player_champion: str
    enemy_champions: list[str]
    fight_type: str
    confidence: float


def classify_fight_participants(
    frames: np.ndarray,
    timestamps: np.ndarray,
    clip_start: float,
    clip_end: float,
) -> VisionFightResult | None:
    if not config.OPENAI_API_KEY or len(frames) == 0 or len(timestamps) == 0:
        return None

    images = _sample_frame_images(frames, timestamps, clip_start, clip_end)
    if not images:
        return None

    content: list[dict] = [
        {
            "type": "input_text",
            "text": (
                "Identify League of Legends champions actively fighting in these raw gameplay frames. "
                f"The player's summoner name is {config.PLAYER_NAME}; the player has the green health bar. "
                "Only include enemy champions with red health bars who are fighting or about to fight the player. "
                "Ignore unrelated champions elsewhere on the minimap, HUD portraits, side portraits, kill feed, and chat. "
                "Return JSON only with keys: player_champion, enemy_champions, fight_type, confidence. "
                "Use exact champion names. If unsure about a champion, omit it instead of guessing."
            ),
        }
    ]
    for image in images:
        content.append({"type": "input_image", "image_url": f"data:image/jpeg;base64,{image}"})

    payload = {
        "model": config.VISION_CLASSIFIER_MODEL,
        "input": [{"role": "user", "content": content}],
        "text": {"format": {"type": "json_object"}},
    }
    try:
        with httpx.Client(timeout=config.VISION_CLASSIFIER_TIMEOUT_SEC) as client:
            response = client.post(
                "https://api.openai.com/v1/responses",
                headers={"Authorization": f"Bearer {config.OPENAI_API_KEY}", "Content-Type": "application/json"},
                json=payload,
            )
            response.raise_for_status()
        parsed = _parse_response_json(response.json())
    except Exception:
        return None

    player = _clean_name(parsed.get("player_champion"))
    enemies = [_clean_name(name) for name in parsed.get("enemy_champions", []) if _clean_name(name)]
    confidence = _as_float(parsed.get("confidence"))
    if not player or confidence < config.VISION_CLASSIFIER_MIN_CONFIDENCE:
        return None
    fight_type = parsed.get("fight_type")
    if not isinstance(fight_type, str) or "v" not in fight_type:
        fight_type = f"1v{max(1, len(enemies))}"
    return VisionFightResult(player, enemies[:5], fight_type, confidence)


def _sample_frame_images(
    frames: np.ndarray,
    timestamps: np.ndarray,
    clip_start: float,
    clip_end: float,
) -> list[str]:
    sample_times = np.linspace(clip_start, clip_end, min(3, max(1, len(frames))))
    images: list[str] = []
    used_indexes: set[int] = set()
    for timestamp in sample_times:
        index = int(np.argmin(np.abs(timestamps - timestamp)))
        if index in used_indexes:
            continue
        used_indexes.add(index)
        frame = frames[index]
        resized = cv2.resize(frame, (1280, 720), interpolation=cv2.INTER_AREA)
        ok, encoded = cv2.imencode(".jpg", cv2.cvtColor(resized, cv2.COLOR_RGB2BGR), [int(cv2.IMWRITE_JPEG_QUALITY), 86])
        if ok:
            images.append(base64.b64encode(encoded.tobytes()).decode("ascii"))
    return images


def _parse_response_json(payload: dict) -> dict:
    text = payload.get("output_text")
    if isinstance(text, str):
        return json.loads(text)
    for item in payload.get("output", []):
        for content in item.get("content", []):
            if content.get("type") in {"output_text", "text"} and isinstance(content.get("text"), str):
                return json.loads(content["text"])
    return {}


def _clean_name(value: object) -> str:
    if not isinstance(value, str):
        return ""
    name = value.strip()
    if not name or name.lower().startswith("unknown"):
        return ""
    return name


def _as_float(value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0
