from __future__ import annotations

import json
import threading
from dataclasses import dataclass
from typing import Sequence

import httpx

from . import config


@dataclass(frozen=True)
class CaptionResult:
    captions: dict[str, dict]
    flags: list[str]


def _clean_champion_names(champion_names: Sequence[str]) -> list[str]:
    names: list[str] = []
    seen: set[str] = set()
    for name in champion_names:
        if not name or name.startswith("unknown"):
            continue
        if name in seen:
            continue
        names.append(name)
        seen.add(name)
    return names


def _fallback_caption(
    player_champion: str,
    enemy_names: Sequence[str],
    fight_type: str,
    platform: str,
    minimap_champions: Sequence[str] | None = None,
) -> dict:
    clean_enemies = _clean_champion_names(enemy_names)
    minimap_context = [name for name in _clean_champion_names(minimap_champions or []) if name != player_champion]
    enemy_text = ", ".join(clean_enemies or minimap_context[:4]) or "the enemy"
    if len(clean_enemies or minimap_context) == 1:
        hook = f"{player_champion} took the {(clean_enemies or minimap_context)[0]} duel"
    else:
        hook = f"{player_champion} turned this {fight_type} into a highlight"
    body_limit = 300 if platform == "tiktok" else 500
    caption = (
        f"{hook}\n"
        f"Clean spacing, clutch timing, and {enemy_text} had one chance to back off. "
        f"Would you take this fight? #leagueoflegends"
    )[:body_limit]
    tags = ["#gaming", "#gamer", "#clips", "#fyp", "#viral", "#leagueoflegends", "#lolclips", "#riotgames", "#summonersrift", "#leagueclips"]
    return {"caption": caption, "hashtags": tags, "hook_line": hook}


def _build_prompt(
    player_champion: str,
    enemy_names: Sequence[str],
    minimap_champions: Sequence[str],
    fight_type: str,
    fight_duration: float,
    dialog_text: str,
    platform: str,
) -> str:
    enemies = ", ".join(_clean_champion_names(enemy_names)) or "unknown enemies"
    minimap_context = ", ".join(_clean_champion_names(minimap_champions)) or "unknown"
    return f"""Write a viral social media caption for this LoL clip.

Clip details:
- Player champion: {player_champion}
- Enemies fought: {enemies}
- Champions actively fighting or collapsing on the fight from original minimap/HUD: {minimap_context}
- Fight type: {fight_type}
- Fight duration: {fight_duration:.1f} seconds
- Spoken lines detected: {dialog_text or 'none'}
- Platform: {platform}

Rules:
- First line: hook only, no hashtags, no generic openings
- 3-5 emojis placed naturally in text, not clustered at end
- Exactly 1 rhetorical question or call-to-action
- If any champion is 'unknown': say 'the enemy', never guess name
- TikTok: body max 300 chars, punchy, meme-aware
- Instagram: body max 500 chars, slightly more narrative
- Hashtag block at end: 5 platform tags + 5 LoL-specific tags
- Never write: 'Check this out', 'Amazing', 'Watch this', 'Incredible'
- Use only the listed active-fight champions; do not pull in unrelated champions from elsewhere on the map

Respond with valid JSON only, no markdown, no preamble:
{{"caption": "...", "hashtags": ["..."], "hook_line": "..."}}"""


class CaptionGenerator:
    def __init__(self) -> None:
        self.api_key = config.OPENAI_API_KEY
        self.model = config.CAPTION_MODEL

    def _generate_one(
        self,
        platform: str,
        player_champion: str,
        enemy_names: Sequence[str],
        minimap_champions: Sequence[str],
        fight_type: str,
        fight_duration: float,
        dialog_text: str,
    ) -> dict:
        if not self.api_key:
            raise RuntimeError("OPENAI_API_KEY is not configured")
        prompt = _build_prompt(player_champion, enemy_names, minimap_champions, fight_type, fight_duration, dialog_text, platform)
        payload = {
            "model": self.model,
            "input": [
                {
                    "role": "system",
                    "content": [{"type": "input_text", "text": config.CAPTION_SYSTEM_PROMPT}],
                },
                {
                    "role": "user",
                    "content": [{"type": "input_text", "text": prompt}],
                },
            ],
            "temperature": 0.8,
            "text": {"format": {"type": "json_object"}},
        }
        with httpx.Client(timeout=config.CAPTION_TIMEOUT_SEC) as client:
            response = client.post(
                "https://api.openai.com/v1/responses",
                headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
                json=payload,
            )
            response.raise_for_status()
        content = _parse_response_text(response.json())
        return json.loads(content)

    def generate(
        self,
        player_champion: str,
        enemy_names: Sequence[str],
        fight_type: str,
        fight_duration: float,
        dialog_text: str,
        minimap_champions: Sequence[str] | None = None,
    ) -> CaptionResult:
        flags: list[str] = []
        minimap_champions = minimap_champions or []
        if not self.api_key:
            flags.append("caption_api_key_missing")
            return CaptionResult(
                {
                    "tiktok": _fallback_caption(player_champion, enemy_names, fight_type, "tiktok", minimap_champions),
                    "instagram": _fallback_caption(player_champion, enemy_names, fight_type, "instagram", minimap_champions),
                },
                flags,
            )

        results: dict[str, dict] = {}
        errors: list[str] = []

        def worker(platform: str) -> None:
            try:
                results[platform] = self._generate_one(platform, player_champion, enemy_names, minimap_champions, fight_type, fight_duration, dialog_text)
            except Exception as exc:  # noqa: BLE001
                errors.append(f"{platform}: {exc}")
                results[platform] = _fallback_caption(player_champion, enemy_names, fight_type, platform, minimap_champions)

        threads = [threading.Thread(target=worker, args=(platform,), daemon=True) for platform in ("tiktok", "instagram")]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(config.CAPTION_TIMEOUT_SEC)
        for platform, thread in zip(("tiktok", "instagram"), threads):
            if thread.is_alive() and platform not in results:
                flags.append("caption_fallback")
                results[platform] = _fallback_caption(player_champion, enemy_names, fight_type, platform, minimap_champions)
        if errors:
            flags.append("caption_fallback")
        return CaptionResult(results, flags)


def _parse_response_text(payload: dict) -> str:
    text = payload.get("output_text")
    if isinstance(text, str):
        return text
    for item in payload.get("output", []):
        for content in item.get("content", []):
            if content.get("type") in {"output_text", "text"} and isinstance(content.get("text"), str):
                return content["text"]
    raise ValueError("OpenAI response did not include output text")
