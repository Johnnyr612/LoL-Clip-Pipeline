from __future__ import annotations

import json
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
    minimap_champions: Sequence[str] | None = None,
) -> dict:
    clean_enemies = _clean_champion_names(enemy_names)
    minimap_context = [name for name in _clean_champion_names(minimap_champions or []) if name != player_champion]
    enemy_text = ", ".join(clean_enemies or minimap_context[:4]) or "the enemy"
    detected_text = _detected_matchup_text(player_champion, clean_enemies, minimap_context)
    if len(clean_enemies or minimap_context) == 1:
        hook = f"{player_champion} took the {(clean_enemies or minimap_context)[0]} duel"
    else:
        hook = f"{player_champion} turned this {fight_type} into a highlight"
    caption = (
        f"{hook}\n"
        f"{detected_text} made this fight feel personal. Clean spacing, clutch timing, and {enemy_text} had one chance to back off. "
        f"Would you take this fight? #leagueoflegends"
    )[:500]
    tags = ["#gaming", "#gamer", "#clips", "#fyp", "#viral", "#leagueoflegends", "#lolclips", "#riotgames", "#summonersrift", "#leagueclips"]
    return {"caption": caption, "hashtags": tags, "hook_line": hook}


def _detected_matchup_text(player_champion: str, enemy_names: Sequence[str], minimap_context: Sequence[str]) -> str:
    player = player_champion if player_champion and not player_champion.startswith("unknown") else "The player"
    opponents = _clean_champion_names(enemy_names) or [name for name in _clean_champion_names(minimap_context) if name != player]
    if not opponents:
        return f"{player}'s detected champion read"
    if len(opponents) == 1:
        return f"{player} into {opponents[0]}"
    return f"{player} into {', '.join(opponents[:4])}"


def _build_prompt(
    player_champion: str,
    enemy_names: Sequence[str],
    minimap_champions: Sequence[str],
    fight_type: str,
    fight_duration: float,
    dialog_text: str,
) -> str:
    enemies = ", ".join(_clean_champion_names(enemy_names)) or "unknown enemies"
    detected_champions = _clean_champion_names(minimap_champions)
    minimap_context = ", ".join(detected_champions) or "unknown"
    matchup = _detected_matchup_text(player_champion, enemy_names, detected_champions)
    return f"""Write a viral social media description for this LoL clip.

Clip details:
- Player champion: {player_champion}
- Enemies fought: {enemies}
- Detected champion matchup to build the description around: {matchup}
- All detected champions actively fighting or collapsing from minimap/HUD: {minimap_context}
- Fight type: {fight_type}
- Fight duration: {fight_duration:.1f} seconds
- Spoken lines detected: {dialog_text or 'none'}

Rules:
- First line: hook only, no hashtags, no generic openings
- Base the hook and body on the detected champion matchup when champion names are known
- 3-5 emojis placed naturally in text, not clustered at end
- Exactly 1 rhetorical question or call-to-action
- If any champion is 'unknown': say 'the enemy', never guess name
- Body max 500 chars, punchy, meme-aware
- Hashtag block at end: 5 gaming tags + 5 LoL-specific tags
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
        player_champion: str,
        enemy_names: Sequence[str],
        minimap_champions: Sequence[str],
        fight_type: str,
        fight_duration: float,
        dialog_text: str,
    ) -> dict:
        if not self.api_key:
            raise RuntimeError("OPENAI_API_KEY is not configured")
        prompt = _build_prompt(player_champion, enemy_names, minimap_champions, fight_type, fight_duration, dialog_text)
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
                {"default": _fallback_caption(player_champion, enemy_names, fight_type, minimap_champions)},
                flags,
            )

        try:
            description = self._generate_one(player_champion, enemy_names, minimap_champions, fight_type, fight_duration, dialog_text)
        except Exception:  # noqa: BLE001
            flags.append("caption_fallback")
            description = _fallback_caption(player_champion, enemy_names, fight_type, minimap_champions)
        return CaptionResult({"default": description}, flags)


def _parse_response_text(payload: dict) -> str:
    text = payload.get("output_text")
    if isinstance(text, str):
        return text
    for item in payload.get("output", []):
        for content in item.get("content", []):
            if content.get("type") in {"output_text", "text"} and isinstance(content.get("text"), str):
                return content["text"]
    raise ValueError("OpenAI response did not include output text")
