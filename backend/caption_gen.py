from __future__ import annotations

import json
import threading
from dataclasses import dataclass
from typing import Sequence

from . import config


@dataclass(frozen=True)
class CaptionResult:
    captions: dict[str, dict]
    flags: list[str]


def _fallback_caption(player_champion: str, enemy_names: Sequence[str], fight_type: str, platform: str) -> dict:
    enemy_text = ", ".join(name for name in enemy_names if not name.startswith("unknown")) or "the enemy"
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
    fight_type: str,
    fight_duration: float,
    dialog_text: str,
    platform: str,
) -> str:
    enemies = ", ".join(enemy_names) or "unknown enemies"
    return f"""Write a viral social media caption for this LoL clip.

Clip details:
- Player champion: {player_champion}
- Enemies fought: {enemies}
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

Respond with valid JSON only, no markdown, no preamble:
{{"caption": "...", "hashtags": ["..."], "hook_line": "..."}}"""


class CaptionGenerator:
    def __init__(self) -> None:
        self.model_path = config.LLAMA_MODEL_PATH

    def _generate_one(self, platform: str, player_champion: str, enemy_names: Sequence[str], fight_type: str, fight_duration: float, dialog_text: str) -> dict:
        if not self.model_path.exists():
            raise FileNotFoundError(self.model_path)
        from llama_cpp import Llama

        llm = Llama(model_path=str(self.model_path), n_gpu_layers=0, verbose=False)
        prompt = _build_prompt(player_champion, enemy_names, fight_type, fight_duration, dialog_text, platform)
        response = llm.create_chat_completion(
            messages=[
                {"role": "system", "content": config.CAPTION_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=0.8,
        )
        content = response["choices"][0]["message"]["content"]
        return json.loads(content)

    def generate(self, player_champion: str, enemy_names: Sequence[str], fight_type: str, fight_duration: float, dialog_text: str) -> CaptionResult:
        flags: list[str] = []
        if not self.model_path.exists():
            flags.append("caption_model_unavailable")
            return CaptionResult(
                {
                    "tiktok": _fallback_caption(player_champion, enemy_names, fight_type, "tiktok"),
                    "instagram": _fallback_caption(player_champion, enemy_names, fight_type, "instagram"),
                },
                flags,
            )

        results: dict[str, dict] = {}
        errors: list[str] = []

        def worker(platform: str) -> None:
            try:
                results[platform] = self._generate_one(platform, player_champion, enemy_names, fight_type, fight_duration, dialog_text)
            except Exception as exc:  # noqa: BLE001
                errors.append(f"{platform}: {exc}")
                results[platform] = _fallback_caption(player_champion, enemy_names, fight_type, platform)

        threads = [threading.Thread(target=worker, args=(platform,), daemon=True) for platform in ("tiktok", "instagram")]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(config.CAPTION_TIMEOUT_SEC)
        for platform, thread in zip(("tiktok", "instagram"), threads):
            if thread.is_alive() and platform not in results:
                flags.append("caption_fallback")
                results[platform] = _fallback_caption(player_champion, enemy_names, fight_type, platform)
        if errors:
            flags.append("caption_fallback")
        return CaptionResult(results, flags)
