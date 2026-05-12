from __future__ import annotations

from backend.minimap_detector import ChampionResult, FightParticipants
from backend.pipeline import _apply_vision_participants
from backend.vision_classifier import VisionFightResult


def test_vision_participants_preserve_trusted_hud_player():
    participants = FightParticipants(
        ChampionResult("Aatrox", 0.7, "ally", (0.4, 0.4), True),
        [],
        [ChampionResult("Kayn", 0.5, "enemy", (0.5, 0.4))],
        "1v1",
        [],
    )
    vision = VisionFightResult("Irelia", ["Kayn"], "1v1", 0.9)

    result = _apply_vision_participants(participants, vision, "Aatrox")

    assert result.player.champion_name == "Aatrox"
    assert [enemy.champion_name for enemy in result.enemies] == ["Kayn"]
    assert "vision_player_override_ignored" in result.flags
