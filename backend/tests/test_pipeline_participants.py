from __future__ import annotations

from backend.main import ProcessingSettingsRequest
from backend.minimap_detector import ChampionResult, FightParticipants
from backend.pipeline import _apply_vision_participants, _participants_without_minimap
from backend.vision_classifier import VisionFightResult


def test_vision_participants_use_vision_when_hud_confidence_is_not_decisive():
    participants = FightParticipants(
        ChampionResult("Aatrox", 0.7, "ally", (0.4, 0.4), True),
        [],
        [ChampionResult("Kayn", 0.5, "enemy", (0.5, 0.4))],
        "1v1",
        [],
    )
    vision = VisionFightResult("Irelia", ["Kayn"], "1v1", 0.9)

    result = _apply_vision_participants(participants, vision, "Aatrox", 0.70)

    assert result.player.champion_name == "Irelia"
    assert [enemy.champion_name for enemy in result.enemies] == ["Kayn"]
    assert "hud_player_disagreed_with_vision" in result.flags


def test_vision_participants_preserve_very_confident_hud_player():
    participants = FightParticipants(
        ChampionResult("Aatrox", 0.95, "ally", (0.4, 0.4), True),
        [],
        [ChampionResult("Kayn", 0.5, "enemy", (0.5, 0.4))],
        "1v1",
        [],
    )
    vision = VisionFightResult("Irelia", ["Kayn"], "1v1", 0.9)

    result = _apply_vision_participants(participants, vision, "Aatrox", 0.95)

    assert result.player.champion_name == "Aatrox"
    assert "vision_player_override_ignored" in result.flags


def test_skip_minimap_participants_use_hud_player_and_healthbar_enemy_count():
    result = _participants_without_minimap("Ahri", 0.72, 3)

    assert result.player.champion_name == "Ahri"
    assert result.player.is_player is True
    assert result.fight_type == "1v3"
    assert len(result.enemies) == 3
    assert "minimap_detection_skipped" in result.flags
    assert "enemy_count_from_healthbars" in result.flags


def test_processing_settings_request_accepts_skip_minimap_flag():
    settings = ProcessingSettingsRequest(skip_minimap_detection=True).to_processing_settings()

    assert settings.skip_minimap_detection is True
