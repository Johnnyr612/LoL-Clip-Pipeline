from __future__ import annotations

from backend.caption_gen import CaptionGenerator, _build_prompt


def test_caption_prompt_includes_original_minimap_context():
    prompt = _build_prompt(
        "Aatrox",
        ["Ahri", "unknown_champion_1"],
        ["Aatrox", "Ahri", "Thresh", "Ahri"],
        "1v2",
        12.5,
        "",
        "tiktok",
    )

    assert "Champions actively fighting or collapsing on the fight from original minimap/HUD: Aatrox, Ahri, Thresh" in prompt
    assert "Enemies fought: Ahri" in prompt


def test_fallback_caption_uses_minimap_context_when_enemies_unknown(monkeypatch):
    generator = CaptionGenerator()
    monkeypatch.setattr(generator, "model_path", type("MissingPath", (), {"exists": lambda self: False})())

    result = generator.generate(
        "Aatrox",
        ["unknown_champion_1"],
        "1v1",
        8.0,
        "",
        ["Aatrox", "Ahri", "Thresh"],
    )

    assert "Ahri, Thresh" in result.captions["tiktok"]["caption"]
