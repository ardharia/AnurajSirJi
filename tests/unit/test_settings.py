"""Unit tests for configuration loading."""

from __future__ import annotations

from pathlib import Path

from seismic_damage.config import FragilityMethod, load_settings, reset_settings_cache


def test_load_default_settings() -> None:
    reset_settings_cache()
    settings = load_settings()
    assert settings.app.name == "seismic-damage-assessment"
    assert settings.rag.top_k >= 1
    assert settings.fragility.method == FragilityMethod.LOGNORMAL
    assert "building_type" in settings.extraction.required_parameters


def test_load_settings_from_explicit_path() -> None:
    reset_settings_cache()
    path = Path("config") / "defaults.yaml"
    settings = load_settings(path)
    assert settings.vlm.model
    assert settings.paths.knowledge_dir.name == "knowledge"
