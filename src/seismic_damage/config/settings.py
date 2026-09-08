"""Application configuration via YAML defaults + environment overrides."""

from __future__ import annotations

from enum import Enum
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# Project root: src/seismic_damage/config -> parents[3]
_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_CONFIG_PATH = _PROJECT_ROOT / "config" / "defaults.yaml"


class FragilityMethod(str, Enum):
    LOGNORMAL = "lognormal"
    EMPIRICAL = "empirical"
    BAYESIAN = "bayesian"


class IntensityMeasure(str, Enum):
    PGA = "PGA"
    SA = "Sa"
    PGV = "PGV"


class DamageState(str, Enum):
    NONE = "none"
    SLIGHT = "slight"
    MODERATE = "moderate"
    EXTENSIVE = "extensive"
    COMPLETE = "complete"


class AppConfig(BaseModel):
    name: str = "seismic-damage-assessment"
    version: str = "0.1.0"
    env: str = "development"
    log_level: str = "INFO"


class PathsConfig(BaseModel):
    data_dir: Path = Path("data")
    knowledge_dir: Path = Path("data/knowledge")
    raw_dir: Path = Path("data/raw")
    processed_dir: Path = Path("data/processed")


class RAGConfig(BaseModel):
    enabled: bool = True
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    vector_store_path: Path = Path("data/processed/vector_store")
    top_k: int = Field(default=5, ge=1)
    chunk_size: int = Field(default=512, ge=64)
    chunk_overlap: int = Field(default=64, ge=0)
    collection_name: str = "seismic_knowledge"


class VLMConfig(BaseModel):
    enabled: bool = True
    provider: str = "openai"
    model: str = "gpt-4o"
    api_key: str | None = None
    base_url: str | None = None
    max_tokens: int = Field(default=2048, ge=1)
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    supported_formats: list[str] = Field(
        default_factory=lambda: ["jpg", "jpeg", "png", "webp", "tiff"]
    )


class LLMConfig(BaseModel):
    provider: str = "openai"
    model: str = "gpt-4o"
    api_key: str | None = None
    base_url: str | None = None
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    max_tokens: int = Field(default=4096, ge=1)


class ExtractionConfig(BaseModel):
    required_parameters: list[str] = Field(
        default_factory=lambda: [
            "building_type",
            "number_of_stories",
            "year_built",
            "lateral_system",
            "material",
        ]
    )
    optional_parameters: list[str] = Field(
        default_factory=lambda: [
            "floor_area_m2",
            "occupancy",
            "soft_story",
            "retrofitted",
            "observed_damage_state",
        ]
    )


class ValidationConfig(BaseModel):
    strict: bool = True
    min_confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    cross_check_rag_vlm: bool = True


class FragilityConfig(BaseModel):
    method: FragilityMethod = FragilityMethod.LOGNORMAL
    n_samples: int = Field(default=10_000, ge=100)
    random_seed: int = 42
    damage_states: list[DamageState] = Field(
        default_factory=lambda: list(DamageState)
    )
    intensity_measure: IntensityMeasure = IntensityMeasure.PGA
    intensity_unit: str = "g"


class Settings(BaseSettings):
    """Root settings: YAML defaults merged with environment / .env overrides."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app: AppConfig = Field(default_factory=AppConfig)
    paths: PathsConfig = Field(default_factory=PathsConfig)
    rag: RAGConfig = Field(default_factory=RAGConfig)
    vlm: VLMConfig = Field(default_factory=VLMConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    extraction: ExtractionConfig = Field(default_factory=ExtractionConfig)
    validation: ValidationConfig = Field(default_factory=ValidationConfig)
    fragility: FragilityConfig = Field(default_factory=FragilityConfig)

    app_env: str | None = None
    log_level: str | None = None
    vlm_api_key: str | None = None
    llm_api_key: str | None = None
    vlm_base_url: str | None = None
    llm_base_url: str | None = None


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config file must be a mapping: {path}")
    return data


def load_settings(config_path: str | Path | None = None) -> Settings:
    """Load settings from YAML, then apply pydantic-settings / .env overrides."""
    path = Path(config_path) if config_path else _DEFAULT_CONFIG_PATH
    yaml_data = _load_yaml(path)
    settings = Settings(**yaml_data)

    updates: dict[str, Any] = {}
    if settings.app_env:
        updates["app"] = settings.app.model_copy(update={"env": settings.app_env})
    if settings.log_level:
        app_cfg = updates.get("app", settings.app)
        assert isinstance(app_cfg, AppConfig)
        updates["app"] = app_cfg.model_copy(update={"log_level": settings.log_level})
    if settings.vlm_api_key:
        updates["vlm"] = settings.vlm.model_copy(update={"api_key": settings.vlm_api_key})
    if settings.vlm_base_url:
        vlm_cfg = updates.get("vlm", settings.vlm)
        assert isinstance(vlm_cfg, VLMConfig)
        updates["vlm"] = vlm_cfg.model_copy(update={"base_url": settings.vlm_base_url})
    if settings.llm_api_key:
        updates["llm"] = settings.llm.model_copy(update={"api_key": settings.llm_api_key})
    if settings.llm_base_url:
        llm_cfg = updates.get("llm", settings.llm)
        assert isinstance(llm_cfg, LLMConfig)
        updates["llm"] = llm_cfg.model_copy(update={"base_url": settings.llm_base_url})

    if updates:
        settings = settings.model_copy(update=updates)
    return settings


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Cached singleton settings instance for application use."""
    return load_settings()


def reset_settings_cache() -> None:
    """Clear the settings cache (useful in tests)."""
    get_settings.cache_clear()
