"""VLM observation schema helpers."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class StructuredVLMObservation(BaseModel):
    """Structured building-damage observation produced by a VLM backend."""

    damage_grade: int | None = Field(default=None, ge=0, le=5)
    material: str | None = None
    structural_system: str | None = None
    number_of_stories: int | None = Field(default=None, ge=1)
    crack_severity: str | None = None
    crack_pattern: str | None = None
    collapse_mode: str | None = None
    soft_story_failure: bool | None = None
    wall_failure: bool | None = None
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    evidence: str | None = None
    notes: str | None = None

    def as_inferred_attributes(self) -> dict[str, Any]:
        return self.model_dump()
