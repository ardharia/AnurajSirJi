"""Schemas for structured parameter extraction from multimodal sources."""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class SourceModality(str, Enum):
    RAG = "rag"
    VLM = "vlm"
    FUSED = "fused"
    MANUAL = "manual"


class ParameterValue(BaseModel):
    """A single extracted parameter with provenance and uncertainty."""

    name: str
    value: Any
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    source: SourceModality
    evidence: str | None = None
    unit: str | None = None


class ExtractionResult(BaseModel):
    """Collection of structured parameters from one or more modalities."""

    parameters: list[ParameterValue] = Field(default_factory=list)
    missing_required: list[str] = Field(default_factory=list)
    conflicts: list[str] = Field(default_factory=list)
    overall_confidence: float = Field(default=0.0, ge=0.0, le=1.0)

    def as_dict(self) -> dict[str, Any]:
        return {item.name: item.value for item in self.parameters}

    def get(self, name: str) -> ParameterValue | None:
        for item in self.parameters:
            if item.name == name:
                return item
        return None
