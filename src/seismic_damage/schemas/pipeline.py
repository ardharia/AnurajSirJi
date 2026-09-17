"""Request/response schemas for independent RAG and VLM pipelines."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, field_validator

from seismic_damage.schemas.building import BuildingParameters
from seismic_damage.schemas.damage import ObservedDamage
from seismic_damage.schemas.damage_assessment import DamageAssessment
from seismic_damage.schemas.extraction import ExtractionResult
from seismic_damage.schemas.fragility import FragilityResult
from seismic_damage.schemas.rag import RAGCitation


class RAGDocument(BaseModel):
    """Retrieved knowledge chunk."""

    document_id: str
    content: str
    score: float = Field(default=0.0, ge=0.0)
    metadata: dict[str, Any] = Field(default_factory=dict)
    source_path: str | None = None


class RAGResult(BaseModel):
    """Output of the RAG pipeline (independent of VLM)."""

    query: str
    documents: list[RAGDocument] = Field(default_factory=list)
    synthesized_context: str | None = None
    extracted_hints: dict[str, Any] = Field(default_factory=dict)
    answer: str | None = None
    citations: list[RAGCitation] = Field(default_factory=list)
    evidence_sufficient: bool = False


class VLMObservation(BaseModel):
    """Structured visual finding from a single image or frame."""

    image_path: str
    caption: str | None = None
    damage: ObservedDamage | None = None
    inferred_attributes: dict[str, Any] = Field(default_factory=dict)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)


class VLMResult(BaseModel):
    """Output of the VLM pipeline (independent of RAG)."""

    observations: list[VLMObservation] = Field(default_factory=list)
    summary: str | None = None
    extracted_hints: dict[str, Any] = Field(default_factory=dict)


class AssessmentRequest(BaseModel):
    """Top-level multimodal assessment request."""

    building_id: str | None = None
    image_paths: list[Path] = Field(default_factory=list)
    text_query: str | None = None
    intensity_value: float | None = Field(default=None, gt=0)
    known_parameters: dict[str, Any] = Field(default_factory=dict)
    run_rag: bool = True
    run_vlm: bool = True

    @field_validator("image_paths", mode="before")
    @classmethod
    def _coerce_paths(cls, value: Any) -> list[Path]:
        if value is None:
            return []
        if isinstance(value, (str, Path)):
            return [Path(value)]
        return [Path(item) for item in value]


class AssessmentResult(BaseModel):
    """End-to-end assessment artifact (pipelines remain independently runnable)."""

    request_id: str
    rag: RAGResult | None = None
    vlm: VLMResult | None = None
    extraction: ExtractionResult | None = None
    building: BuildingParameters | None = None
    damage: DamageAssessment | None = None
    fragility: FragilityResult | None = None
    validation_passed: bool = False
    validation_messages: list[str] = Field(default_factory=list)
