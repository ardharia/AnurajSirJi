"""Common normalized schema for building-level assessment comparison.

Enables RAG, VLM, and Ground Truth outputs to be compared at the
building + parameter level without fusing predictions. Preserves original
outputs and maintains strict separation between pre-earthquake vulnerability,
post-earthquake damage, and building typology.
"""

from __future__ import annotations

from typing import Any, Mapping
import numpy as np
from pydantic import BaseModel, Field

# Core parameter set evaluated in cross-validation
CORE_PARAMETERS: frozenset[str] = frozenset(
    {
        "damage_grade",
        "soft_story_failure",
        "wall_failure",
        "number_of_stories",
        "pga",
        "magnitude",
        "structural_system",
        "material_type",
        "crack_severity",
        "crack_pattern",
        "collapse_mode",
        "soil_type",
    }
)


def _check_missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str) and not value.strip():
        return True
    if isinstance(value, float) and np.isnan(value):
        return True
    return False


class TypologyAttributes(BaseModel):
    """Pre-earthquake building typology attributes."""

    material_type: str | None = None
    structural_system: str | None = None
    number_of_stories: int | None = Field(default=None, ge=1)
    occupancy: str | None = None
    ndma_typology: str | None = None
    year_built: int | None = None
    floor_type: str | None = None
    roof_geometry: str | None = None
    roof_material: str | None = None


class VulnerabilityAttributes(BaseModel):
    """Pre-earthquake vulnerability and RVS parameters.

    NOTE: Post-earthquake damage observations must NEVER be converted
    into pre-earthquake vulnerability parameters.
    """

    soft_story: bool | None = None
    floating_columns: bool | None = None
    plan_irregularity: bool | None = None
    vertical_irregularity: bool | None = None
    short_columns: bool | None = None
    large_overhangs: bool | None = None
    pounding_risk: bool | None = None
    slope_hazard: bool | None = None
    liquefaction_risk: bool | None = None
    soil_type: str | None = None
    seismic_zone: str | None = None
    rvs_tag: str | None = None
    ems98_vulnerability_class: str | None = None


class DamageAttributes(BaseModel):
    """Post-earthquake physical damage observations."""

    damage_grade: int | None = Field(default=None, ge=0, le=5)
    structural_damage_level: str | None = None
    non_structural_damage_level: str | None = None
    crack_severity: str | None = None
    crack_pattern: str | None = None
    crack_width: float | None = None
    collapse_mode: str | None = None
    soft_story_failure: bool | None = None
    wall_failure: bool | None = None
    wall_separation: bool | None = None
    floor_roof_failure: bool | None = None
    rebar_failure: bool | None = None
    concrete_spalling: bool | None = None
    foundation_settlement: bool | None = None
    ground_failure: bool | None = None
    post_eq_rvs_tag: str | None = None
    retrofitting_action: str | None = None
    pga: float | None = Field(default=None, ge=0.0)
    magnitude: float | None = Field(default=None, ge=0.0)


class ProvenanceInfo(BaseModel):
    """Provenance and source documentation tracking."""

    source_modality: str
    source_collection: str | None = None
    document_ids: list[str] = Field(default_factory=list)
    image_ids: list[str] = Field(default_factory=list)
    image_paths: list[str] = Field(default_factory=list)
    source_urls: list[str] = Field(default_factory=list)
    citations: list[dict[str, Any]] = Field(default_factory=list)
    evidence_text: str | None = None
    notes: str | None = None


class NormalizedAssessmentRecord(BaseModel):
    """Canonical normalized assessment record for one building and one modality.

    Designed for cross-source comparison (GT vs RAG vs VLM) at the
    building + parameter level. Predictions from different modalities
    are never fused in this schema.
    """

    building_id: str
    earthquake_event: str = "Bhuj_2001"
    modality: str = Field(..., description="'ground_truth' | 'rag' | 'vlm'")
    typology: TypologyAttributes = Field(default_factory=TypologyAttributes)
    vulnerability: VulnerabilityAttributes = Field(default_factory=VulnerabilityAttributes)
    damage: DamageAttributes = Field(default_factory=DamageAttributes)
    provenance: ProvenanceInfo = Field(default_factory=lambda: ProvenanceInfo(source_modality="unknown"))
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    raw_output: dict[str, Any] = Field(default_factory=dict)

    @property
    def parameters(self) -> dict[str, Any]:
        """Flat dictionary of all populated evaluation parameters."""
        flat: dict[str, Any] = {}
        for block in (self.typology, self.vulnerability, self.damage):
            for k, v in block.model_dump().items():
                if not _check_missing(v) and k in CORE_PARAMETERS:
                    flat[k] = v
        return flat

    @property
    def missing_parameters(self) -> list[str]:
        """List of standard parameters that are missing/unstated for this building."""
        present = set(self.parameters.keys())
        return sorted(CORE_PARAMETERS - present)

    def to_dict(self) -> dict[str, Any]:
        """Export dictionary compatible with cross_validate and compare_sources."""
        payload: dict[str, Any] = {
            "building_id": self.building_id,
            "earthquake_event": self.earthquake_event,
            "evidence_source": self.modality,
            "confidence": self.confidence,
            "evidence_text": self.provenance.evidence_text,
            "parameters": self.parameters,
            "missing_parameters": self.missing_parameters,
            "typology": self.typology.model_dump(),
            "vulnerability": self.vulnerability.model_dump(),
            "damage": self.damage.model_dump(),
            "provenance": self.provenance.model_dump(),
        }
        # Populate top-level fields for direct access in compare_sources
        payload.update(self.parameters)
        return payload
