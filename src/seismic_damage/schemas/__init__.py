"""Pydantic schemas for structured seismic assessment data."""

from seismic_damage.schemas.building import BuildingParameters, LateralSystem, Material
from seismic_damage.schemas.damage import ObservedDamage
from seismic_damage.schemas.damage_assessment import (
    CollapseMode,
    CrackPattern,
    CrackSeverity,
    DamageAssessment,
    SoilType,
)
from seismic_damage.schemas.extraction import (
    ExtractionResult,
    ParameterValue,
    SourceModality,
)
from seismic_damage.schemas.fragility import (
    FragilityCurve,
    FragilityInput,
    FragilityResult,
)
from seismic_damage.schemas.normalized import (
    DamageAttributes,
    NormalizedAssessmentRecord,
    ProvenanceInfo,
    TypologyAttributes,
    VulnerabilityAttributes,
)
from seismic_damage.schemas.pipeline import (
    AssessmentRequest,
    AssessmentResult,
    RAGDocument,
    RAGResult,
    VLMObservation,
    VLMResult,
)

__all__ = [
    "AssessmentRequest",
    "AssessmentResult",
    "BuildingParameters",
    "CollapseMode",
    "CrackPattern",
    "CrackSeverity",
    "DamageAssessment",
    "DamageAttributes",
    "ExtractionResult",
    "FragilityCurve",
    "FragilityInput",
    "FragilityResult",
    "LateralSystem",
    "Material",
    "NormalizedAssessmentRecord",
    "ObservedDamage",
    "ParameterValue",
    "ProvenanceInfo",
    "RAGDocument",
    "RAGResult",
    "SoilType",
    "SourceModality",
    "TypologyAttributes",
    "VLMObservation",
    "VLMResult",
    "VulnerabilityAttributes",
]
