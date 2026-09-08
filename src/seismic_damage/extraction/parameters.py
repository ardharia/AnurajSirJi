"""Structured parameter extraction (stub)."""

from __future__ import annotations

from typing import Any

from seismic_damage.config import Settings, get_settings
from seismic_damage.schemas.extraction import ExtractionResult, ParameterValue, SourceModality
from seismic_damage.schemas.pipeline import RAGResult, VLMResult


def extract_parameters(
    *,
    rag_result: RAGResult | None = None,
    vlm_result: VLMResult | None = None,
    known_parameters: dict[str, Any] | None = None,
    settings: Settings | None = None,
) -> ExtractionResult:
    """Extract structured building/damage parameters from modality outputs.

    Merges optional manual hints with RAG/VLM hints. Full LLM structuring
    is intentionally not implemented yet.
    """
    cfg = settings or get_settings()
    parameters: list[ParameterValue] = []

    for name, value in (known_parameters or {}).items():
        parameters.append(
            ParameterValue(
                name=name,
                value=value,
                confidence=1.0,
                source=SourceModality.MANUAL,
                evidence="provided_by_caller",
            )
        )

    if rag_result is not None:
        for name, value in rag_result.extracted_hints.items():
            parameters.append(
                ParameterValue(
                    name=name,
                    value=value,
                    confidence=0.0,
                    source=SourceModality.RAG,
                    evidence="rag_hint",
                )
            )

    if vlm_result is not None:
        for name, value in vlm_result.extracted_hints.items():
            parameters.append(
                ParameterValue(
                    name=name,
                    value=value,
                    confidence=0.0,
                    source=SourceModality.VLM,
                    evidence="vlm_hint",
                )
            )

    present = {item.name for item in parameters}
    missing = [name for name in cfg.extraction.required_parameters if name not in present]

    return ExtractionResult(
        parameters=parameters,
        missing_required=missing,
        conflicts=[],
        overall_confidence=0.0,
    )


def fuse_extractions(results: list[ExtractionResult]) -> ExtractionResult:
    """Fuse multiple extraction results into a single structured result.

    Conflict resolution policy will be implemented later.
    """
    merged: list[ParameterValue] = []
    missing: list[str] = []
    conflicts: list[str] = []
    for result in results:
        merged.extend(result.parameters)
        missing.extend(result.missing_required)
        conflicts.extend(result.conflicts)
    return ExtractionResult(
        parameters=merged,
        missing_required=sorted(set(missing)),
        conflicts=conflicts,
        overall_confidence=0.0,
    )
