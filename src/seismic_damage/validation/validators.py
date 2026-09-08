"""Parameter and cross-modal validation (stub)."""

from __future__ import annotations

from pydantic import BaseModel, Field

from seismic_damage.config import Settings, get_settings
from seismic_damage.schemas.building import BuildingParameters
from seismic_damage.schemas.extraction import ExtractionResult


class ValidationReport(BaseModel):
    """Outcome of validation checks prior to fragility analysis."""

    passed: bool
    messages: list[str] = Field(default_factory=list)
    building: BuildingParameters | None = None


def validate_extraction(
    extraction: ExtractionResult,
    settings: Settings | None = None,
) -> ValidationReport:
    """Validate extracted parameters against required fields and confidence.

    Full schema coercion and cross-RAG/VLM checks are deferred.
    """
    cfg = settings or get_settings()
    messages: list[str] = []

    if extraction.missing_required:
        messages.append(
            f"Missing required parameters: {', '.join(extraction.missing_required)}"
        )

    if extraction.overall_confidence < cfg.validation.min_confidence:
        messages.append(
            "Overall confidence "
            f"{extraction.overall_confidence:.2f} below minimum "
            f"{cfg.validation.min_confidence:.2f}"
        )

    if extraction.conflicts:
        messages.append(f"Parameter conflicts detected: {len(extraction.conflicts)}")

    passed = not messages if cfg.validation.strict else True
    if not cfg.validation.strict and messages:
        # Non-strict mode reports issues but still passes.
        passed = True

    return ValidationReport(passed=passed, messages=messages, building=None)
