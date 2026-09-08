"""Unit tests for the MVP DamageAssessment schema."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from seismic_damage.schemas.damage_assessment import (
    CollapseMode,
    CrackPattern,
    CrackSeverity,
    DamageAssessment,
    SoilType,
)


def _required_kwargs(**overrides: object) -> dict[str, object]:
    base: dict[str, object] = {
        "building_id": "B-001",
        "earthquake_event": "EQ-2024-001",
        "confidence": 0.85,
        "evidence_source": "manual",
    }
    base.update(overrides)
    return base


def test_valid_damage_assessment_creation() -> None:
    assessment = DamageAssessment(
        **_required_kwargs(
            structural_system="moment_frame",
            material_type="reinforced_concrete",
            number_of_stories=4,
            damage_grade=3,
            crack_severity=CrackSeverity.MODERATE,
            crack_pattern=CrackPattern.SHEAR,
            collapse_mode=CollapseMode.NONE,
            soft_story_failure=False,
            wall_failure=True,
            pga=0.35,
            magnitude=6.2,
            soil_type=SoilType.STIFF,
            evidence_text="Visual inspection notes",
        )
    )

    assert assessment.building_id == "B-001"
    assert assessment.earthquake_event == "EQ-2024-001"
    assert assessment.damage_grade == 3
    assert assessment.crack_severity == CrackSeverity.MODERATE
    assert assessment.confidence == 0.85
    assert assessment.pga == 0.35
    assert assessment.soil_type == SoilType.STIFF


def test_invalid_damage_grade() -> None:
    with pytest.raises(ValidationError):
        DamageAssessment(**_required_kwargs(damage_grade=6))

    with pytest.raises(ValidationError):
        DamageAssessment(**_required_kwargs(damage_grade=-1))


def test_invalid_confidence_below_zero() -> None:
    with pytest.raises(ValidationError):
        DamageAssessment(**_required_kwargs(confidence=-0.01))


def test_invalid_confidence_above_one() -> None:
    with pytest.raises(ValidationError):
        DamageAssessment(**_required_kwargs(confidence=1.01))


def test_missing_optional_fields_should_work() -> None:
    assessment = DamageAssessment(**_required_kwargs())

    assert assessment.structural_system is None
    assert assessment.material_type is None
    assert assessment.number_of_stories is None
    assert assessment.damage_grade is None
    assert assessment.crack_severity is None
    assert assessment.crack_pattern is None
    assert assessment.collapse_mode is None
    assert assessment.soft_story_failure is None
    assert assessment.wall_failure is None
    assert assessment.pga is None
    assert assessment.magnitude is None
    assert assessment.soil_type is None
    assert assessment.evidence_text is None


def test_negative_pga_should_fail_validation() -> None:
    with pytest.raises(ValidationError):
        DamageAssessment(**_required_kwargs(pga=-0.1))
