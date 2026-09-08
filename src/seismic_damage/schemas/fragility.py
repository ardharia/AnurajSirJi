"""Probabilistic fragility analysis schemas."""

from __future__ import annotations

from pydantic import BaseModel, Field

from seismic_damage.config.settings import DamageState, FragilityMethod, IntensityMeasure
from seismic_damage.schemas.building import BuildingParameters
from seismic_damage.schemas.damage_assessment import DamageAssessment


class FragilityCurve(BaseModel):
    """Lognormal (or equivalent) fragility parameters for one damage state."""

    damage_state: DamageState
    median_im: float = Field(..., gt=0, description="Median intensity measure")
    beta: float = Field(..., gt=0, description="Logarithmic standard deviation")
    probability_at_im: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="P(DS >= ds | IM) evaluated at the analysis IM",
    )


class FragilityInput(BaseModel):
    """Inputs required to run probabilistic fragility analysis."""

    building: BuildingParameters
    damage: DamageAssessment | None = None
    intensity_measure: IntensityMeasure = IntensityMeasure.PGA
    intensity_value: float = Field(..., gt=0)
    method: FragilityMethod = FragilityMethod.LOGNORMAL
    n_samples: int = Field(default=10_000, ge=100)
    random_seed: int = 42


class FragilityResult(BaseModel):
    """Output of probabilistic fragility analysis."""

    intensity_measure: IntensityMeasure
    intensity_value: float
    method: FragilityMethod
    curves: list[FragilityCurve] = Field(default_factory=list)
    exceedance_probabilities: dict[DamageState, float] = Field(default_factory=dict)
    most_likely_state: DamageState = DamageState.NONE
    expected_loss_ratio: float | None = Field(default=None, ge=0.0, le=1.0)
    metadata: dict[str, float | str | int] = Field(default_factory=dict)
