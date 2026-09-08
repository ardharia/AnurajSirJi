"""Probabilistic fragility analysis (stub)."""

from __future__ import annotations

from seismic_damage.config import DamageState, get_settings
from seismic_damage.schemas.fragility import FragilityCurve, FragilityInput, FragilityResult


def analyze_fragility(payload: FragilityInput) -> FragilityResult:
    """Estimate damage-state exceedance probabilities for a given IM.

    Log-normal fragility evaluation will be implemented later. This stub
    returns empty curves with metadata describing the planned method.
    """
    settings = get_settings()
    method = payload.method or settings.fragility.method

    curves = [
        FragilityCurve(damage_state=state, median_im=1.0, beta=0.5, probability_at_im=None)
        for state in settings.fragility.damage_states
        if state is not DamageState.NONE
    ]

    return FragilityResult(
        intensity_measure=payload.intensity_measure,
        intensity_value=payload.intensity_value,
        method=method,
        curves=curves,
        exceedance_probabilities={},
        most_likely_state=DamageState.NONE,
        expected_loss_ratio=None,
        metadata={
            "status": "not_implemented",
            "n_samples": payload.n_samples,
            "random_seed": payload.random_seed,
        },
    )
