"""Step 18: Local Site and Topographic Amplification Layer.

Separated cleanly from regional hazard and building vulnerability.
"""

from __future__ import annotations

from typing import Any
from pydantic import BaseModel, Field


class SiteEffectsRecord(BaseModel):
    """Local geological and topographic influence parameters (IS 1893:2016)."""

    building_id: str
    soil_type: str = Field(default="medium", description="Soil classification (rock | stiff | medium | soft)")
    soil_amplification_factor: float = Field(default=1.36, ge=1.0, le=2.5, description="Site ground motion amplification")
    topographic_factor: float = Field(default=1.0, ge=1.0, le=1.5, description="Topographic / slope amplification")
    liquefaction_susceptibility: str = Field(default="low", description="none | low | moderate | high")

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump()


def calculate_site_effects(building_id: str, soil_type_str: str | None = None) -> SiteEffectsRecord:
    """Derive site amplification factors from local soil classification."""
    s_clean = str(soil_type_str or "medium").strip().lower()

    if "rock" in s_clean or "hard" in s_clean:
        soil_type = "rock"
        amplification = 1.00
        liq = "none"
    elif "soft" in s_clean or "loose" in s_clean or "clay" in s_clean:
        soil_type = "soft"
        amplification = 1.67
        liq = "moderate"
    else:
        soil_type = "medium"
        amplification = 1.36
        liq = "low"

    return SiteEffectsRecord(
        building_id=building_id,
        soil_type=soil_type,
        soil_amplification_factor=amplification,
        topographic_factor=1.0,
        liquefaction_susceptibility=liq,
    )
