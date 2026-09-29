"""Step 18: Earthquake Hazard Representation.

Captures regional seismological demand without conflating it with local site effects
or building vulnerability.
"""

from __future__ import annotations

from typing import Any
from pydantic import BaseModel, Field


class SeismicHazardRecord(BaseModel):
    """Regional seismic hazard parameters (Bhuj 2001 / IS 1893:2016)."""

    earthquake_event: str = "Bhuj_2001"
    magnitude_mw: float | None = Field(default=7.7, description="Moment magnitude")
    epicentral_distance_km: float | None = Field(default=None, ge=0.0)
    hypocentral_depth_km: float | None = Field(default=25.0, ge=0.0)
    seismic_zone: str = Field(default="Zone_V", description="IS 1893:2016 Seismic Zone")
    zone_factor_z: float = Field(default=0.36, description="Zone factor Z (0.36 for Zone V)")
    regional_pga_g: float | None = Field(default=0.38, ge=0.0, description="Estimated regional bed-rock PGA in g")
    macroseismic_intensity: str | None = Field(default="X", description="MSK-64 / EMS-98 intensity")
    importance_factor_i: float = Field(default=1.0, ge=1.0, le=2.0)

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump()
