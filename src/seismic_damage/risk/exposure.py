"""Step 18: Building Exposure Representation.

Captures building occupancy, economic/social importance, and exposure weighting.
"""

from __future__ import annotations

from enum import Enum
from typing import Any
from pydantic import BaseModel, Field


class OccupancyCategory(str, Enum):
    RESIDENTIAL = "residential"
    COMMERCIAL = "commercial"
    EDUCATIONAL = "educational"
    LIFELINE_HEALTHCARE = "lifeline"
    INDUSTRIAL = "industrial"
    UNKNOWN = "unknown"


# Relative social/economic exposure weighting by occupancy (higher = higher consequence of failure)
OCCUPANCY_EXPOSURE_WEIGHTS: dict[str, float] = {
    "residential": 1.0,
    "commercial": 1.2,
    "educational": 1.5,
    "lifeline": 2.0,      # Hospitals, dispensaries, emergency services
    "industrial": 1.3,
    "unknown": 1.0,
}


class BuildingExposureRecord(BaseModel):
    """Building exposure and consequence attributes."""

    building_id: str
    location_name: str | None = None
    occupancy: OccupancyCategory = OccupancyCategory.UNKNOWN
    number_of_stories: int | None = Field(default=None, ge=1)
    estimated_occupancy_load: int | None = Field(default=None, ge=0)
    relative_exposure_weight: float = Field(default=1.0, ge=0.5, le=3.0)

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump()


def build_exposure_record(
    building_id: str,
    occupancy_str: str | None = None,
    stories: int | None = None,
) -> BuildingExposureRecord:
    """Instantiate building exposure record from metadata."""
    occ_clean = str(occupancy_str or "").strip().lower()
    cat = OccupancyCategory.UNKNOWN

    for k in ("lifeline", "hospital", "clinic"):
        if k in occ_clean:
            cat = OccupancyCategory.LIFELINE_HEALTHCARE
            break
    if cat == OccupancyCategory.UNKNOWN:
        for k in ("educational", "school", "college", "institute"):
            if k in occ_clean:
                cat = OccupancyCategory.EDUCATIONAL
                break
    if cat == OccupancyCategory.UNKNOWN:
        for k in ("commercial", "office", "shop"):
            if k in occ_clean:
                cat = OccupancyCategory.COMMERCIAL
                break
    if cat == OccupancyCategory.UNKNOWN:
        for k in ("residential", "housing", "apartment"):
            if k in occ_clean:
                cat = OccupancyCategory.RESIDENTIAL
                break
    if cat == OccupancyCategory.UNKNOWN:
        if "industrial" in occ_clean:
            cat = OccupancyCategory.INDUSTRIAL

    try:
        n_stories = int(stories) if stories is not None and str(stories).strip() else None
    except (ValueError, TypeError):
        n_stories = None

    weight = OCCUPANCY_EXPOSURE_WEIGHTS.get(cat.value, 1.0)
    return BuildingExposureRecord(
        building_id=building_id,
        occupancy=cat,
        number_of_stories=n_stories,
        relative_exposure_weight=weight,
    )
