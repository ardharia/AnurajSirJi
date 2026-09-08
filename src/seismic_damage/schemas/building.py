"""Building taxonomy and structural parameter schemas."""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field, field_validator


class Material(str, Enum):
    REINFORCED_CONCRETE = "reinforced_concrete"
    STEEL = "steel"
    MASONRY = "masonry"
    WOOD = "wood"
    MIXED = "mixed"
    UNKNOWN = "unknown"


class LateralSystem(str, Enum):
    MOMENT_FRAME = "moment_frame"
    SHEAR_WALL = "shear_wall"
    BRACED_FRAME = "braced_frame"
    DUAL_SYSTEM = "dual_system"
    BEARING_WALL = "bearing_wall"
    UNKNOWN = "unknown"


class Occupancy(str, Enum):
    RESIDENTIAL = "residential"
    COMMERCIAL = "commercial"
    INDUSTRIAL = "industrial"
    EDUCATIONAL = "educational"
    HEALTHCARE = "healthcare"
    MIXED_USE = "mixed_use"
    UNKNOWN = "unknown"


class BuildingParameters(BaseModel):
    """Structured building attributes used for fragility analysis."""

    building_id: str | None = None
    building_type: str = Field(..., description="Taxonomy / typology label")
    number_of_stories: int = Field(..., ge=1)
    year_built: int = Field(..., ge=1800, le=2100)
    lateral_system: LateralSystem = LateralSystem.UNKNOWN
    material: Material = Material.UNKNOWN
    floor_area_m2: float | None = Field(default=None, gt=0)
    occupancy: Occupancy = Occupancy.UNKNOWN
    soft_story: bool | None = None
    retrofitted: bool | None = None
    latitude: float | None = Field(default=None, ge=-90, le=90)
    longitude: float | None = Field(default=None, ge=-180, le=180)
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)

    @field_validator("building_type")
    @classmethod
    def _normalize_type(cls, value: str) -> str:
        return value.strip().lower()
