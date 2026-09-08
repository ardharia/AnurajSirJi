"""MVP damage assessment schema for structured seismic observations."""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class CrackSeverity(str, Enum):
    """Categorical crack severity labels."""

    NONE = "none"
    HAIRLINE = "hairline"
    MODERATE = "moderate"
    SEVERE = "severe"
    EXTENSIVE = "extensive"


class CrackPattern(str, Enum):
    """Common seismic crack pattern categories."""

    NONE = "none"
    VERTICAL = "vertical"
    HORIZONTAL = "horizontal"
    DIAGONAL = "diagonal"
    X_SHAPED = "x_shaped"
    STAIR_STEP = "stair_step"
    FLEXURAL = "flexural"
    SHEAR = "shear"
    UNKNOWN = "unknown"


class CollapseMode(str, Enum):
    """Observed or inferred collapse mechanism."""

    NONE = "none"
    PARTIAL = "partial"
    TOTAL = "total"
    SOFT_STORY = "soft_story"
    OUT_OF_PLANE = "out_of_plane"
    IN_PLANE = "in_plane"
    UNKNOWN = "unknown"


class SoilType(str, Enum):
    """Simplified site soil classification."""

    ROCK = "rock"
    STIFF = "stiff"
    MEDIUM = "medium"
    SOFT = "soft"
    UNKNOWN = "unknown"


class DamageAssessment(BaseModel):
    """Structured MVP damage assessment for a building / earthquake event."""

    # Required identifiers
    building_id: str
    earthquake_event: str

    # Building
    structural_system: str | None = None
    material_type: str | None = None
    number_of_stories: int | None = Field(default=None, ge=1)

    # Damage
    damage_grade: int | None = Field(default=None, ge=0, le=5)
    crack_severity: CrackSeverity | None = None
    crack_pattern: CrackPattern | None = None
    collapse_mode: CollapseMode | None = None

    # Failure
    soft_story_failure: bool | None = None
    wall_failure: bool | None = None

    # Seismic
    pga: float | None = Field(default=None, ge=0.0)
    magnitude: float | None = Field(default=None, ge=0.0)

    # Location
    soil_type: SoilType | None = None

    # Output
    confidence: float = Field(..., ge=0.0, le=1.0)
    evidence_source: str
    evidence_text: str | None = None
