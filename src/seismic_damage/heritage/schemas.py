"""Step 19: Heritage and Ancient Monument Structural Schemas.

Modular extension:
- Heritage structures are evaluated with specialized historic typology parameters.
- Kept completely separate from the ordinary modern RC/masonry calibration dataset.
"""

from __future__ import annotations

from enum import Enum
from typing import Any
from pydantic import BaseModel, Field


class HeritageTypology(str, Enum):
    ASHLAR_STONE_TEMPLE = "ashlar_stone_temple"           # Carved sandstone historic temples (e.g., Kera, Bhuj Chhatris)
    RUBBLE_LIME_FORT_WALL = "rubble_lime_fort_wall"       # Thick defensive fort walls / bastions (e.g., Bhujia Fort)
    TIMBER_LACED_MASONRY = "timber_laced_masonry"         # Traditional vernacular havelis with wood framing
    TRADITIONAL_BHUNGA = "traditional_bhunga"             # Circular earth/wattle-and-daub Kutch vernacular
    UNREINFORCED_LIME_MASONRY = "unreinforced_lime_masonry"  # Historic palace / colonial load-bearing masonry
    HERITAGE_MONUMENT_COMPLEX = "heritage_monument_complex"
    OTHER_HERITAGE = "other_heritage"


class HeritageMasonryBond(str, Enum):
    ASHLAR_FINE = "ashlar_fine"
    COURSED_RUBBLE_LIME = "coursed_rubble_lime"
    RANDOM_RUBBLE_MUD = "random_rubble_mud"
    MULTI_LEAF_STONE = "multi_leaf_stone"
    UNKNOWN = "unknown"


class HeritageRoofVaultSystem(str, Enum):
    STONE_CORBELLED_DOME = "stone_corbelled_dome"
    TRUE_MASONRY_ARCH_VAULT = "true_masonry_arch_vault"
    TIMBER_BEAM_TILE = "timber_beam_tile"
    FLAT_STONE_LINTELS = "flat_stone_lintels"
    UNKNOWN = "unknown"


class HeritageVulnerabilityAttributes(BaseModel):
    """Heritage-specific vulnerability modifiers and decay indicators."""

    multi_leaf_wall_delamination_risk: bool | None = None
    unrestrained_arch_thrust: bool | None = None
    heavy_dome_pounding: bool | None = None
    timber_rot_or_termite_decay: bool | None = None
    mortar_leaching_or_loss: bool | None = None
    out_of_plumb_walls: bool | None = None
    incompatible_modern_interventions: bool | None = None  # e.g., heavy rigid concrete slab on soft stone walls
    previous_earthquake_cracking: bool | None = None


class HeritageBuildingRecord(BaseModel):
    """Canonical assessment record for a heritage / ancient monument asset."""

    monument_id: str
    monument_name: str
    location: str | None = "Kutch_Gujarat"
    heritage_typology: HeritageTypology
    masonry_bond: HeritageMasonryBond = HeritageMasonryBond.UNKNOWN
    roof_vault_system: HeritageRoofVaultSystem = HeritageRoofVaultSystem.UNKNOWN
    wall_thickness_m: float | None = Field(default=None, gt=0.0)
    height_m: float | None = Field(default=None, gt=0.0)
    number_of_stories: int | None = Field(default=1, ge=1)
    construction_era: str | None = None  # e.g., "18th_century", "Solanki_dynasty"
    vulnerability_attributes: HeritageVulnerabilityAttributes = Field(
        default_factory=HeritageVulnerabilityAttributes
    )

    heritage_vulnerability_index: float = Field(
        default=0.5, ge=0.0, le=1.0, description="Heritage-calibrated vulnerability index [0=least, 1=most vulnerable]"
    )
    heritage_vulnerability_class: str = Field(
        default="Class_A_Heritage", description="Specialized heritage vulnerability class"
    )
    conservation_priority: str = Field(
        default="High", description="Low | Moderate | High | Critical"
    )
    preservation_guidelines: list[str] = Field(default_factory=list)
    source_reference: str = Field(
        default="IS_13935_Heritage_Guidelines_ASI_Kutch_Survey"
    )

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump()
