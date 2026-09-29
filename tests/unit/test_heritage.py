"""Unit tests for Step 19: Heritage and Ancient Monument Extension."""

from __future__ import annotations

import pytest

from seismic_damage.heritage.assessment import assess_heritage_monument
from seismic_damage.heritage.mapping import map_text_to_heritage_typology
from seismic_damage.heritage.schemas import (
    HeritageBuildingRecord,
    HeritageMasonryBond,
    HeritageRoofVaultSystem,
    HeritageTypology,
)


def test_heritage_typology_mapping() -> None:
    assert map_text_to_heritage_typology("Ancient Shiva Temple with sandstone Shikhara") == HeritageTypology.ASHLAR_STONE_TEMPLE
    assert map_text_to_heritage_typology("Historic Bhunga hut in Hodka village") == HeritageTypology.TRADITIONAL_BHUNGA
    assert map_text_to_heritage_typology("Old Bhujia Fort stone wall and bastion") == HeritageTypology.RUBBLE_LIME_FORT_WALL


def test_traditional_bhunga_vs_stone_temple_vulnerability() -> None:
    bhunga = {
        "monument_id": "BHU_01",
        "name": "Kutch Vernacular Bhunga",
        "heritage_typology": "traditional_bhunga",
    }
    temple = {
        "monument_id": "TEM_01",
        "name": "Kera Shiva Temple",
        "heritage_typology": "ashlar_stone_temple",
        "vulnerability_attributes": {
            "unrestrained_arch_thrust": True,
            "multi_leaf_wall_delamination_risk": True,
        },
    }

    res_bhunga = assess_heritage_monument("BHU_01", bhunga)
    res_temple = assess_heritage_monument("TEM_01", temple)

    # Traditional Bhunga is circular and lightweight -> significantly lower vulnerability index
    assert res_bhunga.heritage_vulnerability_index < 0.40
    assert res_bhunga.conservation_priority == "Low"

    # Ashlar Temple with thrust and delamination has high vulnerability
    assert res_temple.heritage_vulnerability_index > 0.70
    assert res_temple.conservation_priority in ("High", "Critical")
    assert any("hydraulic lime grout" in g for g in res_temple.preservation_guidelines)
    assert any("tie-rods" in g for g in res_temple.preservation_guidelines)


def test_heritage_record_validation() -> None:
    rec = {
        "monument_id": "HAV_01",
        "name": "Old Bhuj Haveli",
        "heritage_typology": "timber_laced_masonry",
        "masonry_bond": "coursed_rubble_lime",
        "roof_vault_system": "timber_beam_tile",
        "number_of_stories": 2,
    }
    result = assess_heritage_monument("HAV_01", rec)
    assert isinstance(result, HeritageBuildingRecord)
    assert result.masonry_bond == HeritageMasonryBond.COURSED_RUBBLE_LIME
    assert result.roof_vault_system == HeritageRoofVaultSystem.TIMBER_BEAM_TILE
    assert result.source_reference is not None
