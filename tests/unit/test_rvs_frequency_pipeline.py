"""Unit tests for NDMA RVS Primer parameter extraction, building registry, and frequency analysis."""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from seismic_damage.config.settings import PROJECT_ROOT
from seismic_damage.extraction.frequency_pipeline import (
    load_rvs_dictionary,
    match_additional_parameters,
    match_rvs_parameters,
    run_parameter_frequency_analysis,
)
from seismic_damage.linking.registry import build_building_registry, classify_typology


def test_rvs_parameter_dictionary_structure() -> None:
    """Check that RVS dictionary contains required typologies and fields."""
    params = load_rvs_dictionary()
    assert len(params) >= 300

    typologies = {p["typology"] for p in params}
    required = {"RC", "BCBM", "CM", "RRSM", "MH", "DD", "EH"}
    assert required.issubset(typologies)

    sample = params[0]
    for field in ["typology", "parameter_id", "parameter_category", "parameter_text", "source_document", "source_page"]:
        assert field in sample
        assert sample[field]


def test_building_registry_structure_and_deduplication() -> None:
    """Verify building registry has unique stable IDs, typologies, and evidence."""
    registry = build_building_registry()
    assert len(registry) > 0

    ids = [b["building_id"] for b in registry]
    assert len(ids) == len(set(ids)), "Building IDs must be uniquely assigned"

    for b in registry:
        assert b["building_id"].startswith("BHJ_B")
        assert b["event_id"] == "BHUJ_2001"
        assert b["building_typology"] in {"RC", "BCBM", "CM", "RRSM", "MH", "DD", "EH", "UNKNOWN", "NON_BUILDING"}
        assert "evidence_text" in b


def test_typology_classification() -> None:
    """Verify evidence-based typology classification mapping."""
    assert classify_typology("4-storey reinforced concrete frame building")[0] == "RC"
    assert classify_typology("load bearing burnt clay brick masonry wall")[0] == "BCBM"
    assert classify_typology("random rubble stone masonry house with mud mortar")[0] == "RRSM"
    assert classify_typology("confined masonry with tie columns and tie beams")[0] == "CM"
    assert classify_typology("traditional mud house with mud plaster")[0] == "MH"
    assert classify_typology("dhajji diwari timber frame structure")[0] == "DD"
    assert classify_typology("ekra house with bamboo panels")[0] == "EH"
    assert classify_typology("some non-specific building text without structural cues")[0] == "UNKNOWN"


def test_typology_isolation_in_rvs_matching() -> None:
    """Verify that a building only matches parameters applicable to its own typology."""
    rc_building = {
        "building_id": "TEST_RC_01",
        "event_id": "BHUJ_2001",
        "building_typology": "RC",
        "evidence_text": "open ground storey with slender columns",
        "building_description": "commercial RC building",
        "source_document": "test_doc",
        "page_number": "1",
    }
    all_dict = load_rvs_dictionary()
    rc_params = [p for p in all_dict if p["typology"] == "RC"]
    rrsm_params = [p for p in all_dict if p["typology"] == "RRSM"]

    rc_matches = match_rvs_parameters(rc_building, rc_params)
    assert any(m["rvs_parameter_id"] in ["RC_RVS_3.4", "RC_POST_3.3"] for m in rc_matches)

    # RRSM parameters should not match or be evaluated against RC building
    rrsm_matches = match_rvs_parameters(rc_building, rrsm_params)
    assert len(rrsm_matches) == 0


def test_building_level_frequency_correctness() -> None:
    """Ensure frequency equals unique buildings exhibiting the parameter, not mentions."""
    obs, freqs = run_parameter_frequency_analysis()
    assert len(obs) > 0
    assert len(freqs) > 0

    # Test duplicate protection: each (building_id, parameter_id) is recorded at most once
    pairs = [(o["building_id"], o["rvs_parameter_id"]) for o in obs]
    assert len(pairs) == len(set(pairs))

    # Test frequency computation matches unique buildings
    for f in freqs:
        matching = {
            o["building_id"]
            for o in obs
            if o["rvs_parameter_id"] == f["parameter_id"] and o["typology"] == f["typology"]
        }
        assert len(matching) == f["frequency"]
        assert f["parameter_source"] in {"RVS_PRIMER", "ADDITIONAL_EARTHQUAKE_OBSERVED"}


def test_traceability_of_observations() -> None:
    """Ensure every observation traces to building_id, source document, page, and evidence."""
    obs, _ = run_parameter_frequency_analysis()
    for o in obs:
        assert o["building_id"]
        assert o["source_document"]
        assert o["page_number"]
        assert o["evidence_text"]
        assert o["match_confidence"]
