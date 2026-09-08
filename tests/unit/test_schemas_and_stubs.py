"""Smoke checks for schema construction and stub pipelines."""

from __future__ import annotations

from seismic_damage.extraction import extract_parameters
from seismic_damage.pipelines.rag import run_rag
from seismic_damage.pipelines.vlm import run_vlm
from seismic_damage.schemas.building import BuildingParameters, LateralSystem, Material
from seismic_damage.schemas.fragility import FragilityInput
from seismic_damage.fragility import analyze_fragility
from seismic_damage.config import IntensityMeasure


def test_building_parameters_schema() -> None:
    building = BuildingParameters(
        building_type="RC3",
        number_of_stories=3,
        year_built=1985,
        lateral_system=LateralSystem.MOMENT_FRAME,
        material=Material.REINFORCED_CONCRETE,
    )
    assert building.building_type == "rc3"


def test_independent_pipeline_stubs() -> None:
    rag = run_rag("soft story fragility")
    assert rag.query == "soft story fragility"
    vlm = run_vlm([])
    assert vlm.extracted_hints.get("n_images") == 0


def test_extraction_and_fragility_stubs() -> None:
    extraction = extract_parameters(known_parameters={"building_type": "rc3"})
    assert extraction.get("building_type") is not None

    building = BuildingParameters(
        building_type="rc3",
        number_of_stories=3,
        year_built=1990,
    )
    result = analyze_fragility(
        FragilityInput(
            building=building,
            intensity_measure=IntensityMeasure.PGA,
            intensity_value=0.3,
        )
    )
    assert result.metadata.get("status") == "not_implemented"
