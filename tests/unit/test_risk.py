"""Unit tests for Step 18: Integrated Seismic Risk Framework."""

from __future__ import annotations

from pathlib import Path
import pytest

from seismic_damage.risk.assessment import (
    assess_all_buildings_risk,
    assess_building_risk,
)
from seismic_damage.risk.exposure import build_exposure_record
from seismic_damage.risk.hazard import SeismicHazardRecord
from seismic_damage.risk.reporting import export_risk_pipeline_artifacts
from seismic_damage.risk.site_effects import calculate_site_effects


def test_hazard_site_vulnerability_separation() -> None:
    hazard = SeismicHazardRecord(regional_pga_g=0.40)
    site = calculate_site_effects("BHJ_001", "soft")
    # Soft soil should amplify ground motion by 1.67
    assert site.soil_amplification_factor == 1.67

    building_rec = {
        "building_id": "BHJ_001",
        "material_type": "burnt_clay_brick",
        "structural_system": "bearing_wall",
        "occupancy": "hospital",  # Lifeline occupancy -> higher consequence
        "soil_type": "soft",
    }

    risk = assess_building_risk("BHJ_001", building_rec, hazard)
    assert risk.building_id == "BHJ_001"
    # Effective PGA = 0.40 * 1.67 = 0.668 g
    assert abs(risk.effective_ground_motion_pga_g - 0.668) < 1e-3
    assert risk.exposure.relative_exposure_weight == 2.0  # Lifeline
    assert risk.relative_risk_score > 0.0
    assert risk.relative_risk_category in ("High", "Very_High", "Severe")


def test_missing_hazard_values() -> None:
    # Hazard with None PGA defaults to zone factor Z
    hazard = SeismicHazardRecord(regional_pga_g=None, zone_factor_z=0.36)
    building_rec = {
        "building_id": "BHJ_002",
        "material_type": "reinforced_concrete",
        "number_of_stories": 3,
        "soil_type": "rock",
    }

    risk = assess_building_risk("BHJ_002", building_rec, hazard)
    assert risk.effective_ground_motion_pga_g == 0.36  # 0.36 * 1.0 (rock)
    assert risk.relative_risk_score > 0.0


def test_export_risk_pipeline_artifacts(tmp_path: Path) -> None:
    records = [
        {"building_id": "B1", "material_type": "brick", "occupancy": "residential"},
        {"building_id": "B2", "material_type": "concrete", "occupancy": "school"},
    ]
    hazard = SeismicHazardRecord()
    risk_records = assess_all_buildings_risk(records, hazard)

    res = export_risk_pipeline_artifacts(risk_records, hazard, output_dir=tmp_path)

    assert (tmp_path / "risk_assessment.csv").exists()
    assert (tmp_path / "hazard_summary.csv").exists()
    assert (tmp_path / "vulnerability_summary.csv").exists()
    assert (tmp_path / "risk_manifest.json").exists()
    assert res["n_buildings"] == 2
