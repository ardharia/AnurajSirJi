"""Master runner script for Steps 15 through 19.

Executes:
  Step 15: Full Robustness Framework (Scenarios A through J)
  Step 16: Final Ablation Study (Experiments A through J)
  Step 17 & 18: Pre-Earthquake Vulnerability & Integrated Seismic Risk Assessment
  Step 19: Heritage Monument Assessment
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from seismic_damage.ablation.runner import run_full_ablation_framework
from seismic_damage.config.settings import PROJECT_ROOT
from seismic_damage.heritage.assessment import assess_heritage_monument
from seismic_damage.linking.catalog import build_catalog, load_image_rows
from seismic_damage.risk.assessment import assess_all_buildings_risk
from seismic_damage.risk.hazard import SeismicHazardRecord
from seismic_damage.risk.reporting import export_risk_pipeline_artifacts
from seismic_damage.robustness.runner import run_full_robustness_framework


def load_csv_records(path: Path) -> list[dict[str, Any]]:
    """Load records from CSV into a list of dicts."""
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader)


def run_all() -> dict[str, Any]:
    print("=== Loading Catalog and Reference Datasets for Steps 15–19 ===")
    image_rows = load_image_rows()
    catalog = build_catalog(image_rows)
    building_image_map = {b.building_id: b.image_ids for b in catalog}

    gt_path = PROJECT_ROOT / "data" / "processed" / "reference" / "building_ground_truth.csv"
    rag_path = PROJECT_ROOT / "data" / "processed" / "results" / "rag_building_summary.csv"
    vlm_path = PROJECT_ROOT / "data" / "processed" / "results" / "vlm_building_summary.csv"

    gt_records = load_csv_records(gt_path)
    rag_records = load_csv_records(rag_path)
    vlm_records = load_csv_records(vlm_path)

    print(f"Loaded {len(gt_records)} GT, {len(rag_records)} RAG, {len(vlm_records)} VLM records.")

    # ---------------------------------------------------------
    # STEP 15: Robustness Analysis (Scenarios A through J)
    # ---------------------------------------------------------
    print("\n=== Running Step 15: Full Robustness Framework ===")
    step15_res = run_full_robustness_framework(
        rag_records=rag_records,
        vlm_records=vlm_records,
        ground_truth_records=gt_records,
        building_image_map=building_image_map,
    )
    print(f"Step 15 complete: {step15_res['n_rows']} evaluation metric rows across 10 scenarios.")

    # ---------------------------------------------------------
    # STEP 16: Final Ablation Study (Experiments A through J)
    # ---------------------------------------------------------
    print("\n=== Running Step 16: Final Ablation Study ===")
    step16_res = run_full_ablation_framework(
        rag_records=rag_records,
        vlm_records=vlm_records,
        ground_truth_records=gt_records,
    )
    print(f"Step 16 complete: {step16_res['n_experiments']} ablation experiments executed.")

    # ---------------------------------------------------------
    # STEP 17 & 18: Integrated Seismic Risk Framework
    # ---------------------------------------------------------
    print("\n=== Running Step 17 & 18: Vulnerability & Integrated Risk Framework ===")
    hazard = SeismicHazardRecord(regional_pga_g=0.38, seismic_zone="Zone_V", zone_factor_z=0.36)
    risk_records = assess_all_buildings_risk(gt_records, hazard)
    step18_res = export_risk_pipeline_artifacts(risk_records, hazard)
    print(f"Step 18 complete: {step18_res['n_buildings']} building risk profiles evaluated.")

    # ---------------------------------------------------------
    # STEP 19: Heritage Monument Extension Sample Run
    # ---------------------------------------------------------
    print("\n=== Running Step 19: Heritage Monument Assessment ===")
    sample_heritage = [
        {
            "monument_id": "ASI_BHJ_01",
            "name": "Kera Shiva Temple (10th Century)",
            "heritage_typology": "ashlar_stone_temple",
            "masonry_bond": "ashlar_fine",
            "roof_vault_system": "stone_corbelled_dome",
            "wall_thickness_m": 1.2,
            "height_m": 14.0,
            "construction_era": "Solanki_dynasty",
            "vulnerability_attributes": {
                "unrestrained_arch_thrust": True,
                "multi_leaf_wall_delamination_risk": True,
                "mortar_leaching_or_loss": True,
            },
        },
        {
            "monument_id": "ASI_BHJ_02",
            "name": "Bhujia Fort Bastion",
            "heritage_typology": "rubble_lime_fort_wall",
            "masonry_bond": "multi_leaf_stone",
            "roof_vault_system": "flat_stone_lintels",
            "wall_thickness_m": 2.5,
            "height_m": 8.0,
            "construction_era": "18th_century",
            "vulnerability_attributes": {
                "multi_leaf_wall_delamination_risk": True,
                "out_of_plumb_walls": True,
            },
        },
        {
            "monument_id": "VERN_KUTCH_01",
            "name": "Hodka Vernacular Bhunga",
            "heritage_typology": "traditional_bhunga",
            "masonry_bond": "random_rubble_mud",
            "roof_vault_system": "timber_beam_tile",
            "wall_thickness_m": 0.45,
            "height_m": 3.2,
            "construction_era": "Traditional_Vernacular",
            "vulnerability_attributes": {},
        },
    ]
    heritage_results = [assess_heritage_monument(m["monument_id"], m) for m in sample_heritage]
    print(f"Step 19 complete: Evaluated {len(heritage_results)} landmark heritage structures.")
    for h in heritage_results:
        print(f"  - {h.monument_name}: Index={h.heritage_vulnerability_index:.2f}, Class={h.heritage_vulnerability_class}, Priority={h.conservation_priority}")

    return {
        "step15": step15_res,
        "step16": step16_res,
        "step18": step18_res,
        "heritage_count": len(heritage_results),
    }


if __name__ == "__main__":
    run_all()
