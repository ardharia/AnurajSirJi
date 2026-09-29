"""Master runner script for Steps 11–14.

Executes:
  Step 11: VLM Missing/Poor Robustness Experiments
  Step 12: Independent RAG Optimization
  Step 13: Independent VLM Optimization
  Step 14: Calibrated RAG + VLM Fusion
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from seismic_damage.config.settings import PROJECT_ROOT
from seismic_damage.fusion.evaluation import run_fusion_pipeline
from seismic_damage.io_utils import read_jsonl
from seismic_damage.linking.catalog import build_catalog, load_image_rows
from seismic_damage.optimization.rag_optimizer import run_rag_optimization
from seismic_damage.optimization.vlm_optimizer import run_vlm_optimization
from seismic_damage.robustness.pipeline import run_robustness_pipeline


def load_csv_records(path: Path) -> list[dict[str, Any]]:
    """Load records from CSV into a list of dicts."""
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader)


def run_all() -> dict[str, Any]:
    print("=== Loading Catalog and Reference Datasets ===")
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
    # STEP 11: Robustness Experiments
    # ---------------------------------------------------------
    print("\n=== Running Step 11: VLM Robustness Experiments ===")
    step11_res = run_robustness_pipeline(
        vlm_records=vlm_records,
        ground_truth_records=gt_records,
        building_image_map=building_image_map,
    )
    print(f"Step 11 complete: {step11_res['n_availability_rows']} availability rows, {step11_res['n_degradation_rows']} degradation rows.")

    # ---------------------------------------------------------
    # STEP 12: Independent RAG Optimization
    # ---------------------------------------------------------
    print("\n=== Running Step 12: Independent RAG Optimization ===")
    building_queries = [
        (
            b.building_id,
            f"What is the damage grade, structural system, material type, and crack severity for building {b.building_id}? "
            f"Are there soft story failures or wall failures?",
        )
        for b in catalog
    ]
    step12_res = run_rag_optimization(
        building_queries=building_queries,
        ground_truth_records=gt_records,
    )
    print(f"Step 12 complete: Best experiment = {step12_res['best_experiment_id']} (score = {step12_res['best_aggregate_score']:.4f})")

    # ---------------------------------------------------------
    # STEP 13: Independent VLM Optimization
    # ---------------------------------------------------------
    print("\n=== Running Step 13: Independent VLM Optimization ===")
    step13_res = run_vlm_optimization(
        vlm_records=vlm_records,
        ground_truth_records=gt_records,
        image_level_records=image_rows,
    )
    print(f"Step 13 complete: Best experiment = {step13_res['best_experiment_id']} (score = {step13_res['best_aggregate_score']:.4f})")

    # ---------------------------------------------------------
    # STEP 14: Calibrated RAG + VLM Fusion
    # ---------------------------------------------------------
    print("\n=== Running Step 14: Calibrated RAG + VLM Fusion ===")
    step14_res = run_fusion_pipeline(
        rag_records=rag_records,
        vlm_records=vlm_records,
        ground_truth_records=gt_records,
    )
    print(f"Step 14 complete: {len(step14_res['comparison'])} parameter comparisons generated.")

    return {
        "step11": step11_res,
        "step12": step12_res,
        "step13": step13_res,
        "step14": step14_res,
    }


if __name__ == "__main__":
    run_all()
