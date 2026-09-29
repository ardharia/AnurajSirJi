"""Unit tests for Step 11: VLM Robustness Experiments."""

from __future__ import annotations

import pytest
from pathlib import Path

from seismic_damage.robustness.sampling import (
    AVAILABILITY_LEVELS,
    RANDOM_SEED,
    build_availability_sets,
    describe_availability_set,
    deterministic_sample,
)
from seismic_damage.robustness.degradation import (
    degrade_image_blur_low_contrast,
    degrade_image_pixelate,
    describe_degradation_condition,
)
from seismic_damage.robustness.evaluation import (
    aggregate_condition_metrics,
    build_robustness_summary,
    evaluate_condition,
)
from seismic_damage.robustness.pipeline import (
    _apply_degradation_proxy,
    _stub_vlm_for_condition,
    run_availability_experiments,
    run_degradation_experiments,
    run_robustness_pipeline,
)
from seismic_damage.validation.metrics import is_missing


def test_deterministic_image_sampling() -> None:
    images = [f"img_{i:03d}" for i in range(10)]
    # Repeated calls with same seed must return identical results
    sample_a = deterministic_sample(images, 0.5, seed=RANDOM_SEED)
    sample_b = deterministic_sample(images, 0.5, seed=RANDOM_SEED)
    assert sample_a == sample_b
    assert len(sample_a) == 5

    # Changing seed should produce valid deterministic subset
    sample_c = deterministic_sample(images, 0.5, seed=999)
    assert len(sample_c) == 5


def test_availability_levels_and_fractions() -> None:
    images = [f"img_{i:03d}" for i in range(12)]
    
    # 100% -> all images
    s100 = deterministic_sample(images, AVAILABILITY_LEVELS["A_100pct"])
    assert s100 == images
    assert len(s100) == 12

    # 75% -> 9 images
    s75 = deterministic_sample(images, AVAILABILITY_LEVELS["B_75pct"])
    assert len(s75) == 9
    assert set(s75).issubset(set(images))

    # 50% -> 6 images
    s50 = deterministic_sample(images, AVAILABILITY_LEVELS["C_50pct"])
    assert len(s50) == 6
    assert set(s50).issubset(set(images))

    # 25% -> 3 images
    s25 = deterministic_sample(images, AVAILABILITY_LEVELS["D_25pct"])
    assert len(s25) == 3
    assert set(s25).issubset(set(images))

    # 0% -> 0 images
    s0 = deterministic_sample(images, AVAILABILITY_LEVELS["G_0pct"])
    assert s0 == []


def test_missing_vlm_handling_never_zero() -> None:
    full_vlm = [
        {"building_id": "BHJ_001", "damage_grade": 4, "soft_story_failure": True},
        {"building_id": "BHJ_002", "damage_grade": 2, "soft_story_failure": False},
    ]
    # Building 1 has images, building 2 has none
    available_map = {"BHJ_001": ["img_001"], "BHJ_002": []}
    stubbed = _stub_vlm_for_condition(full_vlm, available_map)

    rec1 = next(r for r in stubbed if r["building_id"] == "BHJ_001")
    rec2 = next(r for r in stubbed if r["building_id"] == "BHJ_002")

    assert rec1["damage_grade"] == 4
    # Missing VLM evidence MUST be None, never 0 or false
    assert rec2.get("damage_grade") is None
    assert rec2.get("damage_grade") != 0
    assert rec2.get("soft_story_failure") is None
    assert rec2.get("soft_story_failure") is not False


def test_building_identity_preservation() -> None:
    bids = ["BHJ_001", "BHJ_002", "BHJ_003"]
    gt_records = [
        {"building_id": bid, "damage_grade": 3, "material_type": "reinforced_concrete"}
        for bid in bids
    ]
    vlm_records = [
        {"building_id": bid, "damage_grade": 3, "material_type": "reinforced_concrete"}
        for bid in bids
    ]
    meta = [
        {"building_id": bid, "image_count_total": 2, "image_count_available": 1, "availability_fraction": 0.5}
        for bid in bids
    ]

    rows = evaluate_condition("C_50pct", vlm_records, gt_records, availability_meta=meta)
    
    evaluated_bids = {r["building_id"] for r in rows}
    assert evaluated_bids == set(bids)

    for row in rows:
        assert "building_id" in row
        assert "condition" in row
        assert "parameter" in row
        assert "vlm_prediction" in row
        assert "ground_truth" in row
        assert "availability_fraction" in row
        assert "image_count_available" in row
        assert "image_count_total" in row


def test_degradation_handling_proxy() -> None:
    vlm_records = [
        {"building_id": "BHJ_001", "damage_grade": 4, "material_type": "brick", "parameters": {"damage_grade": 4, "material_type": "brick"}}
    ]
    bmap = {"BHJ_001": ["img_001"]}

    # Severe degradation (proxy: 100% missing)
    deg_e = _apply_degradation_proxy(vlm_records, bmap, missing_rate=1.0)
    assert deg_e[0]["building_id"] == "BHJ_001"
    assert is_missing(deg_e[0].get("damage_grade"))
    assert is_missing(deg_e[0].get("material_type"))

    # Mild degradation (proxy: 50% missing)
    deg_f = _apply_degradation_proxy(vlm_records, bmap, missing_rate=0.5)
    assert deg_f[0]["building_id"] == "BHJ_001"


def test_robustness_pipeline_end_to_end(tmp_path: Path) -> None:
    gt_records = [
        {"building_id": "B1", "damage_grade": 3, "soft_story_failure": 0, "material_type": "brick"},
        {"building_id": "B2", "damage_grade": 1, "soft_story_failure": 0, "material_type": "concrete"},
    ]
    vlm_records = [
        {"building_id": "B1", "damage_grade": 3, "soft_story_failure": 0, "material_type": "brick"},
        {"building_id": "B2", "damage_grade": 2, "soft_story_failure": 1, "material_type": "concrete"},
    ]
    bmap = {
        "B1": ["img_01", "img_02"],
        "B2": ["img_03"],
    }

    result = run_robustness_pipeline(
        vlm_records=vlm_records,
        ground_truth_records=gt_records,
        building_image_map=bmap,
        output_dir=tmp_path,
    )

    assert result["n_availability_rows"] > 0
    assert result["n_degradation_rows"] > 0
    assert (tmp_path / "vlm_availability_results.csv").exists()
    assert (tmp_path / "vlm_degradation_results.csv").exists()
    assert (tmp_path / "vlm_robustness_summary.csv").exists()
    assert (tmp_path / "robustness_manifest.json").exists()
