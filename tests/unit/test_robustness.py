"""Unit tests for Step 15: Robustness and Missing/Poor Data Analysis."""

from __future__ import annotations

from pathlib import Path
import pytest

from seismic_damage.robustness.scenarios import (
    SCENARIO_DESCRIPTIONS,
    build_scenario_image_map,
    sample_for_scenario,
)
from seismic_damage.robustness.runner import (
    _condition_vlm_records,
    evaluate_modality_predictions,
    run_full_robustness_framework,
)
from seismic_damage.validation.metrics import is_missing


def test_scenario_coverage_and_determinism() -> None:
    expected_scenarios = {
        "A_100pct", "B_75pct", "C_50pct", "D_25pct", "E_pixelated",
        "F_low_quality", "G_0pct", "H_random_removal", "I_remove_primary", "J_single_image"
    }
    assert set(SCENARIO_DESCRIPTIONS.keys()) == expected_scenarios

    images = ["img_a", "img_b", "img_c", "img_d"]
    # Deterministic behavior
    s1 = sample_for_scenario(images, "H_random_removal", seed=42)
    s2 = sample_for_scenario(images, "H_random_removal", seed=42)
    assert s1 == s2
    assert len(s1) == 3

    # Primary removal
    s_prim = sample_for_scenario(images, "I_remove_primary")
    assert s_prim == ["img_b", "img_c", "img_d"]

    # Single image
    s_single = sample_for_scenario(images, "J_single_image")
    assert s_single == ["img_a"]


def test_missing_vlm_never_zero_in_runner() -> None:
    vlm_records = [
        {"building_id": "B1", "damage_grade": 4, "soft_story_failure": True},
        {"building_id": "B2", "damage_grade": 2, "soft_story_failure": False},
    ]
    # Under Scenario G (0%), everything is missing
    img_map = {"B1": [], "B2": []}
    cond = _condition_vlm_records(vlm_records, "G_0pct", img_map)

    for rec in cond:
        assert rec["vlm_available"] is False
        assert is_missing(rec.get("damage_grade"))
        assert rec.get("damage_grade") != 0
        assert is_missing(rec.get("soft_story_failure"))
        assert rec.get("soft_story_failure") is not False


def test_evaluate_modality_predictions_coverage() -> None:
    preds = [
        {"building_id": "B1", "damage_grade": 4},
        {"building_id": "B2", "damage_grade": None},
    ]
    gt = [
        {"building_id": "B1", "damage_grade": 4},
        {"building_id": "B2", "damage_grade": 2},
    ]

    metrics = evaluate_modality_predictions(preds, gt, "test_mod", "A_100pct")
    dg_metric = next(m for m in metrics if m["parameter"] == "damage_grade")

    assert dg_metric["coverage"] == 0.5
    assert dg_metric["abstention_rate"] == 0.5
    assert dg_metric["score"] == 1.0  # Perfect on the 1 evaluated sample


def test_robustness_framework_end_to_end(tmp_path: Path) -> None:
    gt = [
        {"building_id": "B1", "damage_grade": 4, "material_type": "brick"},
        {"building_id": "B2", "damage_grade": 2, "material_type": "concrete"},
    ]
    rag = [
        {"building_id": "B1", "damage_grade": 4, "material_type": "brick"},
        {"building_id": "B2", "damage_grade": 1, "material_type": "concrete"},
    ]
    vlm = [
        {"building_id": "B1", "damage_grade": 4, "material_type": "brick"},
        {"building_id": "B2", "damage_grade": 2, "material_type": "concrete"},
    ]
    bmap = {"B1": ["img1", "img2"], "B2": ["img3"]}

    res = run_full_robustness_framework(rag, vlm, gt, bmap, output_dir=tmp_path)

    assert (tmp_path / "robustness_results.csv").exists()
    assert (tmp_path / "robustness_summary.json").exists()
    assert (tmp_path / "robustness_manifest.json").exists()
    assert (tmp_path / "plots" / "performance_vs_image_availability.png").exists()
    assert (tmp_path / "plots" / "coverage_vs_image_availability.png").exists()
