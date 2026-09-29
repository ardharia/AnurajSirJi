"""Unit tests for Step 13: Independent VLM Optimization."""

from __future__ import annotations

import json
from pathlib import Path
import pytest

from seismic_damage.optimization.config import (
    ExperimentResult,
    evaluate_against_ground_truth,
    select_best_experiment,
)
from seismic_damage.optimization.vlm_optimizer import (
    VLM_EXPERIMENT_MATRIX,
    _apply_confidence_filter,
    _apply_image_selection,
    _apply_majority_vote,
    run_vlm_optimization,
)


def test_vlm_configuration_loading() -> None:
    assert len(VLM_EXPERIMENT_MATRIX) >= 5
    baseline = VLM_EXPERIMENT_MATRIX[0]
    assert baseline["experiment_id"] == "V1_baseline"
    assert "confidence_threshold" in baseline
    assert "image_selection" in baseline
    assert "aggregation" in baseline


def test_vlm_baseline_reproducibility() -> None:
    vlm_records = [
        {"building_id": "B1", "damage_grade": 4, "confidence": 0.8},
        {"building_id": "B2", "damage_grade": 2, "confidence": 0.9},
    ]
    gt_records = [
        {"building_id": "B1", "damage_grade": 4},
        {"building_id": "B2", "damage_grade": 2},
    ]
    metrics_1, n1 = evaluate_against_ground_truth(vlm_records, gt_records)
    metrics_2, n2 = evaluate_against_ground_truth(vlm_records, gt_records)
    assert n1 == n2 == 2
    assert metrics_1 == metrics_2
    assert metrics_1["damage_grade"]["exact_match"] == 1.0


def test_no_rag_dependency_in_vlm_optimization() -> None:
    vlm_records = [
        {"building_id": "B1", "damage_grade": 3, "confidence": 0.7},
    ]
    gt_records = [
        {"building_id": "B1", "damage_grade": 3},
    ]
    # Evaluate strictly against Ground Truth
    metrics, n = evaluate_against_ground_truth(vlm_records, gt_records)
    assert metrics["damage_grade"]["exact_match"] == 1.0
    # No RAG artifacts or records are referenced
    assert "rag" not in str(metrics)


def test_vlm_confidence_filter() -> None:
    records = [
        {"building_id": "B1", "damage_grade": 3, "confidence": 0.9},
        {"building_id": "B2", "damage_grade": 2, "confidence": 0.3},
    ]
    filtered = _apply_confidence_filter(records, threshold=0.5)
    rec1 = next(r for r in filtered if r["building_id"] == "B1")
    rec2 = next(r for r in filtered if r["building_id"] == "B2")

    assert rec1["damage_grade"] == 3
    # Low confidence -> parameters should be absent/missing, NEVER zero
    assert "damage_grade" not in rec2 or rec2["damage_grade"] is None
    assert rec2.get("damage_grade") != 0


def test_vlm_majority_vote_aggregation() -> None:
    vlm_records = [{"building_id": "B1"}]
    image_observations = [
        {"building_id": "B1", "image_id": "img1", "damage_grade": 4},
        {"building_id": "B1", "image_id": "img2", "damage_grade": 4},
        {"building_id": "B1", "image_id": "img3", "damage_grade": 2},
    ]
    aggregated = _apply_majority_vote(vlm_records, image_observations)
    assert len(aggregated) == 1
    # Majority of [4, 4, 2] is 4
    assert aggregated[0]["damage_grade"] == 4


def test_vlm_optimization_run(tmp_path: Path) -> None:
    gt_records = [
        {"building_id": "B1", "damage_grade": 4, "soft_story_failure": 1},
        {"building_id": "B2", "damage_grade": 2, "soft_story_failure": 0},
    ]
    vlm_records = [
        {"building_id": "B1", "damage_grade": 4, "soft_story_failure": 1, "confidence": 0.8},
        {"building_id": "B2", "damage_grade": 3, "soft_story_failure": 0, "confidence": 0.4},
    ]

    res = run_vlm_optimization(vlm_records, gt_records, output_dir=tmp_path)
    assert res["n_experiments"] >= 5
    assert (tmp_path / "baseline" / "baseline_results.csv").exists()
    assert (tmp_path / "experiment_results.csv").exists()
    assert (tmp_path / "best_configuration.json").exists()
    assert (tmp_path / "optimization_manifest.json").exists()
