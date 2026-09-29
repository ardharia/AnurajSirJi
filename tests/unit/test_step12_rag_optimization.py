"""Unit tests for Step 12: Independent RAG Optimization."""

from __future__ import annotations

import json
from pathlib import Path
import pytest

from seismic_damage.optimization.config import (
    ExperimentResult,
    evaluate_against_ground_truth,
    experiment_results_to_rows,
    select_best_experiment,
)
from seismic_damage.optimization.rag_optimizer import (
    RAG_EXPERIMENT_MATRIX,
    run_rag_optimization,
)


def test_rag_configuration_loading() -> None:
    assert len(RAG_EXPERIMENT_MATRIX) >= 5
    baseline = RAG_EXPERIMENT_MATRIX[0]
    assert baseline["experiment_id"] == "R1_baseline"
    assert "top_k" in baseline
    assert "min_score" in baseline
    assert "chunk_size" in baseline
    assert "chunk_overlap" in baseline
    assert "prompt_variant" in baseline


def test_baseline_reproducibility() -> None:
    predictions = [
        {"building_id": "B1", "damage_grade": 4, "structural_system": "masonry"},
        {"building_id": "B2", "damage_grade": 2, "structural_system": "rc_frame"},
    ]
    ground_truth = [
        {"building_id": "B1", "damage_grade": 4, "structural_system": "masonry"},
        {"building_id": "B2", "damage_grade": 2, "structural_system": "rc_frame"},
    ]
    # Running evaluation twice must yield identical scores
    metrics_a, n_a = evaluate_against_ground_truth(predictions, ground_truth)
    metrics_b, n_b = evaluate_against_ground_truth(predictions, ground_truth)
    assert n_a == n_b == 2
    assert metrics_a == metrics_b
    assert metrics_a["damage_grade"]["exact_match"] == 1.0


def test_no_vlm_dependency_in_rag_evaluation() -> None:
    # Ground truth is the ONLY evaluation target
    rag_preds = [
        {"building_id": "B1", "damage_grade": 4},
        {"building_id": "B2", "damage_grade": 1},
    ]
    gt_records = [
        {"building_id": "B1", "damage_grade": 4},
        {"building_id": "B2", "damage_grade": 2},
    ]
    vlm_preds = [
        {"building_id": "B1", "damage_grade": 1},  # Disagrees with RAG
        {"building_id": "B2", "damage_grade": 5},  # Disagrees with RAG
    ]

    metrics, n = evaluate_against_ground_truth(rag_preds, gt_records)
    # RAG matches B1 (1 exact match out of 2)
    assert metrics["damage_grade"]["exact_match"] == 0.5
    # VLM predictions are never inspected or involved in calculating RAG metrics
    assert "vlm" not in str(metrics)


def test_optimization_result_generation_and_selection(tmp_path: Path) -> None:
    exp1 = ExperimentResult(
        experiment_id="exp_low",
        modality="rag",
        configuration={"top_k": 3},
        parameter_metrics={
            "damage_grade": {"exact_match": 0.4, "n": 10},
            "soft_story_failure": {"f1": 0.5, "n": 10},
        },
        n_buildings=10,
    )
    exp2 = ExperimentResult(
        experiment_id="exp_high",
        modality="rag",
        configuration={"top_k": 5},
        parameter_metrics={
            "damage_grade": {"exact_match": 0.8, "n": 10},
            "soft_story_failure": {"f1": 0.9, "n": 10},
        },
        n_buildings=10,
    )

    best = select_best_experiment([exp1, exp2])
    assert best.experiment_id == "exp_high"
    assert best.aggregate_score() > exp1.aggregate_score()

    rows = experiment_results_to_rows([exp1, exp2])
    assert len(rows) == 4  # 2 experiments * 2 parameters
