"""Unit tests for Step 16: Final Ablation Study."""

from __future__ import annotations

from pathlib import Path
import pytest

from seismic_damage.ablation.experiments import ABLATION_EXPERIMENTS
from seismic_damage.ablation.runner import (
    run_full_ablation_framework,
    run_single_ablation_experiment,
)


def test_ablation_experiment_matrix_completeness() -> None:
    assert len(ABLATION_EXPERIMENTS) == 10
    exp_ids = [e.experiment_id for e in ABLATION_EXPERIMENTS]
    assert exp_ids == [f"Exp_{c}" for c in "ABCDEFGHIJ"]


def test_single_ablation_experiment_execution() -> None:
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
    weights = {
        "damage_grade": {"rag_weight": 0.5, "vlm_weight": 0.5},
        "material_type": {"rag_weight": 0.5, "vlm_weight": 0.5},
    }

    # Run Exp A (RAG only)
    preds_a, metrics_a = run_single_ablation_experiment(
        ABLATION_EXPERIMENTS[0], rag, vlm, gt, weights
    )
    assert len(preds_a) == 2
    assert len(metrics_a) > 0

    # Run Exp B (VLM only)
    preds_b, metrics_b = run_single_ablation_experiment(
        ABLATION_EXPERIMENTS[1], rag, vlm, gt, weights
    )
    assert len(preds_b) == 2


def test_ablation_framework_end_to_end(tmp_path: Path) -> None:
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

    res = run_full_ablation_framework(rag, vlm, gt, output_dir=tmp_path)

    assert (tmp_path / "ablation_results.csv").exists()
    assert (tmp_path / "ablation_summary.json").exists()
    assert (tmp_path / "ablation_manifest.json").exists()
    assert (tmp_path / "plots" / "ablation_comparison.png").exists()
    assert (tmp_path / "experiment_configs" / "Exp_A.json").exists()
    assert (tmp_path / "experiment_configs" / "Exp_J.json").exists()
