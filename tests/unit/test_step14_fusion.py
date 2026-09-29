"""Unit tests for Step 14: Calibrated RAG + VLM Fusion."""

from __future__ import annotations

from pathlib import Path
import pytest

from seismic_damage.fusion.fusion import (
    fuse_building,
    generate_fused_dataset,
    write_predictions_csv,
)
from seismic_damage.fusion.reliability import (
    compute_parameter_reliability,
    reliability_weights_by_param,
)
from seismic_damage.fusion.weighting import (
    fuse_binary,
    fuse_categorical,
    fuse_numerical,
    fuse_ordinal,
    fuse_parameter,
)
from seismic_damage.fusion.evaluation import (
    build_fusion_comparison,
    evaluate_records_against_gt,
    run_fusion_pipeline,
)


def test_weight_normalization() -> None:
    # Build simple synthetic records with known accuracy
    rag_records = [
        {"building_id": "B1", "damage_grade": 4, "soft_story_failure": 1, "material_type": "brick"},
        {"building_id": "B2", "damage_grade": 2, "soft_story_failure": 0, "material_type": "concrete"},
        {"building_id": "B3", "damage_grade": 3, "soft_story_failure": 1, "material_type": "brick"},
    ]
    vlm_records = [
        {"building_id": "B1", "damage_grade": 4, "soft_story_failure": 1, "material_type": "brick"},
        {"building_id": "B2", "damage_grade": 1, "soft_story_failure": 0, "material_type": "concrete"},
        {"building_id": "B3", "damage_grade": 3, "soft_story_failure": 0, "material_type": "brick"},
    ]
    gt_records = [
        {"building_id": "B1", "damage_grade": 4, "soft_story_failure": 1, "material_type": "brick"},
        {"building_id": "B2", "damage_grade": 2, "soft_story_failure": 0, "material_type": "concrete"},
        {"building_id": "B3", "damage_grade": 3, "soft_story_failure": 1, "material_type": "brick"},
    ]

    reliability_rows = compute_parameter_reliability(rag_records, vlm_records, gt_records)
    assert len(reliability_rows) > 0

    for row in reliability_rows:
        rw = float(row["rag_weight"])
        vw = float(row["vlm_weight"])
        # Weights must sum to 1.0 or both be 0.0 (if insufficient evidence)
        assert (rw == 0.0 and vw == 0.0) or abs((rw + vw) - 1.0) < 1e-5
        assert 0.0 <= rw <= 1.0
        assert 0.0 <= vw <= 1.0


def test_missing_source_logic() -> None:
    weights = {"damage_grade": {"rag_weight": 0.7, "vlm_weight": 0.3}}

    # Case 1: Both available -> calibrated fusion
    c1 = fuse_building(
        "B1",
        rag_rec={"building_id": "B1", "damage_grade": 4},
        vlm_rec={"building_id": "B1", "damage_grade": 2},
        weights_by_param=weights,
        mode="F4_calibrated",
    )
    # 0.7 * 4 + 0.3 * 2 = 2.8 + 0.6 = 3.4 -> rounded to 3
    assert c1["damage_grade"] == 3

    # Case 2: RAG available, VLM missing -> RAG only (never zero)
    c2 = fuse_building(
        "B2",
        rag_rec={"building_id": "B2", "damage_grade": 4},
        vlm_rec=None,
        weights_by_param=weights,
        mode="F4_calibrated",
    )
    assert c2["damage_grade"] == 4
    assert c2["damage_grade"] != 0

    # Case 3: VLM available, RAG missing -> VLM only (never zero)
    c3 = fuse_building(
        "B3",
        rag_rec=None,
        vlm_rec={"building_id": "B3", "damage_grade": 2},
        weights_by_param=weights,
        mode="F4_calibrated",
    )
    assert c3["damage_grade"] == 2
    assert c3["damage_grade"] != 0

    # Case 4: Both missing -> None (never invent a prediction)
    c4 = fuse_building(
        "B4",
        rag_rec=None,
        vlm_rec=None,
        weights_by_param=weights,
        mode="F4_calibrated",
    )
    assert c4["damage_grade"] is None
    assert c4["damage_grade"] != 0


def test_ordinal_handling() -> None:
    # damage_grade is ordinal
    # 0.8 * 5 + 0.2 * 1 = 4.0 + 0.2 = 4.2 -> rounds to 4
    val = fuse_ordinal(5, 1, rag_weight=0.8, vlm_weight=0.2)
    assert val == 4
    assert isinstance(val, int)

    # Missing cases
    assert fuse_ordinal(None, 3, 0.5, 0.5) == 3
    assert fuse_ordinal(3, None, 0.5, 0.5) == 3
    assert fuse_ordinal(None, None, 0.5, 0.5) is None


def test_binary_handling() -> None:
    # RAG weight is higher -> RAG choice wins
    assert fuse_binary(True, False, rag_weight=0.7, vlm_weight=0.3) is True
    # VLM weight is higher -> VLM choice wins
    assert fuse_binary(True, False, rag_weight=0.3, vlm_weight=0.7) is False
    # Agreement
    assert fuse_binary(True, True, rag_weight=0.5, vlm_weight=0.5) is True
    # Missing handling
    assert fuse_binary(True, None, 0.5, 0.5) is True
    assert fuse_binary(None, False, 0.5, 0.5) is False
    assert fuse_binary(None, None, 0.5, 0.5) is None


def test_categorical_handling() -> None:
    # Agreement
    assert fuse_categorical("brick", "brick", 0.6, 0.4) == "brick"
    # Disagreement: higher weight source wins
    assert fuse_categorical("brick", "stone", 0.7, 0.3) == "brick"
    assert fuse_categorical("brick", "stone", 0.3, 0.7) == "stone"
    # Missing
    assert fuse_categorical("brick", None, 0.5, 0.5) == "brick"
    assert fuse_categorical(None, "stone", 0.5, 0.5) == "stone"
    assert fuse_categorical(None, None, 0.5, 0.5) is None


def test_numerical_handling() -> None:
    # Weighted average: 0.75 * 4.0 + 0.25 * 8.0 = 3.0 + 2.0 = 5.0
    val = fuse_numerical(4.0, 8.0, rag_weight=0.75, vlm_weight=0.25)
    assert abs(val - 5.0) < 1e-6

    # Missing
    assert fuse_numerical(4.0, None, 0.5, 0.5) == 4.0
    assert fuse_numerical(None, 8.0, 0.5, 0.5) == 8.0
    assert fuse_numerical(None, None, 0.5, 0.5) is None


def test_fusion_pipeline_end_to_end(tmp_path: Path) -> None:
    gt_records = [
        {"building_id": "B1", "damage_grade": 4, "soft_story_failure": 1, "material_type": "brick"},
        {"building_id": "B2", "damage_grade": 2, "soft_story_failure": 0, "material_type": "concrete"},
        {"building_id": "B3", "damage_grade": 3, "soft_story_failure": 1, "material_type": "brick"},
    ]
    rag_records = [
        {"building_id": "B1", "damage_grade": 4, "soft_story_failure": 1, "material_type": "brick"},
        {"building_id": "B2", "damage_grade": 2, "soft_story_failure": 0, "material_type": "concrete"},
        {"building_id": "B3", "damage_grade": 2, "soft_story_failure": 0, "material_type": "brick"},
    ]
    vlm_records = [
        {"building_id": "B1", "damage_grade": 4, "soft_story_failure": 1, "material_type": "brick"},
        {"building_id": "B2", "damage_grade": 1, "soft_story_failure": 0, "material_type": "concrete"},
        {"building_id": "B3", "damage_grade": 3, "soft_story_failure": 1, "material_type": "brick"},
    ]

    res = run_fusion_pipeline(
        rag_records=rag_records,
        vlm_records=vlm_records,
        ground_truth_records=gt_records,
        output_dir=tmp_path,
    )

    assert (tmp_path / "parameter_reliability.csv").exists()
    assert (tmp_path / "fusion_predictions.csv").exists()
    assert (tmp_path / "fusion_metrics.csv").exists()
    assert (tmp_path / "fusion_comparison.csv").exists()
    assert (tmp_path / "fusion_manifest.json").exists()
    assert len(res["comparison"]) > 0
