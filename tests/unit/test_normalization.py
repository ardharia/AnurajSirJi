"""Unit tests for RAG/VLM schema normalization and controlled vocabulary mapping."""

from __future__ import annotations

import pytest

from seismic_damage.schemas.normalized import NormalizedAssessmentRecord
from seismic_damage.validation.cross_validation import compare_sources, cross_validate
from seismic_damage.validation.normalize import (
    normalize_binary,
    normalize_collapse_mode,
    normalize_crack_pattern,
    normalize_crack_severity,
    normalize_damage_grade,
    normalize_ground_truth,
    normalize_material_type,
    normalize_number_of_stories,
    normalize_rag_output,
    normalize_record,
    normalize_soil_type,
    normalize_structural_system,
    normalize_vlm_output,
)


def test_controlled_vocabulary_mappings() -> None:
    # Material
    assert normalize_material_type("RC") == "reinforced_concrete"
    assert normalize_material_type("reinforced concrete") == "reinforced_concrete"
    assert normalize_material_type("brick masonry") == "burnt_clay_brick"
    assert normalize_material_type("rubble") == "stone_block"
    assert normalize_material_type("adobe") == "mud"
    assert normalize_material_type("wood") == "timber"
    assert normalize_material_type("unknown_xyz") is None
    assert normalize_material_type(None) is None

    # Structural system
    assert normalize_structural_system("RC frame") == "rc_frame"
    assert normalize_structural_system("moment frame") == "moment_frame"
    assert normalize_structural_system("shear wall") == "shear_wall"
    assert normalize_structural_system("load bearing") == "bearing_wall"
    assert normalize_structural_system("confined masonry") == "confined_masonry"
    assert normalize_structural_system("dhajji dewari") == "timber_frame"
    assert normalize_structural_system("ekra house") == "bamboo_frame"

    # Damage grade
    assert normalize_damage_grade(4) == 4
    assert normalize_damage_grade("Grade 5") == 5
    assert normalize_damage_grade("G3") == 3
    assert normalize_damage_grade("0") == 0
    assert normalize_damage_grade(6) is None
    assert normalize_damage_grade(-1) is None
    assert normalize_damage_grade(None) is None

    # Crack severity & pattern
    assert normalize_crack_severity("minor") == "hairline"
    assert normalize_crack_severity("severe") == "severe"
    assert normalize_crack_pattern("x-shaped") == "x_shaped"
    assert normalize_crack_pattern("stair step") == "stair_step"

    # Collapse mode
    assert normalize_collapse_mode("total collapse") == "total"
    assert normalize_collapse_mode("open ground storey") == "soft_story"
    assert normalize_collapse_mode("pancake") == "pancake"

    # Binary & numeric
    assert normalize_binary("true") is True
    assert normalize_binary(0) is False
    assert normalize_binary(None) is None
    assert normalize_number_of_stories("4") == 4
    assert normalize_number_of_stories(-1) is None
    assert normalize_soil_type("rock") == "hard"


def test_normalize_rag_output() -> None:
    raw_rag = {
        "building_id": "bldg_test_001",
        "earthquake_event": "Bhuj_2001",
        "evidence_source": "rag",
        "evidence_text": "Ground storey collapsed in 4-storey RC frame building.",
        "confidence": 0.85,
        "material_type": "RC Frame",
        "number_of_stories": 4,
        "damage_grade": "Grade 4",
        "collapse_mode": "soft story",
        "soft_story_failure": True,
        "citations": [{"document_id": "doc1", "score": 0.9}],
    }
    rec = normalize_rag_output(raw_rag)

    assert isinstance(rec, NormalizedAssessmentRecord)
    assert rec.building_id == "bldg_test_001"
    assert rec.modality == "rag"
    assert rec.typology.material_type == "reinforced_concrete"
    assert rec.typology.number_of_stories == 4
    assert rec.damage.damage_grade == 4
    assert rec.damage.collapse_mode == "soft_story"
    assert rec.damage.soft_story_failure is True

    # Pre-earthquake vulnerability soft_story must NOT be auto-populated from damage
    assert rec.vulnerability.soft_story is None

    # Provenance and raw_output preserved
    assert len(rec.provenance.citations) == 1
    assert rec.raw_output == raw_rag


def test_normalize_vlm_output() -> None:
    raw_vlm = {
        "building_id": "bldg_test_002",
        "evidence_source": "vlm",
        "material": "reinforced concrete",
        "structural_system": "rc frame",
        "damage_grade": 5,
        "collapse_mode": "total",
        "crack_severity": "severe",
        "crack_pattern": "shear",
        "confidence": 0.9,
        "image_path": "data/raw/images/test.jpg",
        "evidence": "Total pancake collapse observed.",
    }
    rec = normalize_vlm_output(raw_vlm)

    assert rec.building_id == "bldg_test_002"
    assert rec.modality == "vlm"
    assert rec.typology.material_type == "reinforced_concrete"
    assert rec.typology.structural_system == "rc_frame"
    assert rec.damage.damage_grade == 5
    assert rec.damage.collapse_mode == "total"
    assert rec.damage.crack_severity == "severe"
    assert rec.damage.crack_pattern == "shear"
    assert "data/raw/images/test.jpg" in rec.provenance.image_paths
    assert rec.raw_output == raw_vlm


def test_cross_validation_compatibility() -> None:
    gt_rec = normalize_ground_truth(
        {
            "building_id": "bldg_eval_01",
            "damage_grade": 4,
            "material_type": "reinforced_concrete",
            "number_of_stories": 5,
        }
    )
    rag_rec = normalize_rag_output(
        {
            "building_id": "bldg_eval_01",
            "damage_grade": 4,
            "material_type": "RC",
            "number_of_stories": 5,
        }
    )
    vlm_rec = normalize_vlm_output(
        {
            "building_id": "bldg_eval_01",
            "damage_grade": 3,
            "material": "concrete",
            "number_of_stories": 5,
        }
    )

    result = cross_validate(
        ground_truth=[gt_rec.to_dict()],
        rag=[rag_rec.to_dict()],
        vlm=[vlm_rec.to_dict()],
    )

    assert result["fusion"] is False
    assert len(result["comparisons"]) == 3

    # GT vs RAG should be exact match on damage_grade
    gt_rag = next(c for c in result["comparisons"] if c["left"] == "ground_truth" and c["right"] == "rag")
    assert gt_rag["parameters"]["damage_grade"]["exact_match"] == 1.0
    assert gt_rag["parameters"]["material_type"]["accuracy"] == 1.0

    # GT vs VLM has 1-grade difference
    gt_vlm = next(c for c in result["comparisons"] if c["left"] == "ground_truth" and c["right"] == "vlm")
    assert gt_vlm["parameters"]["damage_grade"]["mae"] == 1.0
    assert gt_vlm["parameters"]["damage_grade"]["within_one"] == 1.0


def test_auto_dispatcher() -> None:
    rec_rag = normalize_record({"building_id": "b1", "evidence_source": "rag", "damage_grade": 2})
    assert rec_rag.modality == "rag"
    assert rec_rag.damage.damage_grade == 2

    rec_vlm = normalize_record({"building_id": "b2", "evidence_source": "vlm", "damage_grade": 3})
    assert rec_vlm.modality == "vlm"
    assert rec_vlm.damage.damage_grade == 3

    rec_gt = normalize_record({"building_id": "b3", "damage_grade": 5})
    assert rec_gt.modality == "ground_truth"
    assert rec_gt.damage.damage_grade == 5
