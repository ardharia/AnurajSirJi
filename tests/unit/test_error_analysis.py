import pytest
from pathlib import Path

from seismic_damage.validation.error_analysis import run_error_analysis


def test_error_analysis(tmp_path):
    gt_records = [
        {"building_id": "bldg_1", "parameters": {"damage_grade": 4, "material_type": "rc"}},
        {"building_id": "bldg_2", "parameters": {"damage_grade": 2, "material_type": "masonry"}},
    ]
    
    rag_records = [
        {"building_id": "bldg_1", "parameters": {"damage_grade": 4, "material_type": "rc"}},
        {"building_id": "bldg_2", "parameters": {"damage_grade": 3, "material_type": "rc"}},
    ]
    
    vlm_records = [
        {"building_id": "bldg_1", "parameters": {"damage_grade": 3, "material_type": "rc"}},
        {"building_id": "bldg_2", "parameters": {"damage_grade": 2, "material_type": "masonry"}},
    ]
    
    output_dir = tmp_path / "results"
    report = run_error_analysis(
        gt_records=gt_records,
        rag_records=rag_records,
        vlm_records=vlm_records,
        output_dir=output_dir
    )
    
    assert report["n_buildings_evaluated"] == 2
    
    # RAG matches bldg_1 exactly, misses bldg_2
    assert report["differential"]["damage_grade"]["rag_better"] == 1
    # VLM matches bldg_2 exactly, misses bldg_1
    assert report["differential"]["damage_grade"]["vlm_better"] == 1
    
    # material_type
    assert report["differential"]["material_type"]["both_correct"] == 1 # bldg_1
    assert report["differential"]["material_type"]["vlm_better"] == 1 # bldg_2
    
    # Check outputs
    assert (output_dir / "error_analysis_report.json").exists()
    assert (output_dir / "error_analysis_summary.csv").exists()
