import pytest
import pandas as pd

from seismic_damage.stats.feature_importance import calculate_feature_importance, SKLEARN_AVAILABLE


@pytest.mark.skipif(not SKLEARN_AVAILABLE, reason="Requires scikit-learn")
def test_calculate_feature_importance():
    # Construct dummy ground truth records with perfect correlation
    # material_type strongly dictates damage_grade here
    gt_records = [
        {"building_id": "b1", "parameters": {"damage_grade": 5, "material_type": "masonry", "soft_story": True}},
        {"building_id": "b2", "parameters": {"damage_grade": 5, "material_type": "masonry", "soft_story": False}},
        {"building_id": "b3", "parameters": {"damage_grade": 5, "material_type": "masonry", "soft_story": True}},
        {"building_id": "b4", "parameters": {"damage_grade": 1, "material_type": "rc", "soft_story": False}},
        {"building_id": "b5", "parameters": {"damage_grade": 1, "material_type": "rc", "soft_story": False}},
        {"building_id": "b6", "parameters": {"damage_grade": 1, "material_type": "rc", "soft_story": True}},
    ]
    
    df = calculate_feature_importance(gt_records)
    
    assert not df.empty
    assert "parameter" in df.columns
    assert "importance_score" in df.columns
    
    # material_type perfectly separates damage grade 1 and 5
    # soft_story is randomized/decorrelated
    top_feature = df.iloc[0]["parameter"]
    assert top_feature == "material_type"
    
    # Check that material_type score is higher than soft_story
    mat_score = df.loc[df["parameter"] == "material_type", "importance_score"].values[0]
    soft_score = df.loc[df["parameter"] == "soft_story", "importance_score"].values[0]
    
    assert mat_score > soft_score
