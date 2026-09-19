import pytest
from seismic_damage.fusion.weights import compute_reliability_weights


def test_compute_reliability_weights():
    # Construct a dummy error report mimicking the output of Step 8
    dummy_report = {
        "n_buildings_evaluated": 110,
        "rag_metrics": {
            "soft_story_failure": {"kind": "binary", "n": 110, "f1": 0.8},
            "damage_grade": {"kind": "ordinal", "n": 110, "accuracy": 0.5, "qwk": 0.6},
            "number_of_stories": {"kind": "numerical", "n": 110, "mae": 1.0}
        },
        "vlm_metrics": {
            "soft_story_failure": {"kind": "binary", "n": 110, "f1": 0.4},
            "damage_grade": {"kind": "ordinal", "n": 110, "accuracy": 0.8, "qwk": 0.9},
            "number_of_stories": {"kind": "numerical", "n": 110, "mae": 0.0} # Perfect
        }
    }
    
    weights_doc = compute_reliability_weights(dummy_report)
    
    assert "metadata" in weights_doc
    assert weights_doc["metadata"]["n_buildings_evaluated"] == 110
    
    params = weights_doc["parameter_weights"]
    
    # Check binary (soft_story_failure)
    # RAG has F1 0.8, VLM has F1 0.4
    assert params["soft_story_failure"]["rag_confidence"] == 0.8
    assert params["soft_story_failure"]["vlm_confidence"] == 0.4
    assert params["soft_story_failure"]["primary_source"] == "rag"
    assert params["soft_story_failure"]["fusion_weight_rag"] > params["soft_story_failure"]["fusion_weight_vlm"]
    
    # Check ordinal (damage_grade)
    # VLM has much better accuracy and QWK
    assert params["damage_grade"]["primary_source"] == "vlm"
    assert params["damage_grade"]["vlm_confidence"] > params["damage_grade"]["rag_confidence"]
    
    # Check numerical (number_of_stories)
    # RAG MAE = 1.0 -> 1 / (1+1) = 0.5
    # VLM MAE = 0.0 -> 1 / (1+0) = 1.0
    assert params["number_of_stories"]["rag_confidence"] == 0.5
    assert params["number_of_stories"]["vlm_confidence"] == 1.0
    assert params["number_of_stories"]["primary_source"] == "vlm"
