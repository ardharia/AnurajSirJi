import pytest
from pathlib import Path

from seismic_damage.config.settings import Settings
from seismic_damage.pipelines.vlm.vlm_graph import VLMGraph, VLMGraphResult
from seismic_damage.linking.catalog import BuildingRecord


@pytest.fixture
def mock_catalog():
    return [
        BuildingRecord(
            building_id="bldg_12345",
            source_collection="test",
            image_ids=["img_1"],
            evidence_links=[],
            caption="Masonry building with Grade 4 damage."
        )
    ]

@pytest.fixture
def mock_image_rows():
    return [
        {
            "image_id": "img_1",
            "filename": "img_1.jpg",
            "caption": "Masonry building with Grade 4 damage.",
            "source_collection": "test"
        }
    ]


def test_vlm_graph_execution(mock_catalog, mock_image_rows, tmp_path):
    settings = Settings()
    # Disable VLM or use mock
    settings.vlm.enabled = True
    settings.vlm.provider = "mock"
    
    graph = VLMGraph(settings=settings)
    result = graph.run(catalog=mock_catalog, image_rows=mock_image_rows)
    
    assert isinstance(result, VLMGraphResult)
    assert len(result.records) == 1
    
    assert result.records[0].building_id == "bldg_12345"
    assert result.records[0].modality == "vlm"
    
    # Test JSONL serialization
    out_path = tmp_path / "vlm_out.jsonl"
    result.to_jsonl(out_path)
    
    assert out_path.exists()
    assert "bldg_12345" in out_path.read_text()
