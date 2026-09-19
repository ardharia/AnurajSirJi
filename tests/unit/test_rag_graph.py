import pytest
from pathlib import Path

from seismic_damage.config.settings import Settings
from seismic_damage.pipelines.rag.rag_graph import RAGGraph, RAGGraphResult
from seismic_damage.linking.catalog import BuildingRecord


@pytest.fixture
def mock_catalog():
    return [
        BuildingRecord(
            building_id="bldg_12345",
            source_collection="test",
            image_ids=["img_1"],
            evidence_links=[],
            caption="Masonry building with Grade 4 damage and soft story failure."
        ),
        BuildingRecord(
            building_id="bldg_67890",
            source_collection="test",
            image_ids=["img_2"],
            evidence_links=[],
            caption="RC building with intact walls, Grade 1."
        )
    ]


def test_rag_graph_execution(mock_catalog, tmp_path):
    settings = Settings()
    # Use disabled/mock VLM/LLM if applicable by setting config
    
    graph = RAGGraph(settings=settings)
    
    # We can pass mock catalog directly
    result = graph.run(catalog=mock_catalog)
    
    assert isinstance(result, RAGGraphResult)
    assert len(result.records) == 2
    
    # Check that it outputs NormalizedAssessmentRecord
    assert result.records[0].building_id == "bldg_12345"
    assert result.records[0].modality == "rag"
    assert result.records[1].building_id == "bldg_67890"
    
    # Test JSONL serialization
    out_path = tmp_path / "rag_out.jsonl"
    result.to_jsonl(out_path)
    
    assert out_path.exists()
    content = out_path.read_text()
    assert "bldg_12345" in content
    assert "bldg_67890" in content
