"""Independent execution graph for the RAG (text) module."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from seismic_damage.config.settings import Settings
from seismic_damage.linking.catalog import build_catalog, BuildingRecord
from seismic_damage.pipelines.rag.pipeline import RAGPipeline
from seismic_damage.extraction.text_parser import parse_damage_text
from seismic_damage.validation.normalize import normalize_rag_output
from seismic_damage.schemas.normalized import NormalizedAssessmentRecord
from seismic_damage.io_utils import ensure_parent, write_jsonl


@dataclass
class RAGGraphResult:
    records: list[NormalizedAssessmentRecord]
    metadata: dict[str, Any]
    elapsed_seconds: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "metadata": self.metadata,
            "elapsed_seconds": self.elapsed_seconds,
            "records": [r.to_dict() for r in self.records],
        }

    def to_jsonl(self, path: Path | str) -> None:
        write_jsonl(path, [r.to_dict() for r in self.records])


class RAGGraph:
    """Execution graph for generating independent text-based building assessments."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or Settings()
        self.pipeline = RAGPipeline(settings=self.settings)

    def _query_for_building(self, building: BuildingRecord) -> str:
        """Formulate a specific retrieval query for the building."""
        return (
            f"What is the damage grade, structural system, material type, "
            f"and crack severity for building {building.building_id} or {building.name}? "
            f"Are there soft story failures or wall failures?"
        )

    def run(self, catalog: list[BuildingRecord] | None = None) -> RAGGraphResult:
        """Execute the RAG graph pipeline."""
        start_t = time.time()
        
        if catalog is None:
            catalog = build_catalog()
            
        records: list[NormalizedAssessmentRecord] = []
        for building in catalog:
            query = self._query_for_building(building)
            
            # 1. Retrieve
            result = self.pipeline.retrieve(
                query=query, 
                top_k=self.settings.rag.top_k, 
                score_threshold=self.settings.rag.similarity_threshold
            )
            
            # 2. Extract
            # The context is all documents concatenated
            context_text = "\n\n".join([doc.text for doc in result.documents])
            # If no context found, we still parse the empty text to get empty parameters
            extraction = parse_damage_text(context_text)
            
            # 3. Normalize
            normalized = normalize_rag_output(
                extraction=extraction,
                building=building,
                rag_result=result
            )
            
            records.append(normalized)
            
        elapsed = time.time() - start_t
        
        metadata = {
            "n_buildings": len(catalog),
            "n_records": len(records),
            "modality": "rag",
            "model": self.settings.llm.model_name
        }
        
        return RAGGraphResult(
            records=records,
            metadata=metadata,
            elapsed_seconds=elapsed
        )


def run_rag_graph(
    output_path: Path | str | None = None,
    settings: Settings | None = None
) -> RAGGraphResult:
    """Functional entry point to execute the RAG graph and persist results."""
    settings = settings or Settings()
    graph = RAGGraph(settings=settings)
    result = graph.run()
    
    if output_path:
        path = ensure_parent(output_path)
        result.to_jsonl(path)
        
    return result
