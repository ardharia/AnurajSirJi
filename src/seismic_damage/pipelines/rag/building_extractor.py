# src/seismic_damage/pipelines/rag/building_extractor.py
"""Utilities to extract building identifiers and typologies using the RAG pipeline.

The existing script ``visualization/a_get_building_ids.py`` prints a list of unique
building IDs by issuing a plain RAG query.  This module wraps that logic in a
function that returns the IDs as a Python list and optionally also retrieves the
typology for each building via a second RAG query.
"""

from __future__ import annotations

from typing import List, Tuple

from seismic_damage.pipelines.rag.pipeline import RAGPipeline


def get_unique_building_ids(query: str = "List every unique building identifier mentioned in these Bhuj 2001 reports.") -> List[str]:
    """Return a list of unique building IDs extracted from the indexed corpus.

    The RAG pipeline is expected to have an indexed FAISS store.  The answer is
    parsed for identifiers that look like alphanumeric strings (e.g. ``B123``).
    """
    rag = RAGPipeline()
    # Ensure the vector store is built; if not, index default chunks.
    if rag.store is None:
        rag.index_corpus()
    result = rag.run(query)
    text = result.answer or ""
    # Very simple heuristic: split on whitespace/comma and keep tokens that contain a digit.
    candidates = [tok.strip(",;.") for tok in text.split()]
    ids = [c for c in candidates if any(ch.isdigit() for ch in c)]
    # Preserve order while deduplicating
    seen = set()
    unique_ids = []
    for bid in ids:
        if bid not in seen:
            seen.add(bid)
            unique_ids.append(bid)
    return unique_ids


def get_building_typology(building_id: str) -> Tuple[str, str]:
    """Retrieve typology information for a single building.

    Returns a ``(building_id, typology)`` tuple.  The query asks the RAG store for
    the typology label that appears in the report for the given identifier.
    """
    rag = RAGPipeline()
    if rag.store is None:
        rag.index_corpus()
    query = f"What is the typology (RVS classification) of building {building_id} in the Bhuj 2001 reports?"
    result = rag.run(query)
    typology = result.answer or "UNKNOWN"
    typology = typology.strip().upper()
    return building_id, typology

__all__ = ["get_unique_building_ids", "get_building_typology"]
