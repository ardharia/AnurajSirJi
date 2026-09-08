"""Independent retrieval-augmented generation pipeline (stub)."""

from __future__ import annotations

from pathlib import Path

from seismic_damage.config import Settings, get_settings
from seismic_damage.schemas.pipeline import RAGResult


class RAGPipeline:
    """Knowledge retrieval pipeline for codes, fragility tables, and reports.

    Designed to run independently of the VLM pipeline.
    """

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()

    def index_corpus(self, corpus_dir: Path | None = None) -> int:
        """Index documents from the knowledge corpus. Returns document count.

        Not implemented yet — returns 0.
        """
        _ = corpus_dir or self.settings.paths.knowledge_dir
        return 0

    def retrieve(self, query: str, top_k: int | None = None) -> RAGResult:
        """Retrieve relevant documents for a natural-language query."""
        k = top_k if top_k is not None else self.settings.rag.top_k
        if not self.settings.rag.enabled:
            return RAGResult(query=query, synthesized_context="RAG disabled")
        # Placeholder until vector store + embeddings are wired.
        return RAGResult(
            query=query,
            documents=[],
            synthesized_context=None,
            extracted_hints={"top_k": k, "status": "not_implemented"},
        )

    def run(self, query: str, top_k: int | None = None) -> RAGResult:
        """Public entry point for the RAG pipeline."""
        return self.retrieve(query=query, top_k=top_k)


def run_rag(query: str, settings: Settings | None = None) -> RAGResult:
    """Functional entry point for the independent RAG pipeline."""
    return RAGPipeline(settings=settings).run(query)
