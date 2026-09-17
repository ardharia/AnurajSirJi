"""Independent retrieval-augmented generation pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from seismic_damage.config import Settings, get_settings
from seismic_damage.config.settings import PROJECT_ROOT
from seismic_damage.io_utils import read_jsonl
from seismic_damage.pipelines.rag.embedders import Embedder, default_embedder
from seismic_damage.pipelines.rag.generate import (
    citations_from_documents,
    evidence_is_sufficient,
    extractive_answer,
    refusal_message,
)
from seismic_damage.pipelines.rag.store import FaissVectorStore
from seismic_damage.schemas.ingestion import TextChunk
from seismic_damage.schemas.pipeline import RAGDocument, RAGResult

DEFAULT_CHUNKS_PATH = PROJECT_ROOT / "data" / "processed" / "chunks" / "report_chunks.jsonl"


class RAGPipeline:
    """Knowledge retrieval pipeline for codes, fragility tables, and reports.

    Designed to run independently of the VLM pipeline.
    """

    def __init__(
        self,
        settings: Settings | None = None,
        embedder: Embedder | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self._embedder = embedder
        self.store: FaissVectorStore | None = None

    @property
    def embedder(self) -> Embedder:
        if self._embedder is None:
            self._embedder = default_embedder(self.settings.rag.embedding_model)
        return self._embedder

    def _resolve_dir(self, path: Path) -> Path:
        return path if path.is_absolute() else PROJECT_ROOT / path

    def _chunks_from_dir(self, corpus_dir: Path) -> list[TextChunk]:
        chunks: list[TextChunk] = []
        jsonl_files = sorted(corpus_dir.glob("*.jsonl"))
        if jsonl_files:
            for path in jsonl_files:
                for row in read_jsonl(path):
                    chunks.append(TextChunk.model_validate(row))
            return chunks
        for path in sorted(corpus_dir.rglob("*.txt")):
            text = path.read_text(encoding="utf-8", errors="ignore").strip()
            if not text:
                continue
            chunks.append(
                TextChunk(
                    chunk_id=f"{path.stem}::p0001::c0000",
                    document_id=path.stem,
                    source_path=str(path),
                    filename=path.name,
                    page_number=1,
                    chunk_index=0,
                    text=text,
                    n_chars=len(text),
                    metadata={},
                )
            )
        return chunks

    def load_chunks(self, corpus_dir: Path | None = None) -> list[TextChunk]:
        if corpus_dir is not None:
            directory = self._resolve_dir(Path(corpus_dir))
            if directory.exists():
                return self._chunks_from_dir(directory)
            return []
        if DEFAULT_CHUNKS_PATH.exists():
            return [TextChunk.model_validate(row) for row in read_jsonl(DEFAULT_CHUNKS_PATH)]
        knowledge = self._resolve_dir(Path(self.settings.paths.knowledge_dir))
        if knowledge.exists():
            return self._chunks_from_dir(knowledge)
        return []

    def index_corpus(self, corpus_dir: Path | None = None) -> int:
        """Index report chunks (or a knowledge directory) into FAISS."""

        chunks = self.load_chunks(corpus_dir)
        if not chunks:
            self.store = None
            return 0
        texts = [chunk.text for chunk in chunks]
        if hasattr(self.embedder, "fit"):
            self.embedder.fit(texts)  # type: ignore[attr-defined]
        vectors = np.asarray(self.embedder.encode(texts), dtype=np.float32)
        if vectors.ndim == 1:
            vectors = vectors.reshape(1, -1)
        store = FaissVectorStore(dim=int(vectors.shape[1]))
        payloads = [chunk.model_dump() for chunk in chunks]
        store.add(vectors, [chunk.chunk_id for chunk in chunks], payloads)
        self.store = store
        store.save(self._resolve_dir(Path(self.settings.rag.vector_store_path)))
        return store.size

    def _load_store(self) -> FaissVectorStore | None:
        if self.store is not None:
            return self.store
        vector_dir = self._resolve_dir(Path(self.settings.rag.vector_store_path))
        if (vector_dir / "meta.json").exists():
            self.store = FaissVectorStore.load(vector_dir)
            if hasattr(self.embedder, "fit"):
                texts = [str(item.get("text") or "") for item in self.store.payloads]
                self.embedder.fit(texts)  # type: ignore[attr-defined]
            return self.store
        return None

    def retrieve(self, query: str, top_k: int | None = None) -> RAGResult:
        """Retrieve relevant documents for a natural-language query."""

        k = top_k if top_k is not None else self.settings.rag.top_k
        if not self.settings.rag.enabled:
            return RAGResult(query=query, synthesized_context="RAG disabled")

        store = self._load_store()
        if store is None or store.size == 0:
            return RAGResult(
                query=query,
                documents=[],
                synthesized_context=None,
                extracted_hints={"top_k": k, "n_documents": 0, "status": "empty_index"},
                answer=None,
                evidence_sufficient=False,
            )

        query_vec = np.asarray(self.embedder.encode([query]), dtype=np.float32)
        if query_vec.ndim == 1:
            query_vec = query_vec.reshape(1, -1)
        if query_vec.shape[1] != store.dim and store.dim == 4096:
            from seismic_damage.pipelines.rag.embedders import TfidfEmbedder

            self._embedder = TfidfEmbedder()
            texts = [str(item.get("text") or "") for item in store.payloads]
            self._embedder.fit(texts)
            query_vec = np.asarray(self._embedder.encode([query]), dtype=np.float32)
            if query_vec.ndim == 1:
                query_vec = query_vec.reshape(1, -1)
        hits = store.search(query_vec, k)
        documents: list[RAGDocument] = []
        for chunk_id, score, payload in hits:
            documents.append(
                RAGDocument(
                    document_id=str(payload.get("document_id") or chunk_id),
                    content=str(payload.get("text") or ""),
                    score=max(0.0, float(score)),
                    metadata={
                        "chunk_id": payload.get("chunk_id") or chunk_id,
                        "page_number": payload.get("page_number"),
                        "chunk_index": payload.get("chunk_index"),
                        "filename": payload.get("filename"),
                    },
                    source_path=payload.get("source_path"),
                )
            )
        sufficient = evidence_is_sufficient(
            documents,
            min_score=self.settings.rag.min_score,
            refuse_if_insufficient=self.settings.rag.refuse_if_insufficient,
        )
        citations = citations_from_documents(documents) if sufficient else []
        answer = extractive_answer(query, documents) if sufficient else refusal_message(query)
        context = "\n\n".join(doc.content for doc in documents) if sufficient else None
        return RAGResult(
            query=query,
            documents=documents,
            synthesized_context=context,
            extracted_hints={
                "top_k": k,
                "n_documents": len(documents),
                "backend": store.backend,
                "embedder": getattr(self.embedder, "name", type(self.embedder).__name__),
                "evidence_sufficient": sufficient,
            },
            answer=answer,
            citations=citations,
            evidence_sufficient=sufficient,
        )

    def run(self, query: str, top_k: int | None = None) -> RAGResult:
        """Public entry point for the RAG pipeline."""

        return self.retrieve(query=query, top_k=top_k)

    def assess_building(self, *, building_id: str, query: str, extractor: Any) -> dict[str, Any]:
        """Retrieve evidence for one building and extract parameters independently of VLM."""

        result = self.run(query)
        payload: dict[str, Any] = {
            "building_id": building_id,
            "earthquake_event": "Bhuj_2001",
            "evidence_source": "rag",
            "confidence": 0.0,
            "evidence_text": result.answer,
            "rag_evidence_sufficient": result.evidence_sufficient,
            "citations": [item.model_dump() for item in result.citations],
        }
        if result.evidence_sufficient and result.synthesized_context:
            extracted = extractor(result.synthesized_context)
            payload.update(extracted)
            payload["confidence"] = float(extracted.get("confidence") or 0.0)
        return payload


def run_rag(query: str, settings: Settings | None = None) -> RAGResult:
    """Functional entry point for the independent RAG pipeline."""

    return RAGPipeline(settings=settings).run(query)
