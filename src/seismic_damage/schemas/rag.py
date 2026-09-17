"""Citation-bearing RAG answer schemas."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class RAGCitation(BaseModel):
    """Pointer from an answer back to a retrieved chunk."""

    document_id: str
    chunk_id: str | None = None
    page_number: int | None = None
    score: float = 0.0
    excerpt: str = ""
    source_path: str | None = None
