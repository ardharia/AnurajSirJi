"""Schemas for PDF extraction, cleaning, and chunk provenance."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class TextChunk(BaseModel):
    """A cleaned text window with document/page/chunk provenance."""

    chunk_id: str
    document_id: str
    source_path: str
    filename: str
    page_number: int = Field(..., ge=1)
    chunk_index: int = Field(..., ge=0)
    text: str
    n_chars: int = Field(..., ge=0)
    metadata: dict[str, Any] = Field(default_factory=dict)


class DocumentIngestRecord(BaseModel):
    """Per-document ingestion outcome."""

    document_id: str
    source_path: str
    filename: str
    n_pages: int = 0
    n_chunks: int = 0
    status: str
    error: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class IngestResult(BaseModel):
    """Corpus-level PDF ingestion summary."""

    n_documents: int = 0
    n_pages: int = 0
    n_chunks: int = 0
    skipped: int = 0
    documents: list[DocumentIngestRecord] = Field(default_factory=list)
    chunks: list[TextChunk] = Field(default_factory=list)
    output_path: str | None = None
