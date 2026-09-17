"""Provenance-preserving chunking of cleaned page text."""

from __future__ import annotations

from pathlib import Path

from seismic_damage.schemas.ingestion import TextChunk


def _windows(text: str, chunk_size: int, overlap: int) -> list[str]:
    if not text:
        return []
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    overlap = max(0, min(overlap, chunk_size - 1))
    pieces: list[str] = []
    start = 0
    n = len(text)
    while start < n:
        end = min(n, start + chunk_size)
        piece = text[start:end].strip()
        if piece:
            pieces.append(piece)
        if end >= n:
            break
        next_start = end - overlap
        if next_start <= start:
            next_start = end
        start = next_start
    return pieces


def chunk_page_text(
    *,
    text: str,
    document_id: str,
    source_path: str | Path,
    page_number: int,
    chunk_size: int,
    chunk_overlap: int,
    metadata: dict | None = None,
) -> list[TextChunk]:
    """Split one page into overlapping character windows."""

    path = Path(source_path)
    windows = _windows(text, chunk_size=chunk_size, overlap=chunk_overlap)
    extra = dict(metadata or {})
    chunks: list[TextChunk] = []
    for index, window in enumerate(windows):
        chunk_id = f"{document_id}::p{page_number:04d}::c{index:04d}"
        chunks.append(
            TextChunk(
                chunk_id=chunk_id,
                document_id=document_id,
                source_path=str(path),
                filename=path.name,
                page_number=page_number,
                chunk_index=index,
                text=window,
                n_chars=len(window),
                metadata=extra,
            )
        )
    return chunks
