"""PDF ingestion for Bhuj 2001 reconnaissance reports."""

from __future__ import annotations

import csv
from collections.abc import Callable
from pathlib import Path
from typing import Any

from seismic_damage.config.settings import PROJECT_ROOT, get_settings
from seismic_damage.ingestion.chunk import chunk_page_text
from seismic_damage.ingestion.clean import clean_text
from seismic_damage.io_utils import write_json, write_jsonl
from seismic_damage.schemas.ingestion import DocumentIngestRecord, IngestResult, TextChunk

PageExtractor = Callable[[Path], list[tuple[int, str]]]

DEFAULT_SOURCES_CSV = PROJECT_ROOT / "data" / "annotations" / "bhuj_2001_github_sources.csv"
DEFAULT_REPORTS_DIR = PROJECT_ROOT / "data" / "reports" / "bhuj_2001_github"
DEFAULT_CHUNKS_PATH = PROJECT_ROOT / "data" / "processed" / "chunks" / "report_chunks.jsonl"
DEFAULT_SUMMARY_PATH = PROJECT_ROOT / "data" / "processed" / "chunks" / "ingest_summary.json"


def extract_pdf_pages(path: Path) -> list[tuple[int, str]]:
    """Extract raw text per page using PyMuPDF."""

    try:
        import fitz  # type: ignore
    except ImportError as exc:  # pragma: no cover - exercised when dependency missing
        raise RuntimeError(
            "pymupdf is required for PDF extraction. Install with: pip install pymupdf"
        ) from exc

    pages: list[tuple[int, str]] = []
    with fitz.open(path) as document:
        for index, page in enumerate(document, start=1):
            pages.append((index, page.get_text("text") or ""))
    return pages


def _load_source_rows(csv_path: Path) -> list[dict[str, str]]:
    if not csv_path.exists():
        return []
    with csv_path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def ingest_pdf_reports(
    *,
    sources_csv: Path | None = None,
    reports_dir: Path | None = None,
    output_path: Path | None = None,
    summary_path: Path | None = None,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
    extractor: PageExtractor | None = None,
) -> IngestResult:
    """Extract, clean, and chunk allowlisted earthquake reports."""

    settings = get_settings()
    csv_path = sources_csv or DEFAULT_SOURCES_CSV
    pdf_dir = reports_dir or DEFAULT_REPORTS_DIR
    chunks_path = output_path or DEFAULT_CHUNKS_PATH
    summary_out = summary_path or DEFAULT_SUMMARY_PATH
    size = chunk_size if chunk_size is not None else settings.rag.chunk_size
    overlap = chunk_overlap if chunk_overlap is not None else settings.rag.chunk_overlap
    page_extractor = extractor or extract_pdf_pages

    rows = _load_source_rows(csv_path)
    documents: list[DocumentIngestRecord] = []
    chunks: list[TextChunk] = []
    skipped = 0

    for row in rows:
        document_id = (row.get("document_id") or "").strip()
        filename = (row.get("filename") or "").strip()
        relative = (row.get("local_relative_path") or "").strip()
        source_path = PROJECT_ROOT / relative if relative else pdf_dir / filename
        metadata = {
            "source_collection": row.get("source_collection"),
            "source_repository": row.get("source_repository"),
            "source_raw_url": row.get("source_raw_url"),
            "usefulness": row.get("usefulness"),
            "notes": row.get("notes"),
        }
        if not document_id or not filename:
            skipped += 1
            documents.append(
                DocumentIngestRecord(
                    document_id=document_id or "unknown",
                    source_path=str(source_path),
                    filename=filename,
                    status="skipped_incomplete_row",
                    metadata=metadata,
                )
            )
            continue
        if not source_path.exists() or source_path.stat().st_size == 0:
            skipped += 1
            documents.append(
                DocumentIngestRecord(
                    document_id=document_id,
                    source_path=str(source_path),
                    filename=filename,
                    status="skipped_missing_pdf",
                    error=f"PDF not found or empty: {source_path}",
                    metadata=metadata,
                )
            )
            continue
        try:
            raw_pages = page_extractor(source_path)
            page_chunks: list[TextChunk] = []
            for page_number, raw in raw_pages:
                cleaned = clean_text(raw)
                if not cleaned:
                    continue
                page_chunks.extend(
                    chunk_page_text(
                        text=cleaned,
                        document_id=document_id,
                        source_path=source_path,
                        page_number=page_number,
                        chunk_size=size,
                        chunk_overlap=overlap,
                        metadata=metadata,
                    )
                )
            documents.append(
                DocumentIngestRecord(
                    document_id=document_id,
                    source_path=str(source_path),
                    filename=filename,
                    n_pages=len(raw_pages),
                    n_chunks=len(page_chunks),
                    status="ok",
                    metadata=metadata,
                )
            )
            chunks.extend(page_chunks)
        except Exception as exc:  # noqa: BLE001 - isolate per-document failures
            skipped += 1
            documents.append(
                DocumentIngestRecord(
                    document_id=document_id,
                    source_path=str(source_path),
                    filename=filename,
                    status="error",
                    error=str(exc),
                    metadata=metadata,
                )
            )

    write_jsonl(chunks_path, [chunk.model_dump() for chunk in chunks])
    result = IngestResult(
        n_documents=len(documents),
        n_pages=sum(item.n_pages for item in documents),
        n_chunks=len(chunks),
        skipped=skipped,
        documents=documents,
        chunks=chunks,
        output_path=str(chunks_path),
    )
    summary_payload: dict[str, Any] = result.model_dump(exclude={"chunks"})
    write_json(summary_out, summary_payload)
    return result
