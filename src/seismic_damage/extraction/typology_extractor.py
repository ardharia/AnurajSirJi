# src/seismic_damage/extraction/typology_extractor.py
"""Extract typology‑wise parameters from the NDMA RVS primer PDF.

The module reads the PDF, extracts cleaned text per page, detects typology
headings (e.g. "Typology A") and collects the bullet‑point parameters that
follow each heading. For each parameter we preserve the original wording and
the page number on which it appears. The result is written as one JSON file
per typology under ``data/typology_parameters``.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import List, Tuple, Dict, Any

from seismic_damage.ingestion.pdf import extract_pdf_pages
from seismic_damage.ingestion.clean import clean_text


def _pdf_to_pages(pdf_path: Path) -> List[Tuple[int, str]]:
    """Extract cleaned text per page, preserving page numbers.

    Returns a list of ``(page_number, cleaned_text)`` tuples.
    """
    pages = extract_pdf_pages(pdf_path)
    cleaned: List[Tuple[int, str]] = []
    for page_number, raw in pages:
        txt = clean_text(raw)
        if txt:
            cleaned.append((page_number, txt))
    return cleaned


def _parse_typologies(pages: List[Tuple[int, str]]) -> Dict[str, List[Dict[str, Any]]]:
    """Parse typology headings and associate parameters with page numbers.

    Returns a mapping ``typology_name -> list of {"parameter": str, "page": int}``.
    """
    typology_pattern = re.compile(r"^Typology\s+([A-Za-z0-9_-]+)", re.IGNORECASE)
    bullet_pattern = re.compile(r"^[\-*]\s+(.*)")
    typologies: Dict[str, List[Dict[str, Any]]] = {}
    current: str | None = None
    current_page: int | None = None
    for page_number, page_text in pages:
        for line in page_text.splitlines():
            line = line.strip()
            if not line:
                continue
            m = typology_pattern.match(line)
            if m:
                current = m.group(1).strip()
                current_page = page_number
                typologies[current] = []
                continue
            if current is None:
                continue
            b = bullet_pattern.match(line)
            if b:
                param = b.group(1).strip()
                typologies[current].append({"parameter": param, "page": current_page})
    return typologies


def extract_typology_params(rvs_pdf_path: Path, out_dir: Path) -> None:
    """Parse the NDMA RVS primer PDF and write JSON per typology.

    Each JSON file contains a list of objects with ``parameter`` and ``page``
    fields, preserving the official terminology and page reference.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    pages = _pdf_to_pages(rvs_pdf_path)
    typology_map = _parse_typologies(pages)
    for typology, entries in typology_map.items():
        out_path = out_dir / f"{typology}.json"
        out_path.write_text(
            json.dumps(entries, indent=2, ensure_ascii=False), encoding="utf-8"
        )


if __name__ == "__main__":
    import sys
    if len(sys.argv) != 3:
        print("Usage: python -m seismic_damage.extraction.typology_extractor <pdf_path> <out_dir>")
        sys.exit(1)
    extract_typology_params(Path(sys.argv[1]), Path(sys.argv[2]))
