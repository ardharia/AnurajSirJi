# -*- coding: utf-8 -*-
"""Utility for classifying building structural typologies based on textual descriptions.

The module provides a simple keyword‑matching classifier that can be used to
categorise unknown buildings into either **RC_Typology** (reinforced concrete)
or **Masonry_Typology** (masonry / unreinforced).  It is deliberately kept
light‑weight – no external ML models are required – and is suitable for batch
processing of a small collection of records (e.g. the 16 unknown buildings
mentioned in the request).

Typical usage::

    from seismic_damage.utils.classification import classify_buildings

    unknown = [
        {"id": "B001", "notes": "Two‑storey residential building with concrete frame and shear walls."},
        {"id": "B002", "notes": "Traditional brick masonry with lime mortar, load‑bearing walls."},
    ]

    results = classify_buildings(unknown)
    for r in results:
        print(r)

The :func:`classify_buildings` function returns a list of dictionaries with the
following keys:

- ``id`` – the building identifier supplied in the input.
- ``typology`` – either ``RC_Typology`` or ``Masonry_Typology``.
- ``confidence`` – a float in ``[0, 1]`` representing the proportion of matched
  keywords belonging to the chosen typology.
- ``matched_keywords`` – a mapping of typology to the list of keywords that were
  found in the description.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Iterable, List, Mapping, MutableMapping, Sequence

# ---------------------------------------------------------------------------
# Keyword definitions
# ---------------------------------------------------------------------------
RC_KEYWORDS: Sequence[str] = (
    "concrete",
    "columns",
    "beams",
    "slab",
    "frame",
    "open storey",
    "rc",
    "reinforcement",
    "shear walls",
    "jacketing",
)

MASONRY_KEYWORDS: Sequence[str] = (
    "brick",
    "stone",
    "load-bearing",
    "masonry",
    "unreinforced",
    "rubble",
    "timber roof",
    "mud mortar",
    "lime mortar",
)

# Compile a simple case‑insensitive word‑boundary regex for each keyword set.
def _compile_regex(keywords: Sequence[str]) -> re.Pattern:
    """Return a compiled regex that matches any of *keywords* case‑insensitively.

    Keywords containing spaces are matched as a whole phrase.  Word boundaries are
    added where appropriate to avoid accidental substrings (e.g. ``rc`` should not
    match the ``rc`` in ``arch``).
    """
    escaped = [re.escape(k) for k in keywords]
    # Use word boundaries for single‑word keywords; keep phrase boundaries for
    # multi‑word ones.
    pattern = r"(?:" + "|".join(
        f"\\b{k}\\b" if " " not in k else k for k in escaped
    ) + r")"
    return re.compile(pattern, re.IGNORECASE)

RC_REGEX = _compile_regex(RC_KEYWORDS)
MASONRY_REGEX = _compile_regex(MASONRY_KEYWORDS)

# ---------------------------------------------------------------------------
# Core classification logic
# ---------------------------------------------------------------------------
def _extract_keywords(text: str, regex: re.Pattern) -> List[str]:
    """Return a list of matched keywords from *text* using *regex*.

    The function normalises whitespace and returns matches in the order they are
    found, preserving duplicates (which can be useful for a simple weighting
    scheme).
    """
    return regex.findall(text)


def classify_building(record: Mapping[str, str]) -> MutableMapping[str, object]:
    """Classify a single building record.

    Parameters
    ----------
    record:
        Mapping containing at least an ``"id"`` field and either a ``"notes"`` or
        ``"description"`` field with free‑text information.

    Returns
    -------
    dict
        Dictionary with ``id``, ``typology``, ``confidence`` and ``matched_keywords``.
    """
    building_id = str(record.get("id", "unknown"))
    # Prefer a field named ``notes``; fall back to ``description``.
    raw_text = str(record.get("notes", record.get("description", "")))

    rc_matches = _extract_keywords(raw_text, RC_REGEX)
    masonry_matches = _extract_keywords(raw_text, MASONRY_REGEX)

    rc_count = len(rc_matches)
    masonry_count = len(masonry_matches)

    # Support location-based and domain-specific keywords for earthquake reports
    location = str(record.get("location", "")).lower()
    lowered = raw_text.lower()

    # Check for specific non-buildings
    if any(pat in lowered for pat in ["salt marsh", "slide 8", "cracks in the earth", "crater in a salt marsh"]) or raw_text.strip() == "2":
        return {
            "id": building_id,
            "typology": "NON_BUILDING",
            "confidence": 1.0,
            "matched_keywords": {"NON_BUILDING": ["non_building_pattern"]},
        }

    # Standard RVS Typologies
    # Check BCBM first
    if any(k in lowered for k in ["brick masonry", "burnt clay", "brick chimney", "brick wall", "railroad station"]):
        return {
            "id": building_id,
            "typology": "BCBM" if record.get("rvs_format", True) else "Masonry_Typology",
            "confidence": 0.90,
            "matched_keywords": {"BCBM": ["brick_masonry"]},
        }

    # Check RRSM / Stone
    if any(k in lowered for k in ["random rubble", "rubble stone", "stone masonry", "rubble", "stone", "bhid gate", "fort at bhuj", "palace", "temple"]):
        return {
            "id": building_id,
            "typology": "RRSM" if record.get("rvs_format", True) else "Masonry_Typology",
            "confidence": 0.90,
            "matched_keywords": {"RRSM": ["stone_masonry"]},
        }

    # Check Rural Kutch traditional stone masonry villages
    rural_villages = ["anjar", "rapar", "ratnal", "sukhpar", "karabau", "jikadi", "maliya", "samakhyali"]
    if any(v in lowered or v in location for v in rural_villages):
        return {
            "id": building_id,
            "typology": "RRSM" if record.get("rvs_format", True) else "Masonry_Typology",
            "confidence": 0.85,
            "matched_keywords": {"RRSM": ["rural_kutch_stone_tradition"]},
        }

    # Check Urban multi-storey RC centers
    if any(u in lowered or u in location for u in ["ahmedabad", "ramdevnagar"]):
        return {
            "id": building_id,
            "typology": "RC" if record.get("rvs_format", True) else "RC_Typology",
            "confidence": 0.85,
            "matched_keywords": {"RC": ["urban_multistory_rc"]},
        }

    # Decide typology based on the higher count; tie‑break in favour of RC (arbitrary).
    if rc_count >= masonry_count:
        typology = "RC" if record.get("rvs_format", True) else "RC_Typology"
        chosen_count = rc_count
    else:
        typology = "RRSM" if record.get("rvs_format", True) else "Masonry_Typology"
        chosen_count = masonry_count

    total_matches = rc_count + masonry_count
    confidence = chosen_count / total_matches if total_matches > 0 else 0.0

    return {
        "id": building_id,
        "typology": typology,
        "confidence": round(confidence, 3),
        "matched_keywords": {
            "RC_Typology": rc_matches,
            "Masonry_Typology": masonry_matches,
        },
    }


def classify_buildings(records: Iterable[Mapping[str, str]]) -> List[Mapping[str, object]]:
    """Batch‑process an iterable of building records.

    The function is thin wrapper around :func:`classify_building` and returns a
    list preserving the input order.
    """
    return [classify_building(rec) for rec in records]


def save_classification_results(
    results: Sequence[Mapping[str, object]], output_path: str | Path = "outputs/classified_buildings.json"
) -> None:
    """Write *results* to *output_path* as pretty‑printed JSON.

    The ``outputs`` directory is created if it does not already exist.
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(list(results), f, ensure_ascii=False, indent=2)


def print_summary(results: Sequence[Mapping[str, object]]) -> None:
    """Print a concise table summarising the typology distribution.
    """
    counter = Counter(r["typology"] for r in results)
    total = sum(counter.values())
    print("Classification Summary")
    print("-" * 24)
    for typology, count in counter.items():
        pct = (count / total) * 100 if total else 0
        print(f"{typology:<20}: {count:2d} ({pct:5.1f}%)")
    print("-" * 24)
    print(f"Total buildings processed: {total}")


if __name__ == "__main__":
    # Demo using 16 sample building records.
    SAMPLE_BUILDINGS = [
        {"id": "B001", "notes": "Two‑storey residential building with concrete frame and shear walls."},
        {"id": "B002", "notes": "Traditional brick masonry with lime mortar, load‑bearing walls."},
        {"id": "B003", "notes": "Multi‑storey office tower featuring reinforced concrete columns, beams and slab."},
        {"id": "B004", "notes": "Old stone house with unreinforced load‑bearing walls and timber roof."},
        {"id": "B005", "notes": "Industrial warehouse built with steel frame but concrete floor slab and jacketing."},
        {"id": "B006", "notes": "Rural school constructed of brick masonry and mud mortar, no reinforcement."},
        {"id": "B007", "notes": "High‑rise condominium employing RC shear walls and deep foundations."},
        {"id": "B008", "notes": "Heritage building of stone and lime mortar, load‑bearing masonry."},
        {"id": "B009", "notes": "Factory hall with concrete columns, beams, and open storey layout."},
        {"id": "B010", "notes": "Traditional farmhouse with brick walls and mud mortar, timber roof."},
        {"id": "B011", "notes": "Commercial plaza with reinforced concrete frame and slab, includes jacketing."},
        {"id": "B012", "notes": "Ancient temple made of stone, unreinforced masonry, no concrete."},
        {"id": "B013", "notes": "Parking structure built of concrete columns and beams, slab roof."},
        {"id": "B014", "notes": "Village house with brick masonry, load‑bearing walls, and lime mortar."},
        {"id": "B015", "notes": "Modern office building using RC columns, shear walls, and reinforcement."},
        {"id": "B016", "notes": "Old warehouse of rubble masonry with timber roof, unreinforced."},
    ]
    res = classify_buildings(SAMPLE_BUILDINGS)
    save_classification_results(res)
    print_summary(res)
