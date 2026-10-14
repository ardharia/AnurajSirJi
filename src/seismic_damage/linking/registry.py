"""Building Registry builder adhering strictly to canonical building identification and deduplication."""

from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import Any

from seismic_damage.config.settings import PROJECT_ROOT
from seismic_damage.linking.catalog import build_catalog
from seismic_damage.schemas.linking import BuildingRecord

REGISTRY_CSV = PROJECT_ROOT / "data" / "metadata" / "building_registry.csv"

# Non-building keywords from image/slideshow collections that are pure maps, title slides, dams, embankments, etc.
NON_BUILDING_PATTERNS = [
    r"\bmap detailing damage\b",
    r"\bplate tectonics\b",
    r"\bcauses of earthquakes\b",
    r"\bseismic zone map\b",
    r"\breferences http\b",
    r"\bthank 33 you\b",
    r"\bprepared by: gaurang\b",
    r"\bintroduction time & date\b",
    r"\bthe earthquake was felt in most parts\b",
    r"\b1819 gujarat\b",
    r"\bsite of destruction of the villages\b",
    r"\bsurajbadi sustained severe damage\b",
    r"\bearth dams damages on tappar dam\b",
    r"\bfailure of the upstream slope of the fatehgadh dam\b",
    r"\brail,road and highway embankments\b",
    r"\bdamage to highway pavement\b",
    r"\bgeotechnical damages\b",
    r"\ban indian army helicopter flies over a salt marsh\b",
    r"\bcracks on a salt marsh\b",
    r"\baerial view of the epicenter.*crater\b",
    r"\baerial view of the epicentre.*huge crack\b",
    r"\baerial view of the epicenter.*huge crack\b",
    r"\bsurvivors.*place their hands into\b",
    r"\bstatue of mahatma gandhi\b",
    r"\bquake floods pakistan sugarcane farms\b",
    r"\bcars slowed down, alerting one\b",
    r"\bkandla, india's largest port, wears a deserted look\b",
    r"\bsuresh bhai.*puts his hand into a crater\b",
    r"\bmohan bhai.*inserts a stick into a crater\b",
    r"\bnewly built checkposts\b",
]


def classify_typology(text: str) -> tuple[str, float]:
    """Determine building typology strictly from evidence text."""
    lowered = text.lower()

    # RRSM: Random Rubble Stone Masonry
    if any(k in lowered for k in [
        "random rubble", "rubble stone", "rubble masonry", "rrsm",
        "stone masonry", "block stone masonry", "stone houses", "rubble stone masonary"
    ]):
        return "RRSM", 0.90

    # BCBM: Burnt Clay Brick Masonry
    if any(k in lowered for k in [
        "brick masonry", "burnt clay brick", "brick wall", "load-bearing brick",
        "load bearing brick", "bcbm", "brick chimney"
    ]):
        return "BCBM", 0.90

    # CM: Confined Masonry
    if ("confined masonry" in lowered or ("tie column" in lowered and "masonry" in lowered)) and "prestressed" not in lowered:
        return "CM", 0.90

    # MH: Mud House
    if any(k in lowered for k in [
        "mud house", "mud wall", "mud houses", "mud mortar", "adobe", "kutcha"
    ]) and not any(k in lowered for k in ["rubble", "stone"]):
        return "MH", 0.85

    # DD: Dhajji Diwari
    if any(k in lowered for k in ["dhajji", "dhajji diwari", "timber frame", "wood frame"]):
        return "DD", 0.85

    # EH: Ekra House
    if any(k in lowered for k in ["ekra", "ekra house", "bamboo"]):
        return "EH", 0.85

    # RC: Reinforced Concrete
    if any(k in lowered for k in [
        "rc frame", "rc framed", "reinforced concrete", "rc building", "rc multistory",
        "open ground storey", "open ground story", "soft storey", "soft story",
        "floating column", "cantilever stub beam", "stirrup", "stirrups",
        "pancake collapse", "flat slab", "elevator core", "10-storey", "4-storey",
        "6-storey", "14-storey", "8-storey", "two-storey rc", "ten-story", "four-storey",
        "four-story", "apartment complex", "apartment block"
    ]):
        return "RC", 0.90

    if "masonry" in lowered:
        return "BCBM", 0.50

    return "UNKNOWN", 0.0


def _get_tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def build_building_registry(
    catalog: list[BuildingRecord] | None = None,
    output_path: Path = REGISTRY_CSV,
) -> list[dict[str, Any]]:
    """Build unique, stable building registry with deduplication and typology classification."""
    cat = catalog if catalog is not None else build_catalog()

    # Deduplicate / cluster identical or near-identical buildings across sources
    clusters: list[list[BuildingRecord]] = []
    for b in cat:
        cap = (b.caption or "").strip()
        tokens = _get_tokens(cap)
        matched_idx: int | None = None
        if tokens:
            for c_idx, cluster in enumerate(clusters):
                first_cap = cluster[0].caption or ""
                first_tokens = _get_tokens(first_cap)
                jaccard = (
                    len(tokens & first_tokens) / len(tokens | first_tokens)
                    if (tokens | first_tokens)
                    else 0.0
                )
                if jaccard >= 0.70:
                    matched_idx = c_idx
                    break
        if matched_idx is not None:
            clusters[matched_idx].append(b)
        else:
            clusters.append([b])

    registry_records: list[dict[str, Any]] = []
    for i, cluster in enumerate(clusters):
        bid = f"BHJ_B{i+1:03d}"
        primary = cluster[0]
        caption = primary.caption or ""

        # Extract page number or location if available
        meta = primary.metadata or {}
        page_no = meta.get("page_number", "")

        # All collections in the cluster
        sources = sorted(set(b.source_collection for b in cluster if b.source_collection))
        source_doc = "; ".join(sources) if sources else "bhuj_2001_reports"

        # Check if non-building
        cap_l = caption.lower()
        is_non_bldg = any(re.search(pat, cap_l) for pat in NON_BUILDING_PATTERNS)
        if is_non_bldg:
            typology = "NON_BUILDING"
            conf = 0.0
        else:
            typology, conf = classify_typology(caption)

        # Merge evidence text across cluster members if distinct
        evidence_pieces = []
        for b in cluster:
            if b.caption and b.caption.strip() not in evidence_pieces:
                evidence_pieces.append(b.caption.strip())
        evidence_text = " | ".join(evidence_pieces)

        rec = {
            "building_id": bid,
            "event_id": "BHUJ_2001",
            "event_name": "Bhuj Earthquake 2001",
            "source_document": source_doc,
            "page_number": str(page_no) if page_no else "N/A",
            "building_typology": typology,
            "typology_confidence": f"{conf:.2f}",
            "building_description": caption[:120],
            "evidence_text": evidence_text,
        }
        registry_records.append(rec)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "building_id",
                "event_id",
                "event_name",
                "source_document",
                "page_number",
                "building_typology",
                "typology_confidence",
                "building_description",
                "evidence_text",
            ],
        )
        writer.writeheader()
        writer.writerows(registry_records)

    return registry_records


if __name__ == "__main__":
    records = build_building_registry()
    print(f"Created building registry with {len(records)} buildings at {REGISTRY_CSV}")
    typs: dict[str, int] = {}
    for r in records:
        typs[r["building_typology"]] = typs.get(r["building_typology"], 0) + 1
    print("Typology breakdown:", typs)
