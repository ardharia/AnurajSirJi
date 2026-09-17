"""Building-level ground truth and evidence links dataset builder.

Generates:
1. data/metadata/building_links.csv - evidence-to-building links from canonical registry
2. data/processed/reference/building_ground_truth.csv - clean building-level reference dataset
3. data/processed/reference/building_ground_truth.jsonl - JSONL reference dataset with full provenance
"""

from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import Any

from seismic_damage.config.settings import PROJECT_ROOT
from seismic_damage.extraction.text_parser import parse_damage_text
from seismic_damage.io_utils import ensure_parent, write_jsonl
from seismic_damage.linking.catalog import build_catalog, load_image_rows
from seismic_damage.linking.linker import link_all
from seismic_damage.validation.metrics import ALL_PARAMETERS, is_missing
from seismic_damage.validation.normalize import (
    normalize_collapse_mode,
    normalize_crack_pattern,
    normalize_crack_severity,
    normalize_damage_grade,
    normalize_material_type,
    normalize_structural_system,
)

# Output paths
METADATA_DIR = PROJECT_ROOT / "data" / "metadata"
PROCESSED_REF_DIR = PROJECT_ROOT / "data" / "processed" / "reference"

BUILDING_LINKS_CSV = METADATA_DIR / "building_links.csv"
GROUND_TRUTH_CSV = PROCESSED_REF_DIR / "building_ground_truth.csv"
GROUND_TRUTH_JSONL = PROCESSED_REF_DIR / "building_ground_truth.jsonl"

_OCCUPANCY_KEYWORDS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"\b(?:residential|residence|apartments?|flats?|housing)\b", re.I), "residential"),
    (re.compile(r"\b(?:schools?|colleges?|institutes?|university)\b", re.I), "educational"),
    (re.compile(r"\b(?:hospitals?|dispensar(?:y|ies)|clinic)\b", re.I), "lifeline"),
    (re.compile(r"\b(?:commercial|offices?|shops?|market)\b", re.I), "commercial"),
    (re.compile(r"\b(?:industrial|factory|warehouse)\b", re.I), "industrial"),
)

_DAMAGE_GRADE_MAP: dict[int, dict[str, str]] = {
    0: {
        "structural_level": "none",
        "non_structural_level": "none",
        "retrofitting_action": "not_needed",
    },
    1: {
        "structural_level": "no_structural_damage",
        "non_structural_level": "slight",
        "retrofitting_action": "not_needed",
    },
    2: {
        "structural_level": "slight_structural",
        "non_structural_level": "moderate",
        "retrofitting_action": "non_structural_stabilization",
    },
    3: {
        "structural_level": "moderate_structural",
        "non_structural_level": "heavy",
        "retrofitting_action": "full_structural_and_non_structural_retrofit",
    },
    4: {
        "structural_level": "heavy_structural",
        "non_structural_level": "very_heavy",
        "retrofitting_action": "retrofit_or_replace",
    },
    5: {
        "structural_level": "very_heavy_structural",
        "non_structural_level": "total",
        "retrofitting_action": "replace_or_major_retrofit",
    },
}


def _extract_occupancy(text: str | None) -> str | None:
    if not text:
        return None
    for pattern, label in _OCCUPANCY_KEYWORDS:
        if pattern.search(text):
            return label
    return None


def _extract_plan_irregularity(text: str | None) -> bool | None:
    if not text:
        return None
    lowered = text.lower()
    if any(k in lowered for k in ("l-shaped", "l shaped", "u-shaped", "u shaped", "t-shaped", "irregular plan")):
        return True
    return None


def _extract_pre_eq_soft_story(text: str | None) -> bool | None:
    """Pre-earthquake soft story vulnerability feature (open ground storey design).

    Must be distinguished from post-earthquake soft-story collapse mechanism.
    """
    if not text:
        return None
    lowered = text.lower()
    if "open ground storey" in lowered or "open ground story" in lowered or "stilt" in lowered:
        return True
    return None


def export_building_links(output_path: Path | None = None) -> Path:
    """Export evidence-to-building links to data/metadata/building_links.csv."""
    path = ensure_parent(output_path or BUILDING_LINKS_CSV)
    image_rows = load_image_rows()
    catalog, links = link_all(image_rows=image_rows)

    fieldnames = [
        "evidence_id",
        "evidence_type",
        "building_id",
        "link_method",
        "reason",
        "filename",
        "source_collection",
        "caption",
        "original_image_url",
    ]

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for link in links:
            payload = link.payload or {}
            writer.writerow(
                {
                    "evidence_id": link.evidence_id,
                    "evidence_type": link.evidence_type,
                    "building_id": link.building_id or "",
                    "link_method": link.link_method or "",
                    "reason": link.reason,
                    "filename": payload.get("filename") or "",
                    "source_collection": payload.get("source_collection") or payload.get("_collection") or "",
                    "caption": payload.get("caption") or "",
                    "original_image_url": payload.get("original_image_url") or "",
                }
            )
    return path


def build_ground_truth_dataset(
    *,
    csv_path: Path | None = None,
    jsonl_path: Path | None = None,
) -> tuple[list[dict[str, Any]], Path, Path]:
    """Build the clean building-level reference dataset from canonical registry.

    Separates:
    - Building Typology (pre-earthquake)
    - Vulnerability / RVS Parameters (pre-earthquake)
    - Physical Damage Observations (post-earthquake)
    - Provenance & Evidence
    """
    target_csv = ensure_parent(csv_path or GROUND_TRUTH_CSV)
    target_jsonl = ensure_parent(jsonl_path or GROUND_TRUTH_JSONL)

    image_rows = load_image_rows()
    catalog = build_catalog(image_rows)

    # Index image rows by image_id for provenance
    rows_by_image = {r.get("image_id"): r for r in image_rows if r.get("image_id")}

    records: list[dict[str, Any]] = []

    for building in catalog:
        caption = building.caption or ""
        parsed = parse_damage_text(caption) if caption else {}

        # Gather provenance details from constituent images
        constituent_rows = [rows_by_image[iid] for iid in building.image_ids if iid in rows_by_image]
        source_pages = sorted({r.get("source_page", "") for r in constituent_rows if r.get("source_page")})
        image_urls = sorted({r.get("original_image_url", "") for r in constituent_rows if r.get("original_image_url")})
        filenames = sorted({r.get("filename", "") for r in constituent_rows if r.get("filename")})

        # 1. Building Typology (Pre-Earthquake)
        material_type = normalize_material_type(parsed.get("material_type"))
        structural_system = normalize_structural_system(parsed.get("structural_system"))
        number_of_stories = parsed.get("number_of_stories")
        occupancy = _extract_occupancy(caption)

        typology = {
            "material_type": material_type,
            "structural_system": structural_system,
            "number_of_stories": number_of_stories,
            "occupancy": occupancy,
        }

        # 2. Pre-Earthquake Vulnerability / RVS
        soft_story_vuln = _extract_pre_eq_soft_story(caption)
        plan_irreg_vuln = _extract_plan_irregularity(caption)

        vulnerability = {
            "soft_story": soft_story_vuln,
            "plan_irregularity": plan_irreg_vuln,
            "vertical_irregularity": None,
            "floating_columns": None,
            "short_columns": None,
            "pounding_risk": None,
            "soil_type": None,
        }

        # 3. Post-Earthquake Damage Observations
        damage_grade = normalize_damage_grade(parsed.get("damage_grade"))
        collapse_mode = normalize_collapse_mode(parsed.get("collapse_mode"))
        soft_story_failure = parsed.get("soft_story_failure")
        wall_failure = parsed.get("wall_failure")
        crack_severity = normalize_crack_severity(parsed.get("crack_severity"))
        crack_pattern = normalize_crack_pattern(parsed.get("crack_pattern"))

        dg_info = _DAMAGE_GRADE_MAP.get(damage_grade, {}) if damage_grade is not None else {}
        structural_damage_level = dg_info.get("structural_level")
        non_structural_damage_level = dg_info.get("non_structural_level")
        retrofitting_action = dg_info.get("retrofitting_action")

        damage = {
            "damage_grade": damage_grade,
            "collapse_mode": collapse_mode,
            "soft_story_failure": soft_story_failure,
            "wall_failure": wall_failure,
            "crack_severity": crack_severity,
            "crack_pattern": crack_pattern,
            "structural_damage_level": structural_damage_level,
            "non_structural_damage_level": non_structural_damage_level,
            "retrofitting_action": retrofitting_action,
            "pga": None,
            "magnitude": 7.7,  # Bhuj 2001 event magnitude
        }

        # Parameters block conforming to ALL_PARAMETERS
        parameters: dict[str, Any] = {}
        for block in (typology, vulnerability, damage):
            for k, v in block.items():
                if not is_missing(v) and k in ALL_PARAMETERS:
                    parameters[k] = v

        missing_parameters = sorted(ALL_PARAMETERS - set(parameters.keys()))

        record: dict[str, Any] = {
            "building_id": building.building_id,
            "earthquake_event": "Bhuj_2001",
            "source_collection": building.source_collection,
            "image_ids": building.image_ids,
            "n_images": len(building.image_ids),
            "filenames": filenames,
            "evidence_text": caption,
            "source_pages": source_pages,
            "image_urls": image_urls,
            "confidence": float(parsed.get("confidence") or 0.0),
            "typology": typology,
            "vulnerability": vulnerability,
            "damage": damage,
            "parameters": parameters,
            "missing_parameters": missing_parameters,
        }
        records.append(record)

    # Write CSV
    csv_columns = [
        "building_id",
        "earthquake_event",
        "source_collection",
        "image_ids",
        "n_images",
        "evidence_text",
        "confidence",
        # Typology
        "material_type",
        "structural_system",
        "number_of_stories",
        "occupancy",
        # Vulnerability
        "soft_story",
        "plan_irregularity",
        # Damage
        "damage_grade",
        "collapse_mode",
        "soft_story_failure",
        "wall_failure",
        "crack_severity",
        "crack_pattern",
        "structural_damage_level",
        "non_structural_damage_level",
        "retrofitting_action",
        "magnitude",
        "missing_parameters",
    ]

    with target_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=csv_columns)
        writer.writeheader()
        for r in records:
            row_dict = {
                "building_id": r["building_id"],
                "earthquake_event": r["earthquake_event"],
                "source_collection": r["source_collection"],
                "image_ids": ";".join(r["image_ids"]),
                "n_images": r["n_images"],
                "evidence_text": r["evidence_text"],
                "confidence": r["confidence"],
                "material_type": r["typology"]["material_type"] or "",
                "structural_system": r["typology"]["structural_system"] or "",
                "number_of_stories": r["typology"]["number_of_stories"] or "",
                "occupancy": r["typology"]["occupancy"] or "",
                "soft_story": "" if r["vulnerability"]["soft_story"] is None else str(r["vulnerability"]["soft_story"]),
                "plan_irregularity": "" if r["vulnerability"]["plan_irregularity"] is None else str(r["vulnerability"]["plan_irregularity"]),
                "damage_grade": "" if r["damage"]["damage_grade"] is None else str(r["damage"]["damage_grade"]),
                "collapse_mode": r["damage"]["collapse_mode"] or "",
                "soft_story_failure": "" if r["damage"]["soft_story_failure"] is None else str(r["damage"]["soft_story_failure"]),
                "wall_failure": "" if r["damage"]["wall_failure"] is None else str(r["damage"]["wall_failure"]),
                "crack_severity": r["damage"]["crack_severity"] or "",
                "crack_pattern": r["damage"]["crack_pattern"] or "",
                "structural_damage_level": r["damage"]["structural_damage_level"] or "",
                "non_structural_damage_level": r["damage"]["non_structural_damage_level"] or "",
                "retrofitting_action": r["damage"]["retrofitting_action"] or "",
                "magnitude": str(r["damage"]["magnitude"]),
                "missing_parameters": ";".join(r["missing_parameters"]),
            }
            writer.writerow(row_dict)

    # Write JSONL
    write_jsonl(target_jsonl, records)

    return records, target_csv, target_jsonl
