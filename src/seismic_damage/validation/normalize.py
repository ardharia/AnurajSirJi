"""RAG, VLM, and Ground Truth schema normalization.

Normalizes heterogeneous outputs from independent pipelines to a canonical
building + parameter level schema conforming to the reference framework
(NDMA RVS Primer, IS 13935:2009, EMS-98).

Rules:
1. RAG and VLM outputs remain independent — predictions are NEVER fused.
2. Original raw outputs are preserved intact in `raw_output`.
3. Post-earthquake damage observations (e.g. soft-story collapse, wall failure)
   are NEVER converted into pre-earthquake vulnerability parameters.
4. Missing values are explicitly marked None / missing; never invented.
5. All normalized parameter names and allowed values match the reference dictionary.
"""

from __future__ import annotations

import re
from typing import Any, Mapping

from seismic_damage.schemas.normalized import (
    DamageAttributes,
    NormalizedAssessmentRecord,
    ProvenanceInfo,
    TypologyAttributes,
    VulnerabilityAttributes,
)
from seismic_damage.validation.metrics import is_missing


# ---------------------------------------------------------------------------
# Controlled Vocabulary Mappings (per reference/parameter_dictionary.csv)
# ---------------------------------------------------------------------------

_MATERIAL_MAP: dict[str, str] = {
    "reinforced_concrete": "reinforced_concrete",
    "reinforced concrete": "reinforced_concrete",
    "rc": "reinforced_concrete",
    "r.c.": "reinforced_concrete",
    "r.c": "reinforced_concrete",
    "concrete": "reinforced_concrete",
    "rc frame": "reinforced_concrete",
    "rc_frame": "reinforced_concrete",
    "rc framed": "reinforced_concrete",
    "rc_framed": "reinforced_concrete",
    "burnt_clay_brick": "burnt_clay_brick",
    "brick": "burnt_clay_brick",
    "burnt brick": "burnt_clay_brick",
    "clay brick": "burnt_clay_brick",
    "brick masonry": "burnt_clay_brick",
    "masonry": "burnt_clay_brick",
    "cement_block": "cement_block",
    "cc block": "cement_block",
    "concrete block": "cement_block",
    "stone_block": "stone_block",
    "stone": "stone_block",
    "rubble": "stone_block",
    "stone masonry": "stone_block",
    "random rubble": "stone_block",
    "rrsm": "stone_block",
    "mud": "mud",
    "adobe": "mud",
    "earthen": "mud",
    "rammed earth": "mud",
    "pise": "mud",
    "steel": "steel",
    "structural steel": "steel",
    "timber": "timber",
    "wood": "timber",
    "wooden": "timber",
    "bamboo": "bamboo",
    "mixed": "mixed",
    "composite": "mixed",
}

_SYSTEM_MAP: dict[str, str] = {
    "moment_frame": "moment_frame",
    "mrf": "moment_frame",
    "moment resisting frame": "moment_frame",
    "moment frame": "moment_frame",
    "rc_frame": "rc_frame",
    "rc frame": "rc_frame",
    "reinforced concrete frame": "rc_frame",
    "shear_wall": "shear_wall",
    "shear wall": "shear_wall",
    "sw": "shear_wall",
    "rc wall": "shear_wall",
    "structural wall": "shear_wall",
    "moment_frame_with_shear_walls": "moment_frame_with_shear_walls",
    "bearing_wall": "bearing_wall",
    "bearing wall": "bearing_wall",
    "load bearing": "bearing_wall",
    "load-bearing": "bearing_wall",
    "load bearing masonry": "bearing_wall",
    "urm": "bearing_wall",
    "confined_masonry": "confined_masonry",
    "confined masonry": "confined_masonry",
    "cm": "confined_masonry",
    "flat_slab": "flat_slab",
    "flat slab": "flat_slab",
    "flat plate": "flat_slab",
    "precast": "precast",
    "pc": "precast",
    "timber_frame": "timber_frame",
    "timber frame": "timber_frame",
    "dhajji": "timber_frame",
    "dhajji dewari": "timber_frame",
    "dhajji diwari": "timber_frame",
    "bamboo_frame": "bamboo_frame",
    "bamboo frame": "bamboo_frame",
    "ekra": "bamboo_frame",
    "ekra house": "bamboo_frame",
}

_SEVERITY_MAP: dict[str, str] = {
    "none": "none",
    "hairline": "hairline",
    "fine": "hairline",
    "minor": "hairline",
    "slight": "hairline",
    "thin": "hairline",
    "moderate": "moderate",
    "medium": "moderate",
    "severe": "severe",
    "heavy": "severe",
    "serious": "severe",
    "extensive": "extensive",
    "very heavy": "extensive",
    "major": "extensive",
    "wide": "extensive",
}

_PATTERN_MAP: dict[str, str] = {
    "none": "none",
    "vertical": "vertical",
    "horizontal": "horizontal",
    "diagonal": "diagonal",
    "x_shaped": "x_shaped",
    "x-shaped": "x_shaped",
    "x shaped": "x_shaped",
    "cross": "x_shaped",
    "stair_step": "stair_step",
    "stair-step": "stair_step",
    "stair step": "stair_step",
    "stepped": "stair_step",
    "flexural": "flexural",
    "flexure": "flexural",
    "shear": "shear",
}

_COLLAPSE_MAP: dict[str, str] = {
    "none": "none",
    "partial": "partial",
    "partial collapse": "partial",
    "portion": "partial",
    "one-half": "partial",
    "total": "total",
    "total collapse": "total",
    "complete": "total",
    "complete collapse": "total",
    "destruction": "total",
    "flattened": "total",
    "rubble": "total",
    "soft_story": "soft_story",
    "soft story": "soft_story",
    "open ground storey": "soft_story",
    "ground storey collapse": "soft_story",
    "out_of_plane": "out_of_plane",
    "out of plane": "out_of_plane",
    "in_plane": "in_plane",
    "in plane": "in_plane",
    "pancake": "pancake",
    "pancake collapse": "pancake",
}

_SOIL_MAP: dict[str, str] = {
    "hard": "hard",
    "rock": "hard",
    "stiff": "hard",
    "medium": "medium",
    "soft": "soft",
    "liquefiable": "liquefiable",
    "liquefaction": "liquefiable",
}


# ---------------------------------------------------------------------------
# Individual Value Normalizers
# ---------------------------------------------------------------------------

def normalize_material_type(raw: Any) -> str | None:
    """Normalize material type to reference dictionary allowed values."""
    if is_missing(raw):
        return None
    cleaned = str(raw).strip().lower()
    return (
        _MATERIAL_MAP.get(cleaned)
        or _MATERIAL_MAP.get(cleaned.replace(" ", "_"))
        or _MATERIAL_MAP.get(cleaned.replace("_", " "))
    )


def normalize_structural_system(raw: Any) -> str | None:
    """Normalize structural system to reference dictionary allowed values."""
    if is_missing(raw):
        return None
    cleaned = str(raw).strip().lower()
    return _SYSTEM_MAP.get(cleaned)


def normalize_damage_grade(raw: Any) -> int | None:
    """Normalize damage grade to integer 0..5 per EMS-98 / IS 13935."""
    if is_missing(raw):
        return None
    if isinstance(raw, (int, float)):
        val = int(raw)
        return val if 0 <= val <= 5 else None
    text = str(raw).strip()
    match = re.search(r"(?:grade|g)?\s*([0-5])\b", text, re.I)
    if match:
        return int(match.group(1))
    return None


def normalize_crack_severity(raw: Any) -> str | None:
    """Normalize crack severity to reference dictionary allowed values."""
    if is_missing(raw):
        return None
    cleaned = str(raw).strip().lower()
    return _SEVERITY_MAP.get(cleaned)


def normalize_crack_pattern(raw: Any) -> str | None:
    """Normalize crack pattern to reference dictionary allowed values."""
    if is_missing(raw):
        return None
    cleaned = str(raw).strip().lower()
    return _PATTERN_MAP.get(cleaned)


def normalize_collapse_mode(raw: Any) -> str | None:
    """Normalize collapse mode to reference dictionary allowed values."""
    if is_missing(raw):
        return None
    cleaned = str(raw).strip().lower()
    return _COLLAPSE_MAP.get(cleaned)


def normalize_soil_type(raw: Any) -> str | None:
    """Normalize soil type to reference dictionary allowed values."""
    if is_missing(raw):
        return None
    cleaned = str(raw).strip().lower()
    return _SOIL_MAP.get(cleaned)


def normalize_binary(raw: Any) -> bool | None:
    """Normalize binary flags (soft_story_failure, wall_failure, etc.)."""
    if is_missing(raw):
        return None
    if isinstance(raw, bool):
        return raw
    if isinstance(raw, (int, float)):
        if raw in (1, 1.0):
            return True
        if raw in (0, 0.0):
            return False
        return None
    text = str(raw).strip().lower()
    if text in {"true", "yes", "1"}:
        return True
    if text in {"false", "no", "0"}:
        return False
    return None


def normalize_number_of_stories(raw: Any) -> int | None:
    """Normalize number of stories to positive integer."""
    if is_missing(raw):
        return None
    try:
        val = int(float(str(raw).strip()))
        return val if val >= 1 else None
    except (ValueError, TypeError):
        return None


def normalize_float(raw: Any, *, min_val: float | None = None) -> float | None:
    """Normalize non-negative float values (PGA, magnitude, etc.)."""
    if is_missing(raw):
        return None
    try:
        val = float(str(raw).strip())
        if min_val is not None and val < min_val:
            return None
        return val
    except (ValueError, TypeError):
        return None


# ---------------------------------------------------------------------------
# High-Level Record Normalizers
# ---------------------------------------------------------------------------

def _extract_flat_dict(source: Any) -> dict[str, Any]:
    """Convert heterogeneous input (dict, Pydantic model, Mapping) to flat dict."""
    if hasattr(source, "model_dump"):
        return source.model_dump()
    if isinstance(source, Mapping):
        return dict(source)
    return {}


def normalize_rag_output(
    rag_payload: Any,
    *,
    building_id: str | None = None,
) -> NormalizedAssessmentRecord:
    """Normalize RAG pipeline output to canonical schema.

    Preserves raw RAG output and citations. Never fuses with VLM.
    """
    raw = _extract_flat_dict(rag_payload)
    bid = str(building_id or raw.get("building_id") or "unknown")

    # Parameters can be at top level or in nested parameters dict
    params = raw.get("parameters") if isinstance(raw.get("parameters"), Mapping) else {}

    def get_field(key: str, alt_key: str | None = None) -> Any:
        v = raw.get(key)
        if is_missing(v) and alt_key:
            v = raw.get(alt_key)
        if is_missing(v):
            v = params.get(key)
        if is_missing(v) and alt_key:
            v = params.get(alt_key)
        return v

    typology = TypologyAttributes(
        material_type=normalize_material_type(get_field("material_type", "material")),
        structural_system=normalize_structural_system(get_field("structural_system")),
        number_of_stories=normalize_number_of_stories(get_field("number_of_stories", "storeys")),
        occupancy=get_field("occupancy"),
    )

    # Vulnerability: strictly pre-earthquake
    vulnerability = VulnerabilityAttributes(
        soft_story=normalize_binary(get_field("soft_story")),
        floating_columns=normalize_binary(get_field("floating_columns")),
        plan_irregularity=normalize_binary(get_field("plan_irregularity")),
        vertical_irregularity=normalize_binary(get_field("vertical_irregularity")),
        short_columns=normalize_binary(get_field("short_columns")),
        soil_type=normalize_soil_type(get_field("soil_type")),
    )

    # Damage: strictly post-earthquake
    damage = DamageAttributes(
        damage_grade=normalize_damage_grade(get_field("damage_grade")),
        crack_severity=normalize_crack_severity(get_field("crack_severity")),
        crack_pattern=normalize_crack_pattern(get_field("crack_pattern")),
        collapse_mode=normalize_collapse_mode(get_field("collapse_mode")),
        soft_story_failure=normalize_binary(get_field("soft_story_failure")),
        wall_failure=normalize_binary(get_field("wall_failure")),
        pga=normalize_float(get_field("pga"), min_val=0.0),
        magnitude=normalize_float(get_field("magnitude"), min_val=0.0),
    )

    citations = raw.get("citations") if isinstance(raw.get("citations"), list) else []
    provenance = ProvenanceInfo(
        source_modality="rag",
        source_collection="rag_corpus",
        citations=citations,
        evidence_text=str(raw.get("evidence_text") or raw.get("answer") or ""),
        notes=f"RAG evidence sufficient: {raw.get('rag_evidence_sufficient', raw.get('evidence_sufficient', False))}",
    )

    return NormalizedAssessmentRecord(
        building_id=bid,
        earthquake_event=str(raw.get("earthquake_event") or "Bhuj_2001"),
        modality="rag",
        typology=typology,
        vulnerability=vulnerability,
        damage=damage,
        provenance=provenance,
        confidence=float(raw.get("confidence") or 0.0),
        raw_output=raw,
    )


def normalize_vlm_output(
    vlm_payload: Any,
    *,
    building_id: str | None = None,
) -> NormalizedAssessmentRecord:
    """Normalize VLM pipeline output to canonical schema.

    Preserves raw VLM output and visual notes. Never fuses with RAG.
    """
    raw = _extract_flat_dict(vlm_payload)
    bid = str(building_id or raw.get("building_id") or "unknown")

    params = raw.get("parameters") if isinstance(raw.get("parameters"), Mapping) else {}

    def get_field(key: str, alt_key: str | None = None) -> Any:
        v = raw.get(key)
        if is_missing(v) and alt_key:
            v = raw.get(alt_key)
        if is_missing(v):
            v = params.get(key)
        if is_missing(v) and alt_key:
            v = params.get(alt_key)
        return v

    typology = TypologyAttributes(
        material_type=normalize_material_type(get_field("material_type", "material")),
        structural_system=normalize_structural_system(get_field("structural_system")),
        number_of_stories=normalize_number_of_stories(get_field("number_of_stories")),
    )

    # Vulnerability: strictly pre-earthquake
    vulnerability = VulnerabilityAttributes(
        soft_story=normalize_binary(get_field("soft_story")),
        soil_type=normalize_soil_type(get_field("soil_type")),
    )

    # Damage: strictly post-earthquake
    damage = DamageAttributes(
        damage_grade=normalize_damage_grade(get_field("damage_grade")),
        crack_severity=normalize_crack_severity(get_field("crack_severity")),
        crack_pattern=normalize_crack_pattern(get_field("crack_pattern")),
        collapse_mode=normalize_collapse_mode(get_field("collapse_mode")),
        soft_story_failure=normalize_binary(get_field("soft_story_failure")),
        wall_failure=normalize_binary(get_field("wall_failure")),
    )

    img_path = str(raw.get("image_path") or "")
    image_paths = [img_path] if img_path else []

    provenance = ProvenanceInfo(
        source_modality="vlm",
        source_collection=raw.get("source_collection"),
        image_paths=image_paths,
        evidence_text=raw.get("evidence"),
        notes=raw.get("notes"),
    )

    return NormalizedAssessmentRecord(
        building_id=bid,
        earthquake_event=str(raw.get("earthquake_event") or "Bhuj_2001"),
        modality="vlm",
        typology=typology,
        vulnerability=vulnerability,
        damage=damage,
        provenance=provenance,
        confidence=float(raw.get("confidence") or 0.0),
        raw_output=raw,
    )


def normalize_ground_truth(
    gt_payload: Any,
    *,
    building_id: str | None = None,
) -> NormalizedAssessmentRecord:
    """Normalize Ground Truth record to canonical schema."""
    raw = _extract_flat_dict(gt_payload)
    bid = str(building_id or raw.get("building_id") or "unknown")

    params = raw.get("parameters") if isinstance(raw.get("parameters"), Mapping) else {}

    def get_field(key: str, alt_key: str | None = None) -> Any:
        v = raw.get(key)
        if is_missing(v) and alt_key:
            v = raw.get(alt_key)
        if is_missing(v):
            v = params.get(key)
        if is_missing(v) and alt_key:
            v = params.get(alt_key)
        return v

    typology = TypologyAttributes(
        material_type=normalize_material_type(get_field("material_type", "typology_material")),
        structural_system=normalize_structural_system(get_field("structural_system", "typology_structural_system")),
        number_of_stories=normalize_number_of_stories(get_field("number_of_stories", "typology_number_of_stories")),
        occupancy=get_field("occupancy", "typology_occupancy"),
        ndma_typology=get_field("ndma_typology", "typology_ndma_code"),
    )

    vulnerability = VulnerabilityAttributes(
        soft_story=normalize_binary(get_field("soft_story", "vuln_soft_story")),
        plan_irregularity=normalize_binary(get_field("plan_irregularity", "vuln_plan_irregularity")),
        vertical_irregularity=normalize_binary(get_field("vertical_irregularity", "vuln_vertical_irregularity")),
        soil_type=normalize_soil_type(get_field("soil_type", "vuln_soil_type")),
        ems98_vulnerability_class=get_field("ems98_vulnerability_class", "vuln_ems98_class"),
    )

    damage = DamageAttributes(
        damage_grade=normalize_damage_grade(get_field("damage_grade")),
        structural_damage_level=get_field("structural_damage_level", "damage_structural_level"),
        non_structural_damage_level=get_field("non_structural_damage_level", "damage_non_structural_level"),
        crack_severity=normalize_crack_severity(get_field("crack_severity", "damage_crack_severity")),
        crack_pattern=normalize_crack_pattern(get_field("crack_pattern", "damage_crack_pattern")),
        collapse_mode=normalize_collapse_mode(get_field("collapse_mode", "damage_collapse_mode")),
        soft_story_failure=normalize_binary(get_field("soft_story_failure", "damage_soft_story_failure")),
        wall_failure=normalize_binary(get_field("wall_failure", "damage_wall_failure")),
        retrofitting_action=get_field("retrofitting_action", "damage_retrofitting_action"),
        pga=normalize_float(get_field("pga")),
        magnitude=normalize_float(get_field("magnitude")),
    )

    provenance = ProvenanceInfo(
        source_modality="ground_truth",
        source_collection=raw.get("source_collection"),
        image_ids=raw.get("image_ids", []) if isinstance(raw.get("image_ids"), list) else [str(raw.get("image_ids"))] if raw.get("image_ids") else [],
        evidence_text=raw.get("evidence_text") or raw.get("caption"),
        notes=raw.get("notes"),
    )

    return NormalizedAssessmentRecord(
        building_id=bid,
        earthquake_event=str(raw.get("earthquake_event") or "Bhuj_2001"),
        modality="ground_truth",
        typology=typology,
        vulnerability=vulnerability,
        damage=damage,
        provenance=provenance,
        confidence=float(raw.get("confidence") or 1.0),
        raw_output=raw,
    )


def normalize_record(
    record: Any,
    *,
    modality: str = "auto",
    building_id: str | None = None,
) -> NormalizedAssessmentRecord:
    """Auto-detecting dispatcher to normalize any source to NormalizedAssessmentRecord."""
    raw = _extract_flat_dict(record)
    detected_modality = modality
    if detected_modality == "auto":
        evidence_src = str(raw.get("evidence_source") or raw.get("modality") or "").lower()
        if "rag" in evidence_src:
            detected_modality = "rag"
        elif "vlm" in evidence_src:
            detected_modality = "vlm"
        else:
            detected_modality = "ground_truth"

    if detected_modality == "rag":
        return normalize_rag_output(raw, building_id=building_id)
    if detected_modality == "vlm":
        return normalize_vlm_output(raw, building_id=building_id)
    return normalize_ground_truth(raw, building_id=building_id)
