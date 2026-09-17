"""Building-level consolidation and separate summary generation for RAG and VLM.

Ensures:
1. Exactly one consolidated RAG record and one consolidated VLM record per canonical building_id.
2. Identical building IDs and identical sequence across both modalities.
3. Strict independence — RAG and VLM are NEVER fused, averaged, or combined.
4. No invented or filled missing values; missing parameters explicitly recorded.
5. Separate visualizations for RAG and VLM.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from seismic_damage.config.settings import PROJECT_ROOT
from seismic_damage.extraction.text_parser import parse_damage_text
from seismic_damage.io_utils import ensure_parent, write_jsonl
from seismic_damage.linking.catalog import build_catalog, load_image_rows
from seismic_damage.pipelines.rag.embedders import TfidfEmbedder
from seismic_damage.pipelines.rag.pipeline import RAGPipeline
from seismic_damage.schemas.linking import BuildingRecord
from seismic_damage.schemas.normalized import NormalizedAssessmentRecord
from seismic_damage.validation.cross_validation import records_from_vlm_links
from seismic_damage.validation.metrics import ALL_PARAMETERS, is_missing
from seismic_damage.validation.normalize import normalize_rag_output, normalize_vlm_output

RESULTS_DIR = PROJECT_ROOT / "data" / "processed" / "results"
FIGURES_DIR = RESULTS_DIR / "figures"

RAG_SUMMARY_CSV = RESULTS_DIR / "rag_building_summary.csv"
RAG_SUMMARY_JSONL = RESULTS_DIR / "rag_building_summary.jsonl"
VLM_SUMMARY_CSV = RESULTS_DIR / "vlm_building_summary.csv"
VLM_SUMMARY_JSONL = RESULTS_DIR / "vlm_building_summary.jsonl"

CSV_COLUMNS = [
    "building_id",
    "earthquake_event",
    "evidence_source",
    "confidence",
    "evidence_text",
    # Typology
    "material_type",
    "structural_system",
    "number_of_stories",
    "occupancy",
    # Vulnerability (pre-earthquake)
    "soft_story",
    "plan_irregularity",
    "vertical_irregularity",
    "short_columns",
    "soil_type",
    # Damage (post-earthquake)
    "damage_grade",
    "collapse_mode",
    "soft_story_failure",
    "wall_failure",
    "crack_severity",
    "crack_pattern",
    "pga",
    "magnitude",
    "missing_parameters",
]


def generate_rag_building_summaries(
    catalog: list[BuildingRecord],
    *,
    pipeline: RAGPipeline | None = None,
) -> list[NormalizedAssessmentRecord]:
    """Generate exactly one consolidated RAG record per canonical building in order.

    Preserves citations, document IDs, chunk IDs, and confidence scores.
    Does not invent missing information.
    """
    rag_pipe = pipeline or RAGPipeline(embedder=TfidfEmbedder())

    records: list[NormalizedAssessmentRecord] = []
    for building in catalog:
        query = building.caption or f"Building {building.building_id} Bhuj 2001 damage"
        res = rag_pipe.retrieve(query, top_k=3)

        extracted: dict[str, Any] = {}
        if res.evidence_sufficient and res.synthesized_context:
            extracted = parse_damage_text(res.synthesized_context)

        citations_payload = [item.model_dump() for item in res.citations]

        payload: dict[str, Any] = {
            "building_id": building.building_id,
            "earthquake_event": "Bhuj_2001",
            "evidence_source": "rag",
            "confidence": float(extracted.get("confidence") or 0.0),
            "evidence_text": res.answer or (res.synthesized_context[:300] if res.synthesized_context else ""),
            "rag_evidence_sufficient": res.evidence_sufficient,
            "citations": citations_payload,
            "parameters": extracted,
        }
        payload.update(extracted)

        norm = normalize_rag_output(payload, building_id=building.building_id)
        records.append(norm)

    return records


def generate_vlm_building_summaries(
    catalog: list[BuildingRecord],
    image_rows: list[dict[str, str]] | None = None,
) -> list[NormalizedAssessmentRecord]:
    """Generate exactly one consolidated VLM record per canonical building in order.

    Consolidates image-level observations using records_from_vlm_links:
    - Agreing parameters across images are retained.
    - Conflicting parameters become missing (no voting).
    - Unstated parameters remain explicitly missing.
    """
    rows = image_rows if image_rows is not None else load_image_rows()

    # Create image-level links
    vlm_links: list[dict[str, Any]] = []
    for row in rows:
        iid = row.get("image_id") or row.get("filename")
        caption = row.get("caption") or ""
        parsed = parse_damage_text(caption) if caption else {}

        # Map to canonical building_id
        bid = None
        for b in catalog:
            if iid and iid in b.image_ids:
                bid = b.building_id
                break
        if bid is None and row.get("caption"):
            # Match by identical caption
            for b in catalog:
                if b.caption and b.caption.strip() == row.get("caption", "").strip():
                    bid = b.building_id
                    break

        payload = {
            "image_id": iid,
            "image_path": f"data/raw/images/{row.get('filename', '')}",
            "evidence": caption,
            "source_collection": row.get("source_collection") or row.get("_collection"),
            **parsed,
        }
        vlm_links.append({"building_id": bid, "payload": payload})

    # Consolidate image links to building records without voting
    collapsed = records_from_vlm_links(vlm_links)
    by_bid = {str(r["building_id"]): r for r in collapsed if r.get("building_id")}

    records: list[NormalizedAssessmentRecord] = []
    for building in catalog:
        bid = building.building_id
        raw_building = by_bid.get(bid, {"building_id": bid, "evidence_source": "vlm", "confidence": 0.0})
        # Add image paths
        raw_building["image_paths"] = [
            f"data/raw/images/{iid}" for iid in building.image_ids
        ]
        norm = normalize_vlm_output(raw_building, building_id=bid)
        records.append(norm)

    return records


def _write_summary_csv(records: list[NormalizedAssessmentRecord], target_path: Path) -> Path:
    ensure_parent(target_path)
    with target_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for rec in records:
            d = rec.to_dict()
            t = d.get("typology") or {}
            v = d.get("vulnerability") or {}
            dmg = d.get("damage") or {}
            row = {
                "building_id": rec.building_id,
                "earthquake_event": rec.earthquake_event,
                "evidence_source": rec.modality,
                "confidence": str(rec.confidence),
                "evidence_text": rec.provenance.evidence_text or "",
                "material_type": t.get("material_type") or "",
                "structural_system": t.get("structural_system") or "",
                "number_of_stories": str(t.get("number_of_stories")) if t.get("number_of_stories") is not None else "",
                "occupancy": t.get("occupancy") or "",
                "soft_story": str(v.get("soft_story")) if v.get("soft_story") is not None else "",
                "plan_irregularity": str(v.get("plan_irregularity")) if v.get("plan_irregularity") is not None else "",
                "vertical_irregularity": str(v.get("vertical_irregularity")) if v.get("vertical_irregularity") is not None else "",
                "short_columns": str(v.get("short_columns")) if v.get("short_columns") is not None else "",
                "soil_type": v.get("soil_type") or "",
                "damage_grade": str(dmg.get("damage_grade")) if dmg.get("damage_grade") is not None else "",
                "collapse_mode": dmg.get("collapse_mode") or "",
                "soft_story_failure": str(dmg.get("soft_story_failure")) if dmg.get("soft_story_failure") is not None else "",
                "wall_failure": str(dmg.get("wall_failure")) if dmg.get("wall_failure") is not None else "",
                "crack_severity": dmg.get("crack_severity") or "",
                "crack_pattern": dmg.get("crack_pattern") or "",
                "pga": str(dmg.get("pga")) if dmg.get("pga") is not None else "",
                "magnitude": str(dmg.get("magnitude")) if dmg.get("magnitude") is not None else "",
                "missing_parameters": ";".join(rec.missing_parameters),
            }
            writer.writerow(row)
    return target_path


def export_consolidated_summaries(
    rag_records: list[NormalizedAssessmentRecord],
    vlm_records: list[NormalizedAssessmentRecord],
    *,
    results_dir: Path | None = None,
) -> dict[str, Path]:
    """Export RAG and VLM summaries to CSV and JSONL."""
    base = results_dir or RESULTS_DIR
    rag_csv = base / "rag_building_summary.csv"
    rag_jsonl = base / "rag_building_summary.jsonl"
    vlm_csv = base / "vlm_building_summary.csv"
    vlm_jsonl = base / "vlm_building_summary.jsonl"

    _write_summary_csv(rag_records, rag_csv)
    write_jsonl(rag_jsonl, [r.to_dict() for r in rag_records])

    _write_summary_csv(vlm_records, vlm_csv)
    write_jsonl(vlm_jsonl, [r.to_dict() for r in vlm_records])

    return {
        "rag_csv": rag_csv,
        "rag_jsonl": rag_jsonl,
        "vlm_csv": vlm_csv,
        "vlm_jsonl": vlm_jsonl,
    }


def render_modality_plots(
    rag_records: list[NormalizedAssessmentRecord],
    vlm_records: list[NormalizedAssessmentRecord],
    *,
    output_dir: Path | None = None,
) -> dict[str, Path]:
    """Create separate visualizations for RAG and VLM results."""
    dest = ensure_parent((output_dir or FIGURES_DIR) / ".gitkeep").parent
    dest.mkdir(parents=True, exist_ok=True)

    paths: dict[str, Path] = {}

    # 1. RAG Damage Grade Distribution
    rag_grades = [r.damage.damage_grade for r in rag_records if r.damage.damage_grade is not None]
    fig, ax = plt.subplots(figsize=(6.5, 4))
    counts = [rag_grades.count(g) for g in range(6)]
    ax.bar([f"Grade {g}" for g in range(6)], counts, color="#1f77b4", edgecolor="black")
    ax.set_title("RAG Retrieved: Damage Grade Distribution (110 Buildings)")
    ax.set_xlabel("Damage Grade (EMS-98 / IS 13935)")
    ax.set_ylabel("Building Count")
    ax.set_ylim(0, max(counts + [1]) * 1.2)
    for i, c in enumerate(counts):
        if c > 0:
            ax.text(i, c + 0.5, str(c), ha="center", va="bottom", fontweight="bold")
    fig.tight_layout()
    p1 = dest / "rag_damage_grade_frequency.png"
    fig.savefig(p1, dpi=140)
    plt.close(fig)
    paths["rag_damage_grades"] = p1

    # 2. RAG Parameter Coverage
    rag_coverage: dict[str, float] = {}
    for param in sorted(ALL_PARAMETERS):
        present = sum(1 for r in rag_records if param in r.parameters)
        rag_coverage[param] = present / len(rag_records) if rag_records else 0.0

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(list(rag_coverage.keys()), list(rag_coverage.values()), color="#4c78a8")
    ax.set_xlim(0, 1.0)
    ax.set_title("RAG Retrieved: Parameter Coverage (110 Buildings)")
    ax.set_xlabel("Fraction of Buildings with Grounded Parameter")
    fig.tight_layout()
    p2 = dest / "rag_parameter_coverage.png"
    fig.savefig(p2, dpi=140)
    plt.close(fig)
    paths["rag_coverage"] = p2

    # 3. VLM Damage Grade Distribution
    vlm_grades = [r.damage.damage_grade for r in vlm_records if r.damage.damage_grade is not None]
    fig, ax = plt.subplots(figsize=(6.5, 4))
    counts_vlm = [vlm_grades.count(g) for g in range(6)]
    ax.bar([f"Grade {g}" for g in range(6)], counts_vlm, color="#e45756", edgecolor="black")
    ax.set_title("VLM Observed: Damage Grade Distribution (110 Buildings)")
    ax.set_xlabel("Damage Grade (EMS-98 / IS 13935)")
    ax.set_ylabel("Building Count")
    ax.set_ylim(0, max(counts_vlm + [1]) * 1.2)
    for i, c in enumerate(counts_vlm):
        if c > 0:
            ax.text(i, c + 0.5, str(c), ha="center", va="bottom", fontweight="bold")
    fig.tight_layout()
    p3 = dest / "vlm_damage_grade_frequency.png"
    fig.savefig(p3, dpi=140)
    plt.close(fig)
    paths["vlm_damage_grades"] = p3

    # 4. VLM Parameter Coverage
    vlm_coverage: dict[str, float] = {}
    for param in sorted(ALL_PARAMETERS):
        present = sum(1 for r in vlm_records if param in r.parameters)
        vlm_coverage[param] = present / len(vlm_records) if vlm_records else 0.0

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(list(vlm_coverage.keys()), list(vlm_coverage.values()), color="#f58518")
    ax.set_xlim(0, 1.0)
    ax.set_title("VLM Observed: Parameter Coverage (110 Buildings)")
    ax.set_xlabel("Fraction of Buildings with Visual Finding")
    fig.tight_layout()
    p4 = dest / "vlm_parameter_coverage.png"
    fig.savefig(p4, dpi=140)
    plt.close(fig)
    paths["vlm_coverage"] = p4

    return paths


def run_consolidation_pipeline() -> dict[str, Any]:
    """Execute end-to-end Step 5 and Step 6 consolidation."""
    image_rows = load_image_rows()
    catalog = build_catalog(image_rows)

    rag_records = generate_rag_building_summaries(catalog)
    vlm_records = generate_vlm_building_summaries(catalog, image_rows)

    # Verification: identical building IDs and sequence
    rag_bids = [r.building_id for r in rag_records]
    vlm_bids = [r.building_id for r in vlm_records]
    catalog_bids = [b.building_id for b in catalog]

    if rag_bids != catalog_bids:
        raise ValueError("RAG building IDs do not match canonical catalog order")
    if vlm_bids != catalog_bids:
        raise ValueError("VLM building IDs do not match canonical catalog order")

    exported = export_consolidated_summaries(rag_records, vlm_records)
    plots = render_modality_plots(rag_records, vlm_records)

    return {
        "n_buildings": len(catalog),
        "exported_files": exported,
        "figures": plots,
        "rag_damage_grade_count": sum(1 for r in rag_records if r.damage.damage_grade is not None),
        "vlm_damage_grade_count": sum(1 for r in vlm_records if r.damage.damage_grade is not None),
    }
