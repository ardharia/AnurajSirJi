# src/run_full_pipeline.py
"""Automated Building Vulnerability Assessment Pipeline for Earthquake Response (Bhuj 2001).

Extends and integrates existing pipeline components:
1. Filters vector embeddings from the FAISS vector store specifically for the Bhuj earthquake event.
2. Ingests candidate building records, computes deterministic unique SHA-256 building IDs,
   and deduplicates entries to isolate physical buildings.
3. Classifies each building under standard NDMA Rapid Visual Screening (RVS) typologies,
   applying keyword-based fallback search (augmented by RAG evidence context) for unknown typologies.
4. Performs typology-wise parameter frequency analysis across standard RVS parameters
   and additional earthquake-observed parameters.
5. Generates publication-ready parameter frequency visualizations per typology saved to outputs/plots/.
6. Cleanly logs and outputs summary metrics to terminal and CSV artifacts.
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Reuse existing pipeline modules and data connectors
from seismic_damage.config.settings import PROJECT_ROOT
from seismic_damage.extraction.frequency_pipeline import (
    load_rvs_dictionary,
    match_additional_parameters,
    match_rvs_parameters,
)
from seismic_damage.ingestion.pdf import ingest_pdf_reports
from seismic_damage.linking.catalog import load_image_rows
from seismic_damage.linking.registry import NON_BUILDING_PATTERNS, classify_typology
from seismic_damage.pipelines.rag.pipeline import RAGPipeline


# Standard RVS keyword fallback taxonomy
KEYWORD_FALLBACK_RULES: List[Tuple[str, List[str]]] = [
    (
        "RC",
        [
            "reinforced concrete",
            "rc frame",
            "rc framed",
            "rc building",
            "concrete frame",
            "concrete",
            "column",
            "beam",
            "slab",
            "multistory",
            "multi-storey",
            "multi-story",
            "apartment",
            "cantilever",
            "open ground",
            "soft storey",
            "soft story",
            "jacketing",
            "propped",
            "lift core",
            "elevator core",
            "floating column",
            "stirrup",
            "pancake collapse",
        ],
    ),
    (
        "RRSM",
        [
            "random rubble",
            "rubble stone",
            "stone masonry",
            "stone houses",
            "stone house",
            "stone wall",
            "mud mortar",
            "rubble",
            "stone",
        ],
    ),
    (
        "BCBM",
        [
            "brick masonry",
            "burnt clay",
            "brick chimney",
            "brick wall",
            "load-bearing brick",
            "load bearing brick",
            "brick",
        ],
    ),
    (
        "CM",
        [
            "confined masonry",
            "tie column",
            "tie beam",
        ],
    ),
    (
        "MH",
        [
            "mud house",
            "mud wall",
            "mud houses",
            "adobe",
            "kutcha",
        ],
    ),
    (
        "DD",
        [
            "dhajji",
            "dhajji diwari",
            "timber frame",
            "wood frame",
            "timber post",
            "timber roof",
            "wooden post",
        ],
    ),
    (
        "EH",
        [
            "ekra",
            "ekra house",
            "bamboo",
        ],
    ),
]


def filter_bhuj_vector_store(
    rag: RAGPipeline,
    query: str = "Bhuj 2001 earthquake building damage structural failure and vulnerability reports",
    top_k: int = 50,
) -> List[Dict[str, Any]]:
    """Query and filter FAISS vector store specifically for the Bhuj earthquake event.

    Args:
        rag: Initialized RAGPipeline instance.
        query: Query string for retrieving Bhuj event reports.
        top_k: Maximum number of top-matching chunks to retrieve.

    Returns:
        List of filtered chunk metadata dicts related to the Bhuj event.
    """
    store = rag._load_store()
    if store is None or store.size == 0:
        print("   Vector store empty or not indexed. Indexing corpus...")
        rag.index_corpus()
        store = rag._load_store()

    filtered_chunks: List[Dict[str, Any]] = []
    seen_chunk_ids: Set[str] = set()

    # 1. Semantic retrieval for Bhuj damage reports
    try:
        rag_result = rag.retrieve(query=query, top_k=top_k)
        for doc in rag_result.documents:
            meta = doc.metadata or {}
            cid = str(meta.get("chunk_id") or doc.document_id)
            if cid not in seen_chunk_ids:
                seen_chunk_ids.add(cid)
                filtered_chunks.append({
                    "chunk_id": cid,
                    "document_id": doc.document_id,
                    "text": doc.content,
                    "source_path": doc.source_path,
                    "page_number": meta.get("page_number", "N/A"),
                    "score": doc.score,
                    "event_name": "Bhuj",
                })
    except Exception as e:
        print(f"   [Warning] Semantic retrieval query failed: {e}")

    # 2. Metadata filtering over store payloads
    if store is not None:
        for payload in store.payloads:
            cid = str(payload.get("chunk_id") or "")
            if cid in seen_chunk_ids:
                continue

            fname = str(payload.get("filename") or "").lower()
            doc_id = str(payload.get("document_id") or "").lower()
            text = str(payload.get("text") or "").lower()
            meta = payload.get("metadata") or {}
            event_tag = str(meta.get("event") or meta.get("event_name") or "").lower()

            if (
                "bhuj" in fname
                or "bhuj" in doc_id
                or "bhuj" in text
                or "bhuj" in event_tag
                or event_tag == "bhuj"
            ):
                seen_chunk_ids.add(cid)
                filtered_chunks.append({
                    "chunk_id": cid,
                    "document_id": payload.get("document_id"),
                    "text": payload.get("text", ""),
                    "source_path": payload.get("source_path"),
                    "page_number": payload.get("page_number", "N/A"),
                    "score": 1.0,
                    "event_name": "Bhuj",
                })

    print(f"   Filtered {len(filtered_chunks)} vector chunks specifically for Bhuj Earthquake Event.")
    return filtered_chunks


def generate_deterministic_building_id(
    description: str,
    location: str = "",
    source_collection: str = "",
    page_number: str = "",
    prefix: str = "BHUJ_BLDG_",
) -> str:
    """Generate a repeatable, unique SHA-256 building ID based on normalized description and provenance.

    Args:
        description: Text description or caption detailing the building.
        location: Specific geographic or urban location if available.
        source_collection: Source collection or document identifier.
        page_number: Source page reference.
        prefix: Prefix string for the resulting identifier.

    Returns:
        Deterministic alphanumeric building ID string.
    """
    norm_desc = " ".join((description or "").strip().lower().split())
    norm_loc = " ".join((location or "").strip().lower().split())
    norm_src = " ".join((source_collection or "").strip().lower().split())
    norm_page = str(page_number or "").strip()

    # Base identity is the physical description and location features
    raw_key = f"{norm_loc}|{norm_desc}".strip("|")
    if not raw_key:
        raw_key = f"{norm_src}|{norm_page}"

    digest = hashlib.sha256(raw_key.encode("utf-8")).hexdigest()[:12].upper()
    return f"{prefix}{digest}"


def extract_and_deduplicate_buildings(
    rag: RAGPipeline,
    bhuj_documents: Optional[List[Dict[str, Any]]] = None,
) -> Tuple[pd.DataFrame, int]:
    """Ingest building candidates, filter non-buildings, compute deterministic IDs, and deduplicate.

    Args:
        rag: Initialized RAGPipeline for context retrieval.
        bhuj_documents: Filtered vector store documents for the Bhuj event.

    Returns:
        Tuple of (deduplicated DataFrame of unique physical buildings, count of duplicates removed).
    """
    raw_rows = load_image_rows()
    print(f"   Loaded {len(raw_rows)} raw building and damage evidence records from source connectors.")

    candidate_records: List[Dict[str, Any]] = []

    for item in raw_rows:
        caption = (item.get("caption") or "").strip()
        if not caption:
            continue

        # Filter out non-building entities (maps, title slides, salt marshes, embankments)
        cap_lower = caption.lower()
        if any(re.search(pat, cap_lower) for pat in NON_BUILDING_PATTERNS):
            continue

        source_doc = item.get("filename") or item.get("source_collection", "bhuj_2001_reports")
        page_no = item.get("source_page", "N/A")

        # Extract location if mentioned in caption prefix (e.g., 'AHMEDABAD: ...' or 'BHUJ: ...')
        loc_match = re.match(r"^([A-Z\s]{3,20}):", caption)
        location = loc_match.group(1).strip() if loc_match else ""

        # Deterministic SHA-256 Building ID
        b_id = generate_deterministic_building_id(
            description=caption,
            location=location,
            source_collection=source_doc,
            page_number=page_no,
        )

        candidate_records.append({
            "building_id": b_id,
            "event_id": "BHUJ_2001",
            "event_name": "Bhuj Earthquake 2001",
            "source_document": source_doc,
            "page_number": str(page_no) if page_no else "N/A",
            "location": location or "Bhuj Region / Gujarat",
            "building_description": caption[:140],
            "evidence_text": caption,
        })

    df = pd.DataFrame(candidate_records)
    initial_count = len(df)

    # Deduplicate strictly on the physical building identifier
    df_unique = df.drop_duplicates(subset=["building_id"]).copy()
    duplicates_removed = initial_count - len(df_unique)

    print(f"   Evaluated {initial_count} physical building entries.")
    print(f"   Identified and removed {duplicates_removed} duplicate records.")
    print(f"   Retained {len(df_unique)} unique physical buildings.")

    return df_unique, duplicates_removed


def classify_rvs_typologies_with_fallback(
    df_buildings: pd.DataFrame,
    rag: RAGPipeline,
) -> pd.DataFrame:
    """Categorize buildings using NDMA RVS typology with keyword fallback on failure.

    Args:
        df_buildings: DataFrame of unique physical buildings.
        rag: Initialized RAGPipeline to retrieve context when description is insufficient.

    Returns:
        DataFrame augmented with 'building_typology', 'typology_confidence', and 'classification_method'.
    """
    typologies: List[str] = []
    confidences: List[float] = []
    methods: List[str] = []

    for _, row in df_buildings.iterrows():
        caption = row.get("evidence_text", "")

        # 1. Primary classification via existing regex classifier
        typ, conf = classify_typology(caption)
        method = "PRIMARY_RVS_RULES"

        # 2. Fallback keyword search if unclassified or unknown
        if typ in ["UNKNOWN", "N/A", ""]:
            # Check caption against keyword fallback rules
            cap_lower = caption.lower()
            matched_fallback = False
            for target_typ, keywords in KEYWORD_FALLBACK_RULES:
                if any(kw in cap_lower for kw in keywords):
                    typ = target_typ
                    conf = 0.85
                    method = "KEYWORD_FALLBACK_DESCRIPTION"
                    matched_fallback = True
                    break

            # 3. If still unknown, query RAG for contextual report text around the building description
            if not matched_fallback:
                try:
                    query_text = caption[:120] if len(caption) > 120 else caption
                    rag_res = rag.retrieve(query_text, top_k=2)
                    rag_ctx = (rag_res.synthesized_context or "").lower()
                    for target_typ, keywords in KEYWORD_FALLBACK_RULES:
                        if any(kw in rag_ctx for kw in keywords):
                            typ = target_typ
                            conf = 0.75
                            method = "KEYWORD_FALLBACK_RAG_CONTEXT"
                            matched_fallback = True
                            break
                except Exception:
                    pass  # Graceful fallback handling

            if not matched_fallback:
                typ = "UNKNOWN"
                conf = 0.0
                method = "UNRESOLVED"

        typologies.append(typ)
        confidences.append(conf)
        methods.append(method)

    df_buildings["building_typology"] = typologies
    df_buildings["typology_confidence"] = confidences
    df_buildings["classification_method"] = methods

    return df_buildings


def analyze_parameter_frequencies(
    df_buildings: pd.DataFrame,
    rag: RAGPipeline,
    outputs_dir: Path,
) -> Tuple[pd.DataFrame, pd.DataFrame, List[Dict[str, Any]]]:
    """Extract standard RVS parameters and additional earthquake parameters and compute frequencies.

    Args:
        df_buildings: Categorized unique buildings DataFrame.
        rag: RAGPipeline for context retrieval.
        outputs_dir: Output directory for CSV artifacts.

    Returns:
        Tuple of (observations DataFrame, frequencies summary DataFrame, list of frequency rows).
    """
    rvs_dictionary = load_rvs_dictionary()
    dict_by_typology: Dict[str, List[Dict[str, str]]] = {}
    for row in rvs_dictionary:
        dict_by_typology.setdefault(row["typology"], []).append(row)

    # Count unique buildings per typology
    typology_building_counts = df_buildings["building_typology"].value_counts().to_dict()

    observations: List[Dict[str, Any]] = []
    seen_observations: Set[Tuple[str, str]] = set()

    for _, row in df_buildings.iterrows():
        b_dict = {
            "building_id": row["building_id"],
            "event_id": row["event_id"],
            "building_typology": row["building_typology"],
            "source_document": row["source_document"],
            "page_number": row["page_number"],
            "evidence_text": row["evidence_text"],
            "building_description": row["building_description"],
        }
        typ = row["building_typology"]

        # Retrieve RAG context for building damage details
        query = row["evidence_text"][:120] if row["evidence_text"] else f"Building {row['building_id']} Bhuj 2001 damage"
        try:
            rag_res = rag.retrieve(query, top_k=2)
            rag_context = rag_res.synthesized_context or ""
        except Exception:
            rag_context = ""

        # Match applicable RVS parameters
        applicable_rvs = dict_by_typology.get(typ, [])
        rvs_matches = match_rvs_parameters(b_dict, applicable_rvs, rag_context)
        for m in rvs_matches:
            key = (m["building_id"], m["rvs_parameter_id"])
            if key not in seen_observations:
                seen_observations.add(key)
                observations.append(m)

        # Match additional earthquake-observed parameters
        add_matches = match_additional_parameters(b_dict, rag_context)
        for m in add_matches:
            key = (m["building_id"], m["rvs_parameter_id"])
            if key not in seen_observations:
                seen_observations.add(key)
                observations.append(m)

    df_obs = pd.DataFrame(observations)

    # Calculate building-level frequency
    frequency_map: Dict[Tuple[str, str, str, str, str], Set[str]] = {}
    for o in observations:
        key = (
            o["event_id"],
            o["typology"],
            o["parameter_source"],
            o["rvs_parameter_id"],
            o["rvs_parameter"],
        )
        frequency_map.setdefault(key, set()).add(o["building_id"])

    frequency_rows: List[Dict[str, Any]] = []
    for (event_id, typ, p_source, param_id, param_text), bldg_set in frequency_map.items():
        freq = len(bldg_set)
        total_bldgs = typology_building_counts.get(typ, 0)
        freq_pct = (freq / total_bldgs) * 100.0 if total_bldgs > 0 else 0.0

        frequency_rows.append({
            "event_id": event_id,
            "typology": typ,
            "total_buildings": total_bldgs,
            "parameter_source": p_source,
            "parameter_id": param_id,
            "parameter": param_text,
            "frequency": freq,
            "frequency_percentage": f"{freq_pct:.1f}%",
        })

    # Sort frequency summary by: typology, frequency DESC
    frequency_rows.sort(key=lambda r: (r["typology"], -r["frequency"]))
    df_freq = pd.DataFrame(frequency_rows)

    return df_obs, df_freq, frequency_rows


def generate_typology_visualizations(
    df_freq: pd.DataFrame,
    outputs_dir: Path,
) -> List[Path]:
    """Generate distinct bar charts for each typology found and save to outputs/plots/.

    X-axis: Parameters (RVS parameters + additional observed parameters).
    Y-axis: Frequency count.

    Args:
        df_freq: DataFrame of parameter frequencies.
        outputs_dir: Root output directory.

    Returns:
        List of Paths to saved plot images.
    """
    plots_dir = outputs_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    saved_plots: List[Path] = []

    if df_freq.empty:
        print("   [Warning] No parameter frequencies to visualize.")
        return saved_plots

    for typ in sorted(df_freq["typology"].unique()):
        sub = df_freq[df_freq["typology"] == typ].copy()
        if sub.empty:
            continue

        # Sort descending by frequency count
        sub = sub.sort_values(by="frequency", ascending=False)

        # Plot setup
        n_params = len(sub)
        fig_width = max(10, min(18, n_params * 0.9 + 4))
        plt.figure(figsize=(fig_width, 6.5))

        # Color-code by parameter source
        colors = [
            "#2b5c8f" if src == "RVS_PRIMER" else "#d95f02"
            for src in sub["parameter_source"]
        ]

        # Use parameter ID + truncated description for readable X labels
        x_labels = [
            f"{pid}\n{p_text[:28]}..." if len(p_text) > 28 else f"{pid}\n{p_text}"
            for pid, p_text in zip(sub["parameter_id"], sub["parameter"])
        ]
        y_values = sub["frequency"].values

        bars = plt.bar(
            range(len(x_labels)),
            y_values,
            color=colors,
            edgecolor="#1a1a1a",
            linewidth=0.8,
            width=0.6,
        )

        # Annotate frequency counts on top of bars
        for bar in bars:
            height = bar.get_height()
            plt.annotate(
                f"{int(height)}",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=9,
                fontweight="bold",
            )

        plt.title(
            f"Bhuj 2001 Earthquake: {typ} Typology Parameter Frequency (RVS + Additional Observed)",
            fontsize=13,
            pad=15,
            fontweight="bold",
        )
        plt.xlabel("Vulnerability & Damage Parameters", fontsize=11, labelpad=10)
        plt.ylabel("Observed Frequency (Unique Buildings)", fontsize=11, labelpad=10)
        plt.xticks(
            range(len(x_labels)),
            x_labels,
            rotation=45,
            ha="right",
            fontsize=8.5,
        )
        plt.grid(axis="y", linestyle="--", alpha=0.5)

        # Add custom legend
        has_rvs = any(src == "RVS_PRIMER" for src in sub["parameter_source"])
        has_add = any(src == "ADDITIONAL_EARTHQUAKE_OBSERVED" for src in sub["parameter_source"])
        legend_handles = []
        legend_labels = []
        if has_rvs:
            legend_handles.append(plt.Rectangle((0, 0), 1, 1, color="#2b5c8f"))
            legend_labels.append("NDMA RVS Primer Parameters")
        if has_add:
            legend_handles.append(plt.Rectangle((0, 0), 1, 1, color="#d95f02"))
            legend_labels.append("Additional Earthquake Observations")
        if legend_handles:
            plt.legend(legend_handles, legend_labels, loc="upper right", frameon=True)

        plt.tight_layout()

        # Save primary plot: outputs/plots/{typology}_frequency.png
        primary_plot_path = plots_dir / f"{typ}_frequency.png"
        plt.savefig(primary_plot_path, dpi=160)

        # Save typology_parameter_plot_{typ}.png in outputs/plots/
        param_plot_path = plots_dir / f"typology_parameter_plot_{typ}.png"
        plt.savefig(param_plot_path, dpi=160)

        # Save backward-compatible copy: outputs/typology_parameter_plot_{typology}.png
        compat_plot_path = outputs_dir / f"typology_parameter_plot_{typ}.png"
        plt.savefig(compat_plot_path, dpi=160)

        plt.close()
        saved_plots.append(primary_plot_path)
        saved_plots.append(param_plot_path)

    return saved_plots


def main() -> None:
    """Execute complete automated Building Vulnerability Assessment pipeline."""
    print("=" * 80)
    print("EARTHQUAKE RESPONSE PIPELINE: BHUJ 2001 VULNERABILITY ASSESSMENT")
    print("=" * 80)

    project_root = PROJECT_ROOT
    outputs_dir = project_root / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)

    # 1. Ingest PDFs and build / connect to FAISS vector store
    print("\n[Step 1/5] Ingesting PDF reports & connecting to Vector Store...")
    try:
        ingest_pdf_reports()
    except Exception as e:
        print(f"   [Notice] PDF ingestion step skipped or already ingested: {e}")

    rag = RAGPipeline()
    bhuj_documents = filter_bhuj_vector_store(rag)

    # 2. Extract, assign deterministic SHA-256 IDs, and deduplicate physical buildings
    print("\n[Step 2/5] Ingesting building records, assigning deterministic IDs & deduplicating...")
    df_unique, duplicates_removed = extract_and_deduplicate_buildings(
        rag=rag,
        bhuj_documents=bhuj_documents,
    )

        # 3. Categorize under RVS typology with keyword fallback
    print("\n[Step 3/5] Categorizing RVS Typology with Keyword & RAG Fallback...")
    df_classified = classify_rvs_typologies_with_fallback(df_unique, rag=rag)

    # -----------------------------------------------------------------
    # Re‑classify any remaining UNKNOWN typology using the lightweight
    # keyword‑based classifier from utils.classification.
    # -----------------------------------------------------------------
    from seismic_damage.utils.classification import classify_buildings as keyword_classify

    unknown_mask = df_classified["building_typology"].isin(["UNKNOWN", "N/A", "", "NON_BUILDING", "DD"])
    if unknown_mask.any():
        unknown_rows = df_classified[unknown_mask]
        records = [
            {
                "id": row["building_id"],
                "notes": row.get("evidence_text", "") or row.get("building_description", ""),
                "location": row.get("location", ""),
                "rvs_format": True,
            }
            for _, row in unknown_rows.iterrows()
        ]
        kw_results = keyword_classify(records)
        typology_map = {res["id"]: res["typology"] for res in kw_results}
        confidence_map = {res["id"]: res["confidence"] for res in kw_results}
        df_classified.loc[unknown_mask, "building_typology"] = df_classified.loc[unknown_mask, "building_id"].map(typology_map)
        df_classified.loc[unknown_mask, "classification_method"] = "KEYWORD_CLASSIFIER"
        df_classified.loc[unknown_mask, "typology_confidence"] = df_classified.loc[unknown_mask, "building_id"].map(confidence_map)

    # Filter out non-building entities if any were marked during keyword classification
    df_classified = df_classified[df_classified["building_typology"] != "NON_BUILDING"].copy()

    # Write building summary CSV (updated after possible re‑classification)
    summary_path = outputs_dir / "building_summary.csv"
    df_summary = df_classified[[
        "building_id",
        "building_typology",
        "typology_confidence",
        "classification_method",
        "source_document",
        "page_number",
        "location",
        "building_description",
    ]].rename(columns={"building_typology": "typology"})
    df_summary.to_csv(summary_path, index=False, encoding="utf-8")
    print(f"   Saved building summary CSV to: {summary_path}")

    # Also synchronize data/metadata/building_registry.csv for repository consistency
    metadata_reg_path = project_root / "data" / "metadata" / "building_registry.csv"
    if metadata_reg_path.parent.exists():
        df_classified.to_csv(metadata_reg_path, index=False, encoding="utf-8")

    # 4. Typology-wise parameter frequency analysis
    print("\n[Step 4/5] Computing Typology-Wise Parameter Frequencies...")
    df_obs, df_freq, frequency_rows = analyze_parameter_frequencies(
        df_buildings=df_classified,
        rag=rag,
        outputs_dir=outputs_dir,
    )

    # Write frequency CSV to outputs
    freq_dest = outputs_dir / "typology_parameter_frequency.csv"
    df_freq.to_csv(freq_dest, index=False, encoding="utf-8")
    print(f"   Saved typology parameter frequency CSV to: {freq_dest}")

    # Synchronize data/analysis/typology_parameter_frequency.csv
    analysis_freq_path = project_root / "data" / "analysis" / "typology_parameter_frequency.csv"
    if analysis_freq_path.parent.exists():
        df_freq.to_csv(analysis_freq_path, index=False, encoding="utf-8")

    # Write additional parameters summary text
    add_params = df_freq[df_freq["parameter_source"] == "ADDITIONAL_EARTHQUAKE_OBSERVED"]
    add_path = outputs_dir / "additional_parameters.txt"
    with add_path.open("w", encoding="utf-8") as f:
        f.write("BHUJ 2001 EARTHQUAKE - ADDITIONAL OBSERVED PARAMETERS\n")
        f.write("=" * 60 + "\n\n")
        if add_params.empty:
            f.write("No additional earthquake-observed parameters detected.\n")
        else:
            for _, r in add_params.iterrows():
                f.write(
                    f"[{r['typology']}] {r['parameter_id']}: {r['parameter']} "
                    f"- Frequency: {r['frequency']} ({r['frequency_percentage']})\n"
                )
    print(f"   Saved additional observed parameters to: {add_path}")

    # 5. Visualization Generation
    print("\n[Step 5/6] Generating Typology-Wise Parameter Frequency Plots...")
    saved_plots = generate_typology_visualizations(df_freq, outputs_dir=outputs_dir)

    # Generate building typology counts bar plot
    print("   Generating building typology counts plot...")
    try:
        from visualization.plot_building_typology_counts import main as plot_typology_counts
        plot_typology_counts()
        counts_plot_path = outputs_dir / "building_typology_counts.png"
        if counts_plot_path.exists():
            saved_plots.append(counts_plot_path)
    except Exception as e:
        print(f"   [Warning] Building typology counts plot failed: {e}")

    # Generate normal distribution bell curves for all typologies
    print("   Generating normal distribution bell curves for all typologies...")
    try:
        import sys
        if str(project_root) not in sys.path:
            sys.path.insert(0, str(project_root))
        from scripts.vulnerability_parameter_analysis import run_vulnerability_pipeline
        bell_results = run_vulnerability_pipeline(
            csv_path=freq_dest,
            output_dir=outputs_dir / "plots",
            dpi=300,
        )
        for br in bell_results:
            if br.plot_filepath and br.plot_filepath.exists():
                saved_plots.append(br.plot_filepath)
    except Exception as e:
        print(f"   [Warning] Bell curve generation failed: {e}")

    # 6. Official Report Compilation (.docx and .pdf)
    print("\n[Step 6/6] Compiling Publication-Ready Word (.docx) & PDF Reports...")
    reports_dir = outputs_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    out_docx = reports_dir / "Bhuj_Vulnerability_Assessment_Report.docx"
    out_pdf = reports_dir / "Bhuj_Vulnerability_Assessment_Report.pdf"

    try:
        import sys
        if str(project_root) not in sys.path:
            sys.path.insert(0, str(project_root))
        from scripts.generate_official_reports import generate_pdf_report, generate_word_report
        generate_word_report(
            df_buildings=df_summary,
            df_frequencies=df_freq,
            additional_lines=[
                f"[{r['typology']}] {r['parameter_id']}: {r['parameter']} - Frequency: {r['frequency']} ({r['frequency_percentage']})"
                for _, r in add_params.iterrows()
            ],
            output_docx=out_docx,
        )
        generate_pdf_report(
            df_buildings=df_summary,
            df_frequencies=df_freq,
            additional_lines=[
                f"[{r['typology']}] {r['parameter_id']}: {r['parameter']} - Frequency: {r['frequency']} ({r['frequency_percentage']})"
                for _, r in add_params.iterrows()
            ],
            output_pdf=out_pdf,
        )
        print(f"   Generated Word Report: {out_docx.resolve()}")
        print(f"   Generated PDF Report : {out_pdf.resolve()}")
    except Exception as e:
        print(f"   [Warning] Report compilation failed: {e}")

    # -------------------------------------------------------------------------
    # FINAL OUTPUT LOGGING AS SPECIFIED
    # -------------------------------------------------------------------------
    total_unique_buildings = len(df_classified)
    typology_breakdown = df_classified["building_typology"].value_counts().to_dict()

    print("\n" + "=" * 80)
    print("FINAL SUMMARY REPORT & METRICS")
    print("=" * 80)

    # 1. Total Number of Unique Buildings Found
    print(f"\n1. TOTAL NUMBER OF UNIQUE BUILDINGS FOUND (POST-DEDUPLICATION): {total_unique_buildings}")
    print(f"   (Duplicates Removed: {duplicates_removed})")

    # 2. Building Breakdown by Typology
    print("\n2. BUILDING BREAKDOWN BY TYPOLOGY:")
    print("-" * 55)
    print(f"{'RVS Typology':<20} {'Building Count':<16} {'Percentage':<15}")
    print("-" * 55)
    for typ, count in sorted(typology_breakdown.items(), key=lambda x: -x[1]):
        pct = (count / total_unique_buildings) * 100.0 if total_unique_buildings > 0 else 0.0
        print(f"{typ:<20} {count:<16} {pct:.1f}%")
    print("-" * 55)

    # 3. Typology-Wise Parameter Frequencies
    print("\n3. TYPOLOGY-WISE PARAMETER FREQUENCIES:")
    for typ in sorted(df_freq["typology"].unique()):
        sub_freq = df_freq[df_freq["typology"] == typ]
        n_bldgs = typology_breakdown.get(typ, 0)
        print(f"\n--- Typology: {typ} (Total Buildings: {n_bldgs}) ---")
        print(f"{'Source':<12} {'Parameter ID':<15} {'Frequency':<10} {'% Typology':<12} {'Parameter Description'}")
        print("-" * 95)
        for _, r in sub_freq.iterrows():
            src_tag = "RVS" if r["parameter_source"] == "RVS_PRIMER" else "ADDITIONAL"
            p_desc = r["parameter"][:45] + "..." if len(r["parameter"]) > 48 else r["parameter"]
            print(f"{src_tag:<12} {r['parameter_id']:<15} {r['frequency']:<10} {r['frequency_percentage']:<12} {p_desc}")
        print("-" * 95)

    # 4. Paths to Saved Plot Images
    print("\n4. GENERATED PLOTS SAVED TO FILESYSTEM:")
    for p in saved_plots:
        print(f"   -> {p.resolve()}")

    # 5. Paths to Generated Reports
    print("\n5. OFFICIAL PUBLICATION-READY REPORTS:")
    print(f"   -> Word (.docx): {out_docx.resolve()}")
    print(f"   -> PDF  (.pdf) : {out_pdf.resolve()}")

    print("\n" + "=" * 80)
    print("PIPELINE EXECUTION COMPLETED SUCCESSFULLY.")
    print("=" * 80)


if __name__ == "__main__":
    main()
