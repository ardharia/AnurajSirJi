"""Unit tests for building-level summaries (Step 5 & Step 6)."""

from __future__ import annotations

from pathlib import Path
import pandas as pd
import pytest

from seismic_damage.linking.catalog import build_catalog, load_image_rows
from seismic_damage.pipelines.consolidation import (
    FIGURES_DIR,
    RAG_SUMMARY_CSV,
    RAG_SUMMARY_JSONL,
    VLM_SUMMARY_CSV,
    VLM_SUMMARY_JSONL,
    generate_rag_building_summaries,
    generate_vlm_building_summaries,
)
from seismic_damage.schemas.normalized import NormalizedAssessmentRecord


def test_consolidation_record_counts_and_order() -> None:
    catalog = build_catalog(load_image_rows())
    assert len(catalog) == 110

    rag_records = generate_rag_building_summaries(catalog)
    vlm_records = generate_vlm_building_summaries(catalog)

    assert len(rag_records) == 110
    assert len(vlm_records) == 110

    catalog_bids = [b.building_id for b in catalog]
    rag_bids = [r.building_id for r in rag_records]
    vlm_bids = [r.building_id for r in vlm_records]

    # Exactly 1 record per canonical building, identical ordering
    assert rag_bids == catalog_bids
    assert vlm_bids == catalog_bids

    for r in rag_records:
        assert isinstance(r, NormalizedAssessmentRecord)
        assert r.modality == "rag"

    for r in vlm_records:
        assert isinstance(r, NormalizedAssessmentRecord)
        assert r.modality == "vlm"


def test_persisted_summary_files() -> None:
    assert RAG_SUMMARY_CSV.exists()
    assert RAG_SUMMARY_JSONL.exists()
    assert VLM_SUMMARY_CSV.exists()
    assert VLM_SUMMARY_JSONL.exists()

    df_rag = pd.read_csv(RAG_SUMMARY_CSV)
    df_vlm = pd.read_csv(VLM_SUMMARY_CSV)

    assert len(df_rag) == 110
    assert len(df_vlm) == 110

    # Verify identical building IDs and ordering
    assert (df_rag["building_id"] == df_vlm["building_id"]).all()

    # Verify modality tags
    assert (df_rag["evidence_source"] == "rag").all()
    assert (df_vlm["evidence_source"] == "vlm").all()


def test_figures_exist_and_non_empty() -> None:
    expected_figures = [
        "rag_damage_grade_frequency.png",
        "rag_parameter_coverage.png",
        "vlm_damage_grade_frequency.png",
        "vlm_parameter_coverage.png",
    ]
    for fig_name in expected_figures:
        fig_path = FIGURES_DIR / fig_name
        assert fig_path.exists(), f"Missing figure: {fig_path}"
        assert fig_path.stat().st_size > 1000, f"Figure {fig_path} is empty or too small"
