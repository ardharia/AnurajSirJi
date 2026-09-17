"""Unit tests for building ground truth dataset and building_links export."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from seismic_damage.linking.ground_truth import (
    BUILDING_LINKS_CSV,
    GROUND_TRUTH_CSV,
    GROUND_TRUTH_JSONL,
    build_ground_truth_dataset,
    export_building_links,
)


def test_export_building_links(tmp_path: Path) -> None:
    test_csv = tmp_path / "test_building_links.csv"
    out = export_building_links(test_csv)
    assert out.exists()
    assert out.stat().st_size > 0

    df = pd.read_csv(out)
    assert len(df) == 119
    expected_cols = [
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
    for col in expected_cols:
        assert col in df.columns
    # Every evidence link must have a valid building_id
    assert df["building_id"].notna().all()
    assert (df["building_id"].str.startswith("bldg_")).all()


def test_build_ground_truth_dataset(tmp_path: Path) -> None:
    test_csv = tmp_path / "test_gt.csv"
    test_jsonl = tmp_path / "test_gt.jsonl"
    records, csv_out, jsonl_out = build_ground_truth_dataset(
        csv_path=test_csv, jsonl_path=test_jsonl
    )

    assert len(records) == 110
    assert csv_out.exists()
    assert jsonl_out.exists()

    df = pd.read_csv(csv_out)
    assert len(df) == 110
    assert "building_id" in df.columns
    assert "evidence_text" in df.columns
    assert "damage_grade" in df.columns
    assert "missing_parameters" in df.columns

    # Check that building IDs match canonical format
    assert (df["building_id"].str.startswith("bldg_")).all()

    # Check that damage grades are in valid range 0..5 or null
    valid_grades = df["damage_grade"].dropna()
    assert valid_grades.isin([0, 1, 2, 3, 4, 5]).all()

    # Check that every record has a list of missing parameters
    for r in records:
        assert "missing_parameters" in r
        assert isinstance(r["missing_parameters"], list)
        assert len(r["missing_parameters"]) > 0

        # Check clean separation
        assert "typology" in r
        assert "vulnerability" in r
        assert "damage" in r


def test_persisted_files_exist() -> None:
    """Ensure the standard production files exist and are valid."""
    assert BUILDING_LINKS_CSV.exists(), f"Missing {BUILDING_LINKS_CSV}"
    assert GROUND_TRUTH_CSV.exists(), f"Missing {GROUND_TRUTH_CSV}"
    assert GROUND_TRUTH_JSONL.exists(), f"Missing {GROUND_TRUTH_JSONL}"

    df_links = pd.read_csv(BUILDING_LINKS_CSV)
    assert len(df_links) == 119

    df_gt = pd.read_csv(GROUND_TRUTH_CSV)
    assert len(df_gt) == 110
