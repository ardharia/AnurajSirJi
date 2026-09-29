"""Parameter-specific reliability calculation for Step 14 fusion.

Converts RAG and VLM ground-truth performance metrics into calibrated weights.
These weights reflect SOURCE RELIABILITY, not parameter importance.

Rules:
- Weights are derived from measured performance vs ground truth.
- Never assume 50:50 unless that is an explicit baseline (F3).
- Weights sum to 1.0 per parameter, or are both 0 if evidence is insufficient.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Mapping, Sequence

from seismic_damage.io_utils import ensure_parent
from seismic_damage.validation.metrics import (
    ALL_PARAMETERS,
    is_missing,
    metrics_for,
    parameter_kind,
)
from seismic_damage.validation.cross_validation import _index_by_building, _value


# Minimum number of non-missing samples needed to derive a reliable weight.
MIN_SAMPLES_FOR_WEIGHT = 3


def _score_from_metric(metric: dict[str, Any], kind: str) -> float:
    """Convert a metric dict to a [0, 1] reliability score.

    Higher score = more reliable for this parameter kind.
    For numerical (MAE/RMSE), score is inverted (lower error = higher score).
    """
    n = metric.get("n", 0)
    if n < MIN_SAMPLES_FOR_WEIGHT:
        return 0.0

    if kind == "binary":
        val = metric.get("f1")
        if val is None:
            val = metric.get("accuracy", 0.0)
        return float(max(0.0, min(1.0, val)))

    if kind == "categorical":
        val = metric.get("macro_f1")
        if val is None:
            val = metric.get("accuracy", 0.0)
        return float(max(0.0, min(1.0, val)))

    if kind == "ordinal":
        # Use QWK where available; fall back to exact_match.
        qwk = metric.get("quadratic_weighted_kappa")
        if qwk is not None and isinstance(qwk, (int, float)):
            # QWK can be negative. Clip to [0, 1] for weight purposes.
            return float(max(0.0, min(1.0, float(qwk))))
        exact = metric.get("exact_match", 0.0)
        return float(max(0.0, min(1.0, exact)))

    if kind == "numerical":
        mae = metric.get("mae")
        if mae is not None:
            return 1.0 / (1.0 + float(mae))
        return 0.0

    return 0.0


def compute_parameter_reliability(
    rag_records: list[Mapping[str, Any]],
    vlm_records: list[Mapping[str, Any]],
    ground_truth_records: list[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Compute per-parameter reliability weights for RAG and VLM.

    Args:
        rag_records: Building-level RAG records.
        vlm_records: Building-level VLM records.
        ground_truth_records: Reference ground truth records.

    Returns:
        List of dicts with one row per parameter containing:
            parameter_id, rag_metric, rag_score, vlm_metric, vlm_score,
            rag_reliability, vlm_reliability, rag_weight, vlm_weight, sample_count.
    """
    rag_idx = _index_by_building(rag_records)
    vlm_idx = _index_by_building(vlm_records)
    gt_idx = _index_by_building(ground_truth_records)

    # Three-way intersection for shared buildings
    shared = sorted(set(rag_idx) & set(vlm_idx) & set(gt_idx))

    rows: list[dict[str, Any]] = []
    for param in sorted(ALL_PARAMETERS):
        kind = parameter_kind(param)
        gt_vals = [_value(gt_idx[bid], param) for bid in shared]
        rag_vals = [_value(rag_idx[bid], param) for bid in shared]
        vlm_vals = [_value(vlm_idx[bid], param) for bid in shared]

        rag_metric = metrics_for(param, gt_vals, rag_vals)
        vlm_metric = metrics_for(param, gt_vals, vlm_vals)

        rag_score = _score_from_metric(rag_metric, kind)
        vlm_score = _score_from_metric(vlm_metric, kind)

        total = rag_score + vlm_score
        if total > 0:
            rag_weight = rag_score / total
            vlm_weight = vlm_score / total
        elif rag_score == 0 and vlm_score == 0:
            # Insufficient evidence from both — flag as unknown
            rag_weight = 0.0
            vlm_weight = 0.0
        else:
            rag_weight = 0.5
            vlm_weight = 0.5

        # Weights must sum to 1 (or both be 0 if insufficient)
        assert abs((rag_weight + vlm_weight) - 1.0) < 1e-9 or (rag_weight == vlm_weight == 0)

        # Pick the name of the primary metric used
        if kind == "binary":
            primary_metric_name = "f1"
        elif kind == "categorical":
            primary_metric_name = "macro_f1"
        elif kind == "ordinal":
            qwk = rag_metric.get("quadratic_weighted_kappa")
            primary_metric_name = "quadratic_weighted_kappa" if qwk is not None else "exact_match"
        else:
            primary_metric_name = "mae_inverted"

        rows.append(
            {
                "parameter_id": param,
                "kind": kind,
                "primary_metric_name": primary_metric_name,
                "rag_metric_n": rag_metric.get("n", 0),
                "rag_score": round(rag_score, 6),
                "vlm_metric_n": vlm_metric.get("n", 0),
                "vlm_score": round(vlm_score, 6),
                "rag_reliability": round(rag_score, 6),
                "vlm_reliability": round(vlm_score, 6),
                "rag_weight": round(rag_weight, 6),
                "vlm_weight": round(vlm_weight, 6),
                "sample_count": rag_metric.get("n", 0),
                "sufficient_evidence": rag_score > 0 or vlm_score > 0,
                # Store raw metric details for transparency
                "rag_raw_metric": str(rag_metric),
                "vlm_raw_metric": str(vlm_metric),
            }
        )
    return rows


def write_reliability_csv(rows: list[dict[str, Any]], path: Path) -> Path:
    """Save parameter reliability table to CSV."""
    ensure_parent(path)
    if not rows:
        path.write_text("", encoding="utf-8")
        return path
    fieldnames = [
        "parameter_id", "kind", "primary_metric_name",
        "rag_score", "vlm_score",
        "rag_reliability", "vlm_reliability",
        "rag_weight", "vlm_weight",
        "sample_count", "sufficient_evidence",
        "rag_metric_n", "vlm_metric_n",
    ]
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return path


def reliability_weights_by_param(
    reliability_rows: list[dict[str, Any]],
) -> dict[str, dict[str, float]]:
    """Convert reliability rows to a {param -> {rag_weight, vlm_weight}} dict."""
    return {
        row["parameter_id"]: {
            "rag_weight": float(row["rag_weight"]),
            "vlm_weight": float(row["vlm_weight"]),
        }
        for row in reliability_rows
    }
