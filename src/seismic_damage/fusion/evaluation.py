"""Step 14: Fusion evaluation against Ground Truth and baseline comparison.

Evaluates:
  F1: RAG only vs GT
  F2: VLM only vs GT
  F3: Equal-weight fusion vs GT
  F4: Calibrated fusion vs GT
  Diagnostic: RAG vs VLM

Produces:
  parameter_reliability.csv
  fusion_predictions.csv
  fusion_metrics.csv
  fusion_comparison.csv
  fusion_manifest.json
"""

from __future__ import annotations

import csv
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from seismic_damage.config.settings import PROJECT_ROOT
from seismic_damage.fusion.fusion import (
    FUSION_DIR,
    FUSION_MODES,
    fuse_building,
    generate_fused_dataset,
    write_predictions_csv,
)
from seismic_damage.fusion.reliability import (
    compute_parameter_reliability,
    reliability_weights_by_param,
    write_reliability_csv,
)
from seismic_damage.io_utils import ensure_parent, write_json
from seismic_damage.validation.cross_validation import _index_by_building, _value, compare_sources
from seismic_damage.validation.metrics import (
    ALL_PARAMETERS,
    is_missing,
    metrics_for,
    parameter_kind,
)


PRIMARY_METRICS_MAP: dict[str, str] = {
    "ordinal": "exact_match",
    "binary": "f1",
    "categorical": "macro_f1",
    "numerical": "mae",
}


def _extract_headline_score(metric: dict[str, Any], kind: str) -> tuple[str, float | None]:
    """Return (metric_name, score) for a parameter metric dictionary."""
    pref = PRIMARY_METRICS_MAP.get(kind, "accuracy")
    val = metric.get(pref)
    if val is not None:
        return pref, float(val)
    # Fallback to accuracy
    acc = metric.get("accuracy")
    if acc is not None:
        return "accuracy", float(acc)
    return pref, None


def evaluate_records_against_gt(
    records: list[Mapping[str, Any]],
    ground_truth_records: list[Mapping[str, Any]],
    label: str,
) -> list[dict[str, Any]]:
    """Compute per-parameter metrics for a set of building records against Ground Truth."""
    rec_idx = _index_by_building(records)
    gt_idx = _index_by_building(ground_truth_records)
    shared = sorted(set(rec_idx) & set(gt_idx))

    metric_rows: list[dict[str, Any]] = []
    for param in sorted(ALL_PARAMETERS):
        kind = parameter_kind(param)
        pred_vals = [_value(rec_idx[bid], param) for bid in shared]
        gt_vals = [_value(gt_idx[bid], param) for bid in shared]

        m = metrics_for(param, pred_vals, gt_vals)
        metric_name, score = _extract_headline_score(m, kind)

        row: dict[str, Any] = {
            "source_or_baseline": label,
            "parameter": param,
            "kind": kind,
            "n_matched_buildings": len(shared),
            "metric_sample_count": m.get("n", 0),
            "primary_metric": metric_name,
            "primary_score": score,
        }
        # Add all individual metric fields
        for k, v in m.items():
            if k not in ("parameter", "kind", "n"):
                row[f"metric_{k}"] = v
        metric_rows.append(row)

    return metric_rows


def build_fusion_comparison(
    rag_metrics: list[dict[str, Any]],
    vlm_metrics: list[dict[str, Any]],
    equal_metrics: list[dict[str, Any]],
    calibrated_metrics: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Build the required fusion_comparison table comparing all baselines.

    Columns:
      parameter
      kind
      metric
      RAG_score
      VLM_score
      equal_fusion_score
      calibrated_fusion_score
      sample_count
    """
    rag_by_p = {r["parameter"]: r for r in rag_metrics}
    vlm_by_p = {r["parameter"]: r for r in vlm_metrics}
    eq_by_p = {r["parameter"]: r for r in equal_metrics}
    cal_by_p = {r["parameter"]: r for r in calibrated_metrics}

    comparison: list[dict[str, Any]] = []
    for param in sorted(ALL_PARAMETERS):
        kind = parameter_kind(param)
        metric_name = PRIMARY_METRICS_MAP.get(kind, "accuracy")

        r_row = rag_by_p.get(param, {})
        v_row = vlm_by_p.get(param, {})
        eq_row = eq_by_p.get(param, {})
        cal_row = cal_by_p.get(param, {})

        sample_count = max(
            r_row.get("metric_sample_count", 0),
            v_row.get("metric_sample_count", 0),
            eq_row.get("metric_sample_count", 0),
            cal_row.get("metric_sample_count", 0),
        )

        comparison.append(
            {
                "parameter": param,
                "kind": kind,
                "metric": metric_name,
                "RAG_score": r_row.get("primary_score"),
                "VLM_score": v_row.get("primary_score"),
                "equal_fusion_score": eq_row.get("primary_score"),
                "calibrated_fusion_score": cal_row.get("primary_score"),
                "sample_count": sample_count,
            }
        )

    return comparison


def run_fusion_pipeline(
    rag_records: list[Mapping[str, Any]],
    vlm_records: list[Mapping[str, Any]],
    ground_truth_records: list[Mapping[str, Any]],
    *,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    """Execute Step 14 end-to-end: calibrated fusion and comprehensive evaluation.

    Args:
        rag_records: Building-level RAG predictions.
        vlm_records: Building-level VLM predictions.
        ground_truth_records: Ground truth records for evaluation.
        output_dir: Optional override directory.

    Returns:
        Summary dict containing output paths, metrics, and comparisons.
    """
    out = output_dir or FUSION_DIR
    out.mkdir(parents=True, exist_ok=True)

    # 1. Compute parameter reliability & weights
    rel_rows = compute_parameter_reliability(rag_records, vlm_records, ground_truth_records)
    rel_csv = write_reliability_csv(rel_rows, out / "parameter_reliability.csv")
    weights_by_p = reliability_weights_by_param(rel_rows)

    # 2. Generate predictions for all baselines
    all_pred_records = generate_fused_dataset(
        rag_records=rag_records,
        vlm_records=vlm_records,
        weights_by_param=weights_by_p,
        modes=FUSION_MODES,
    )
    pred_csv = write_predictions_csv(all_pred_records, out / "fusion_predictions.csv")

    # Split predictions by mode for evaluation
    f1_preds = [r for r in all_pred_records if r["fusion_mode"] == "F1_rag_only"]
    f2_preds = [r for r in all_pred_records if r["fusion_mode"] == "F2_vlm_only"]
    f3_preds = [r for r in all_pred_records if r["fusion_mode"] == "F3_equal_weight"]
    f4_preds = [r for r in all_pred_records if r["fusion_mode"] == "F4_calibrated"]

    # 3. Evaluate each against Ground Truth
    f1_metrics = evaluate_records_against_gt(f1_preds, ground_truth_records, "F1_rag_only")
    f2_metrics = evaluate_records_against_gt(f2_preds, ground_truth_records, "F2_vlm_only")
    f3_metrics = evaluate_records_against_gt(f3_preds, ground_truth_records, "F3_equal_weight")
    f4_metrics = evaluate_records_against_gt(f4_preds, ground_truth_records, "F4_calibrated")

    # Diagnostic comparison: RAG vs VLM
    diag = compare_sources("rag", "vlm", list(rag_records), list(vlm_records))

    all_metric_rows = f1_metrics + f2_metrics + f3_metrics + f4_metrics

    # Write fusion_metrics.csv
    ensure_parent(out / "fusion_metrics.csv")
    if all_metric_rows:
        fieldnames = list(all_metric_rows[0].keys())
        # ensure all keys across rows are in fieldnames
        all_keys = dict.fromkeys(k for row in all_metric_rows for k in row.keys())
        with (out / "fusion_metrics.csv").open("w", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(all_keys.keys()), extrasaction="ignore")
            writer.writeheader()
            writer.writerows(all_metric_rows)

    # 4. Build and write fusion_comparison.csv
    comparison_rows = build_fusion_comparison(f1_metrics, f2_metrics, f3_metrics, f4_metrics)
    ensure_parent(out / "fusion_comparison.csv")
    with (out / "fusion_comparison.csv").open("w", encoding="utf-8", newline="") as fh:
        comp_fields = [
            "parameter",
            "kind",
            "metric",
            "RAG_score",
            "VLM_score",
            "equal_fusion_score",
            "calibrated_fusion_score",
            "sample_count",
        ]
        writer = csv.DictWriter(fh, fieldnames=comp_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(comparison_rows)

    # 5. Manifest
    manifest = {
        "step": 14,
        "description": "Calibrated RAG + VLM Fusion",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "baselines": FUSION_MODES,
        "n_rag_records": len(rag_records),
        "n_vlm_records": len(vlm_records),
        "n_ground_truth_records": len(ground_truth_records),
        "n_fused_records": len(all_pred_records),
        "diagnostic_rag_vs_vlm": {
            "n_matched_building_ids": diag.get("n_matched_building_ids", 0),
        },
        "outputs": {
            "parameter_reliability": str(rel_csv),
            "fusion_predictions": str(pred_csv),
            "fusion_metrics": str(out / "fusion_metrics.csv"),
            "fusion_comparison": str(out / "fusion_comparison.csv"),
        },
    }
    write_json(out / "fusion_manifest.json", manifest)

    return {
        "manifest": manifest,
        "reliability_weights": weights_by_p,
        "comparison": comparison_rows,
        "outputs": manifest["outputs"],
    }
