"""Robustness evaluation: metrics per condition per building.

Reuses the Step 7 metrics framework (seismic_damage.validation.metrics).
Missing VLM evidence is represented as None — never as zero.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from seismic_damage.validation.metrics import (
    ALL_PARAMETERS,
    is_missing,
    metrics_for,
    parameter_kind,
)


# Primary metric to report per parameter kind (for summary table).
_PRIMARY_METRIC: dict[str, str] = {
    "ordinal": "exact_match",
    "binary": "f1",
    "categorical": "macro_f1",
    "numerical": "mae",
}


def _primary_score(metric_dict: dict[str, Any], kind: str) -> float | None:
    """Extract the single headline metric value for a given kind."""
    key = _PRIMARY_METRIC.get(kind, "accuracy")
    val = metric_dict.get(key)
    if val is None:
        return None
    return float(val)


def evaluate_condition(
    condition_name: str,
    vlm_records: list[Mapping[str, Any]],
    ground_truth_records: list[Mapping[str, Any]],
    *,
    availability_meta: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Evaluate VLM predictions under one availability/degradation condition.

    Returns:
        List of per-(building, parameter) result rows with all required fields.
    """
    # Index ground truth by building_id
    gt_index: dict[str, Mapping[str, Any]] = {}
    for row in ground_truth_records:
        bid = str(row.get("building_id") or "")
        if bid:
            gt_index[bid] = row

    # Index VLM records by building_id
    vlm_index: dict[str, Mapping[str, Any]] = {}
    for row in vlm_records:
        bid = str(row.get("building_id") or "")
        if bid:
            vlm_index[bid] = row

    # Index availability metadata
    avail_meta: dict[str, dict[str, Any]] = {}
    for meta in (availability_meta or []):
        bid = str(meta.get("building_id") or "")
        if bid:
            avail_meta[bid] = meta

    rows: list[dict[str, Any]] = []
    for bid in sorted(gt_index):
        gt_row = gt_index[bid]
        vlm_row = vlm_index.get(bid)
        meta = avail_meta.get(bid, {})

        for param in sorted(ALL_PARAMETERS):
            gt_val = _get_param(gt_row, param)
            vlm_val = _get_param(vlm_row, param) if vlm_row is not None else None

            rows.append(
                {
                    "building_id": bid,
                    "condition": condition_name,
                    "parameter": param,
                    "ground_truth": gt_val,
                    "vlm_prediction": vlm_val,
                    "availability_fraction": meta.get("availability_fraction"),
                    "image_count_available": meta.get("image_count_available"),
                    "image_count_total": meta.get("image_count_total"),
                }
            )
    return rows


def _get_param(row: Mapping[str, Any] | None, param: str) -> Any:
    """Extract a parameter value from a flat or nested record."""
    if row is None:
        return None
    val = row.get(param)
    if not is_missing(val):
        return val
    nested = row.get("parameters")
    if isinstance(nested, Mapping):
        val = nested.get(param)
    if not is_missing(val):
        return val
    # Try typology / vulnerability / damage sub-dicts
    for key in ("typology", "vulnerability", "damage"):
        sub = row.get(key)
        if isinstance(sub, Mapping):
            val = sub.get(param)
            if not is_missing(val):
                return val
    return None


def aggregate_condition_metrics(
    result_rows: list[dict[str, Any]],
    condition_name: str,
) -> list[dict[str, Any]]:
    """Compute per-parameter metrics across all buildings for one condition.

    Returns:
        One row per parameter with condition-level metric summary.
    """
    # Group by parameter
    by_param: dict[str, tuple[list[Any], list[Any]]] = {}
    for row in result_rows:
        if row["condition"] != condition_name:
            continue
        param = row["parameter"]
        if param not in by_param:
            by_param[param] = ([], [])
        by_param[param][0].append(row["ground_truth"])
        by_param[param][1].append(row["vlm_prediction"])

    summary_rows: list[dict[str, Any]] = []
    for param in sorted(ALL_PARAMETERS):
        gt_vals, vlm_vals = by_param.get(param, ([], []))
        kind = parameter_kind(param)
        metric = metrics_for(param, gt_vals, vlm_vals)
        primary = _primary_score(metric, kind)
        summary_rows.append(
            {
                "condition": condition_name,
                "parameter": param,
                "kind": kind,
                "n_buildings": metric.get("n", 0),
                "primary_metric_name": _PRIMARY_METRIC.get(kind, "accuracy"),
                "primary_metric_value": primary,
                **{k: v for k, v in metric.items() if k not in ("kind", "parameter", "n")},
            }
        )
    return summary_rows


def build_robustness_summary(
    all_condition_metrics: list[dict[str, Any]],
    baseline_condition: str = "A_100pct",
) -> list[dict[str, Any]]:
    """Compute performance degradation relative to the 100% baseline.

    Args:
        all_condition_metrics: Combined output of aggregate_condition_metrics for all conditions.
        baseline_condition: The condition name representing full availability.

    Returns:
        List of summary rows with degradation_from_full_availability.
    """
    # Index baseline scores
    baseline: dict[str, float | None] = {}
    for row in all_condition_metrics:
        if row["condition"] == baseline_condition:
            baseline[row["parameter"]] = row.get("primary_metric_value")

    summary: list[dict[str, Any]] = []
    for row in all_condition_metrics:
        param = row["parameter"]
        base_score = baseline.get(param)
        cur_score = row.get("primary_metric_value")
        if base_score is not None and cur_score is not None:
            degradation = float(cur_score) - float(base_score)
        else:
            degradation = None

        summary.append(
            {
                **row,
                "baseline_condition": baseline_condition,
                "baseline_value": base_score,
                "degradation_from_full_availability": degradation,
            }
        )
    return summary
