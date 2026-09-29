"""Shared configuration and evaluation for optimization experiments.

Both RAG and VLM optimizers evaluate against ground truth only.
They never reference each other's predictions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Mapping

from seismic_damage.validation.metrics import (
    ALL_PARAMETERS,
    is_missing,
    metrics_for,
    parameter_kind,
)
from seismic_damage.validation.cross_validation import _index_by_building, _value


# Maps parameter kind to the primary metric key for comparison.
PRIMARY_METRIC_KEY: dict[str, str] = {
    "ordinal": "exact_match",
    "binary": "f1",
    "categorical": "macro_f1",
    "numerical": "mae",   # Lower is better; invert for ranking
}

RANDOM_SEED = 42


@dataclass
class ExperimentResult:
    """Result of a single optimization experiment."""

    experiment_id: str
    modality: str  # "rag" or "vlm"
    configuration: dict[str, Any]
    parameter_metrics: dict[str, dict[str, Any]]  # {param -> metric_dict}
    n_buildings: int
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    notes: str = ""

    def primary_score(self, param: str) -> float | None:
        """Return headline score for one parameter."""
        m = self.parameter_metrics.get(param, {})
        kind = parameter_kind(param)
        key = PRIMARY_METRIC_KEY.get(kind, "accuracy")
        val = m.get(key)
        if val is None:
            return None
        # For numerical (MAE/RMSE), lower = better. Return inverted score.
        if kind == "numerical":
            return 1.0 / (1.0 + float(val))
        return float(val)

    def aggregate_score(self) -> float:
        """Mean of per-parameter primary scores (only non-None, n>=3)."""
        scores: list[float] = []
        for param in ALL_PARAMETERS:
            m = self.parameter_metrics.get(param, {})
            if m.get("n", 0) < 1:
                continue
            s = self.primary_score(param)
            if s is not None:
                scores.append(s)
        return sum(scores) / len(scores) if scores else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "experiment_id": self.experiment_id,
            "modality": self.modality,
            "configuration": self.configuration,
            "n_buildings": self.n_buildings,
            "aggregate_score": round(self.aggregate_score(), 6),
            "timestamp": self.timestamp,
            "notes": self.notes,
            "parameter_metrics": self.parameter_metrics,
        }


def evaluate_against_ground_truth(
    predictions: list[Mapping[str, Any]],
    ground_truth: list[Mapping[str, Any]],
) -> tuple[dict[str, dict[str, Any]], int]:
    """Compute per-parameter metrics of predictions vs ground truth.

    Returns:
        (parameter_metrics, n_matched_buildings)
    """
    pred_idx = _index_by_building(predictions)
    gt_idx = _index_by_building(ground_truth)
    shared = sorted(set(pred_idx) & set(gt_idx))

    per_param: dict[str, dict[str, Any]] = {}
    for param in sorted(ALL_PARAMETERS):
        pred_vals = [_value(pred_idx[bid], param) for bid in shared]
        gt_vals = [_value(gt_idx[bid], param) for bid in shared]
        per_param[param] = metrics_for(param, pred_vals, gt_vals)

    return per_param, len(shared)


def experiment_results_to_rows(results: list[ExperimentResult]) -> list[dict[str, Any]]:
    """Flatten experiment results to one row per (experiment, parameter)."""
    rows: list[dict[str, Any]] = []
    for result in results:
        for param, metric in result.parameter_metrics.items():
            row: dict[str, Any] = {
                "experiment_id": result.experiment_id,
                "modality": result.modality,
                "parameter": param,
                "kind": parameter_kind(param),
                "n_buildings": metric.get("n", 0),
                "aggregate_score": round(result.aggregate_score(), 6),
                "notes": result.notes,
                **result.configuration,
            }
            row.update({f"metric_{k}": v for k, v in metric.items() if k not in ("kind", "parameter")})
            rows.append(row)
    return rows


def select_best_experiment(
    results: list[ExperimentResult],
    *,
    metric: str = "aggregate_score",
) -> ExperimentResult:
    """Select the experiment with the highest aggregate score."""
    if not results:
        raise ValueError("No experiments to select from.")
    return max(results, key=lambda r: r.aggregate_score())
