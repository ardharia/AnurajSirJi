"""Parameter-level metrics with explicit missing-value handling."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

ORDINAL_PARAMETERS = {"damage_grade"}
BINARY_PARAMETERS = {"soft_story_failure", "wall_failure"}
NUMERICAL_PARAMETERS = {"number_of_stories", "pga", "magnitude"}
CATEGORICAL_PARAMETERS = {
    "structural_system",
    "material_type",
    "crack_severity",
    "crack_pattern",
    "collapse_mode",
    "soil_type",
}

ALL_PARAMETERS = (
    ORDINAL_PARAMETERS | BINARY_PARAMETERS | NUMERICAL_PARAMETERS | CATEGORICAL_PARAMETERS
)


def is_missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str) and not value.strip():
        return True
    if isinstance(value, float) and np.isnan(value):
        return True
    return False


def parameter_kind(name: str) -> str:
    if name in ORDINAL_PARAMETERS:
        return "ordinal"
    if name in BINARY_PARAMETERS:
        return "binary"
    if name in NUMERICAL_PARAMETERS:
        return "numerical"
    if name in CATEGORICAL_PARAMETERS:
        return "categorical"
    return "categorical"


def _paired(left: Sequence[Any], right: Sequence[Any]) -> tuple[list[Any], list[Any]]:
    xs: list[Any] = []
    ys: list[Any] = []
    for a, b in zip(left, right):
        if is_missing(a) or is_missing(b):
            continue
        xs.append(a)
        ys.append(b)
    return xs, ys


def ordinal_metrics(left: Sequence[Any], right: Sequence[Any]) -> dict[str, Any]:
    xs, ys = _paired(left, right)
    n = len(xs)
    if n == 0:
        return {"kind": "ordinal", "n": 0}
    x = np.asarray(xs, dtype=float)
    y = np.asarray(ys, dtype=float)
    mae = float(np.mean(np.abs(x - y)))
    exact = float(np.mean(x == y))
    within_one = float(np.mean(np.abs(x - y) <= 1))
    return {
        "kind": "ordinal",
        "n": n,
        "mae": mae,
        "exact_match": exact,
        "within_one": within_one,
        "quadratic_weighted_kappa": _quadratic_weighted_kappa(x, y),
    }


def binary_metrics(left: Sequence[Any], right: Sequence[Any]) -> dict[str, Any]:
    xs, ys = _paired(left, right)
    n = len(xs)
    if n == 0:
        return {"kind": "binary", "n": 0}
    x = np.asarray([_as_bool(v) for v in xs], dtype=int)
    y = np.asarray([_as_bool(v) for v in ys], dtype=int)
    tp = int(np.sum((x == 1) & (y == 1)))
    fp = int(np.sum((x == 0) & (y == 1)))
    fn = int(np.sum((x == 1) & (y == 0)))
    tn = int(np.sum((x == 0) & (y == 0)))
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        "kind": "binary",
        "n": n,
        "accuracy": float(np.mean(x == y)),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }


def numerical_metrics(left: Sequence[Any], right: Sequence[Any]) -> dict[str, Any]:
    xs, ys = _paired(left, right)
    n = len(xs)
    if n == 0:
        return {"kind": "numerical", "n": 0}
    x = np.asarray(xs, dtype=float)
    y = np.asarray(ys, dtype=float)
    err = x - y
    return {
        "kind": "numerical",
        "n": n,
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err**2))),
    }


def categorical_metrics(left: Sequence[Any], right: Sequence[Any]) -> dict[str, Any]:
    xs, ys = _paired(left, right)
    n = len(xs)
    if n == 0:
        return {"kind": "categorical", "n": 0}
    x = [str(v).strip().lower() for v in xs]
    y = [str(v).strip().lower() for v in ys]
    exact = float(np.mean([a == b for a, b in zip(x, y)]))
    labels = sorted(set(x) | set(y))
    f1s: list[float] = []
    for label in labels:
        tp = sum(a == label and b == label for a, b in zip(x, y))
        fp = sum(a != label and b == label for a, b in zip(x, y))
        fn = sum(a == label and b != label for a, b in zip(x, y))
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1s.append(2 * precision * recall / (precision + recall) if (precision + recall) else 0.0)
    return {
        "kind": "categorical",
        "n": n,
        "accuracy": exact,
        "macro_f1": float(np.mean(f1s)) if f1s else 0.0,
    }


def metrics_for(name: str, left: Sequence[Any], right: Sequence[Any]) -> dict[str, Any]:
    kind = parameter_kind(name)
    if kind == "ordinal":
        result = ordinal_metrics(left, right)
    elif kind == "binary":
        result = binary_metrics(left, right)
    elif kind == "numerical":
        result = numerical_metrics(left, right)
    else:
        result = categorical_metrics(left, right)
    result["parameter"] = name
    return result


def _as_bool(value: Any) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, np.integer)) and value in (0, 1):
        return int(value)
    text = str(value).strip().lower()
    if text in {"true", "yes", "1"}:
        return 1
    if text in {"false", "no", "0"}:
        return 0
    raise ValueError(f"non-binary value: {value!r}")


def _quadratic_weighted_kappa(x: np.ndarray, y: np.ndarray) -> float | None:
    labels = np.unique(np.concatenate([x, y]))
    if labels.size < 2:
        return None
    min_label = int(labels.min())
    max_label = int(labels.max())
    size = max_label - min_label + 1
    o = np.zeros((size, size), dtype=float)
    for a, b in zip(x.astype(int), y.astype(int)):
        o[a - min_label, b - min_label] += 1
    if o.sum() == 0:
        return None
    o /= o.sum()
    hist_x = o.sum(axis=1)
    hist_y = o.sum(axis=0)
    e = np.outer(hist_x, hist_y)
    weights = np.zeros((size, size), dtype=float)
    for i in range(size):
        for j in range(size):
            weights[i, j] = ((i - j) ** 2) / ((size - 1) ** 2)
    denom = float(np.sum(weights * e))
    if denom == 0:
        return None
    return float(1.0 - np.sum(weights * o) / denom)
