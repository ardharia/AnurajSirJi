"""Step 14 fusion weighting and value combination.

Handles all parameter types correctly:
- Ordinal: numeric interpolation via integer encoding
- Binary: boolean selection by weight
- Categorical: reliability-weighted categorical decision
- Numerical: weighted average

Missing-source logic:
  Both available  → calibrated fusion
  RAG only        → RAG result (never invent VLM)
  VLM only        → VLM result (never invent RAG)
  Both missing    → missing (never invent a prediction)
"""

from __future__ import annotations

from typing import Any

from seismic_damage.validation.metrics import is_missing, parameter_kind


# EMS-98 ordinal encoding for damage_grade (integer 0..5)
DAMAGE_GRADE_ENCODING: dict[int, int] = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5}


def fuse_ordinal(
    rag_val: Any,
    vlm_val: Any,
    rag_weight: float,
    vlm_weight: float,
) -> int | None:
    """Fuse two ordinal values using weighted average, rounded to nearest integer.

    Returns None if both sources are missing.
    Falls back to the available source if the other is missing.
    """
    rag_miss = is_missing(rag_val)
    vlm_miss = is_missing(vlm_val)

    if rag_miss and vlm_miss:
        return None
    if rag_miss:
        try:
            return int(round(float(vlm_val)))
        except (TypeError, ValueError):
            return None
    if vlm_miss:
        try:
            return int(round(float(rag_val)))
        except (TypeError, ValueError):
            return None

    try:
        r = float(rag_val)
        v = float(vlm_val)
    except (TypeError, ValueError):
        # Cannot numeric-fuse: fall back to higher-weight source
        return int(round(float(rag_val))) if rag_weight >= vlm_weight else int(round(float(vlm_val)))

    # Weighted average, rounded to integer
    fused = rag_weight * r + vlm_weight * v
    return int(round(fused))


def fuse_binary(
    rag_val: Any,
    vlm_val: Any,
    rag_weight: float,
    vlm_weight: float,
) -> bool | None:
    """Fuse two binary values: select the value from the higher-weight source.

    If weights are equal, RAG wins (arbitrary but deterministic tie-break).
    """
    rag_miss = is_missing(rag_val)
    vlm_miss = is_missing(vlm_val)

    if rag_miss and vlm_miss:
        return None
    if rag_miss:
        return _as_bool(vlm_val)
    if vlm_miss:
        return _as_bool(rag_val)

    # Both available: select from higher-weight source
    chosen_val = rag_val if rag_weight >= vlm_weight else vlm_val
    return _as_bool(chosen_val)


def fuse_categorical(
    rag_val: Any,
    vlm_val: Any,
    rag_weight: float,
    vlm_weight: float,
) -> str | None:
    """Fuse two categorical values via reliability-weighted selection.

    If both sources agree, return the shared value.
    If they disagree, return the value from the higher-weight source.
    If weights are equal, RAG wins (deterministic tie-break).
    """
    rag_miss = is_missing(rag_val)
    vlm_miss = is_missing(vlm_val)

    if rag_miss and vlm_miss:
        return None
    if rag_miss:
        return str(vlm_val).strip()
    if vlm_miss:
        return str(rag_val).strip()

    rag_str = str(rag_val).strip().lower()
    vlm_str = str(vlm_val).strip().lower()

    if rag_str == vlm_str:
        return str(rag_val).strip()  # Agreement: return original
    # Disagreement: pick higher-weight source
    chosen = rag_val if rag_weight >= vlm_weight else vlm_val
    return str(chosen).strip()


def fuse_numerical(
    rag_val: Any,
    vlm_val: Any,
    rag_weight: float,
    vlm_weight: float,
) -> float | None:
    """Fuse two numerical values via weighted average."""
    rag_miss = is_missing(rag_val)
    vlm_miss = is_missing(vlm_val)

    if rag_miss and vlm_miss:
        return None
    if rag_miss:
        try:
            return float(vlm_val)
        except (TypeError, ValueError):
            return None
    if vlm_miss:
        try:
            return float(rag_val)
        except (TypeError, ValueError):
            return None

    try:
        r = float(rag_val)
        v = float(vlm_val)
    except (TypeError, ValueError):
        return None

    return rag_weight * r + vlm_weight * v


def fuse_parameter(
    param: str,
    rag_val: Any,
    vlm_val: Any,
    rag_weight: float,
    vlm_weight: float,
) -> Any:
    """Dispatch fusion to the correct handler for this parameter type.

    Returns None if both sources are missing.
    Never invents a prediction.
    """
    kind = parameter_kind(param)

    if kind == "ordinal":
        return fuse_ordinal(rag_val, vlm_val, rag_weight, vlm_weight)
    if kind == "binary":
        return fuse_binary(rag_val, vlm_val, rag_weight, vlm_weight)
    if kind == "categorical":
        return fuse_categorical(rag_val, vlm_val, rag_weight, vlm_weight)
    if kind == "numerical":
        return fuse_numerical(rag_val, vlm_val, rag_weight, vlm_weight)

    # Unknown kind: fall back to higher-weight source
    if is_missing(rag_val) and is_missing(vlm_val):
        return None
    if is_missing(rag_val):
        return vlm_val
    if is_missing(vlm_val):
        return rag_val
    return rag_val if rag_weight >= vlm_weight else vlm_val


def _as_bool(value: Any) -> bool | None:
    """Safely convert a value to bool."""
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        if value == 1:
            return True
        if value == 0:
            return False
        return None
    text = str(value).strip().lower()
    if text in {"true", "yes", "1"}:
        return True
    if text in {"false", "no", "0"}:
        return False
    return None
