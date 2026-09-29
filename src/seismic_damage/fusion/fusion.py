"""Step 14: Core fusion logic combining RAG and VLM predictions.

Implements the four required baselines:
  F1: RAG only
  F2: VLM only
  F3: Equal-weight fusion (50:50)
  F4: Calibrated fusion (weights from empirical parameter reliability)

Preserves missing-source semantics:
  Both available  → combined via weights
  RAG only        → RAG result (missing VLM is never treated as zero)
  VLM only        → VLM result (missing RAG is never treated as zero)
  Both missing    → None (never invent a prediction)
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Mapping, Sequence

from seismic_damage.config.settings import PROJECT_ROOT
from seismic_damage.fusion.reliability import (
    compute_parameter_reliability,
    reliability_weights_by_param,
    write_reliability_csv,
)
from seismic_damage.fusion.weighting import fuse_parameter
from seismic_damage.io_utils import ensure_parent
from seismic_damage.validation.cross_validation import _index_by_building, _value
from seismic_damage.validation.metrics import (
    ALL_PARAMETERS,
    is_missing,
    parameter_kind,
)

FUSION_DIR = PROJECT_ROOT / "data" / "processed" / "fusion"

FUSION_MODES = ["F1_rag_only", "F2_vlm_only", "F3_equal_weight", "F4_calibrated"]


def fuse_building(
    building_id: str,
    rag_rec: Mapping[str, Any] | None,
    vlm_rec: Mapping[str, Any] | None,
    weights_by_param: dict[str, dict[str, float]],
    mode: str = "F4_calibrated",
) -> dict[str, Any]:
    """Fuse predictions for a single building across all canonical parameters.

    Args:
        building_id: Canonical building identifier.
        rag_rec: Building-level RAG record, or None if unavailable.
        vlm_rec: Building-level VLM record, or None if unavailable.
        weights_by_param: {param -> {'rag_weight': float, 'vlm_weight': float}}.
        mode: One of 'F1_rag_only', 'F2_vlm_only', 'F3_equal_weight', 'F4_calibrated'.

    Returns:
        Dict with building_id, fusion_mode, and each parameter value or None.
    """
    out: dict[str, Any] = {
        "building_id": building_id,
        "fusion_mode": mode,
        "rag_available": rag_rec is not None,
        "vlm_available": vlm_rec is not None,
    }

    for param in sorted(ALL_PARAMETERS):
        rag_val = _value(rag_rec, param) if rag_rec is not None else None
        vlm_val = _value(vlm_rec, param) if vlm_rec is not None else None

        if mode == "F1_rag_only":
            out[param] = None if is_missing(rag_val) else rag_val
            continue

        if mode == "F2_vlm_only":
            out[param] = None if is_missing(vlm_val) else vlm_val
            continue

        if mode == "F3_equal_weight":
            rw, vw = 0.5, 0.5
        elif mode == "F4_calibrated":
            p_weights = weights_by_param.get(param, {"rag_weight": 0.5, "vlm_weight": 0.5})
            rw = p_weights.get("rag_weight", 0.5)
            vw = p_weights.get("vlm_weight", 0.5)
            if rw == 0.0 and vw == 0.0:
                rw, vw = 0.5, 0.5
        else:
            raise ValueError(f"Unknown fusion mode: {mode}")

        fused_val = fuse_parameter(param, rag_val, vlm_val, rw, vw)
        out[param] = fused_val

    return out


def generate_fused_dataset(
    rag_records: list[Mapping[str, Any]],
    vlm_records: list[Mapping[str, Any]],
    weights_by_param: dict[str, dict[str, float]],
    modes: Sequence[str] = ("F4_calibrated",),
) -> list[dict[str, Any]]:
    """Generate fused records for all buildings present in RAG or VLM datasets.

    Args:
        rag_records: Building-level RAG predictions.
        vlm_records: Building-level VLM predictions.
        weights_by_param: Per-parameter reliability weights.
        modes: List of fusion modes to generate.

    Returns:
        List of fused building records.
    """
    rag_idx = _index_by_building(rag_records)
    vlm_idx = _index_by_building(vlm_records)
    all_bids = sorted(set(rag_idx) | set(vlm_idx))

    records: list[dict[str, Any]] = []
    for mode in modes:
        for bid in all_bids:
            rec = fuse_building(
                building_id=bid,
                rag_rec=rag_idx.get(bid),
                vlm_rec=vlm_idx.get(bid),
                weights_by_param=weights_by_param,
                mode=mode,
            )
            records.append(rec)
    return records


def write_predictions_csv(records: list[dict[str, Any]], path: Path) -> Path:
    """Save fusion predictions to CSV."""
    ensure_parent(path)
    if not records:
        path.write_text("", encoding="utf-8")
        return path

    base_cols = ["building_id", "fusion_mode", "rag_available", "vlm_available"]
    param_cols = sorted(ALL_PARAMETERS)
    fieldnames = base_cols + [p for p in param_cols if p not in base_cols]

    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)
    return path
