"""Compute reliability weights for RAG and VLM based on historical error analysis."""

import json
from pathlib import Path
from typing import Any, Mapping

from seismic_damage.validation.metrics import ALL_PARAMETERS, parameter_kind
from seismic_damage.io_utils import ensure_parent


def _score_for_metric(metric: dict[str, Any], kind: str) -> float:
    """Convert raw metrics into a 0.0 - 1.0 confidence score."""
    if not metric or metric.get("n", 0) == 0:
        return 0.0

    if kind in ("categorical", "binary"):
        # F1 score is typically bounded 0-1
        score = metric.get("f1")
        if score is None:
            score = metric.get("macro_f1")
        if score is None:
            score = metric.get("accuracy", 0.0)
        return float(max(0.0, min(1.0, score)))

    elif kind == "ordinal":
        # Can use accuracy or scale QWK
        acc = metric.get("accuracy", 0.0)
        qwk = metric.get("qwk", 0.0)
        # Weighted blend favoring exact accuracy but giving credit for QWK
        return float(max(0.0, min(1.0, 0.7 * acc + 0.3 * max(0.0, qwk))))

    elif kind == "numerical":
        # Invert MAE into a confidence score (1 / 1 + MAE)
        mae = metric.get("mae")
        if mae is not None:
            return 1.0 / (1.0 + float(mae))
        return 0.0

    return 0.0


def compute_reliability_weights(
    error_report: Mapping[str, Any]
) -> dict[str, Any]:
    """Derive parameter-level confidence weights for RAG and VLM."""
    
    rag_metrics = error_report.get("rag_metrics", {})
    vlm_metrics = error_report.get("vlm_metrics", {})
    
    weights = {}

    for param in ALL_PARAMETERS:
        kind = parameter_kind(param)
        
        r_m = rag_metrics.get(param, {})
        v_m = vlm_metrics.get(param, {})
        
        rag_conf = _score_for_metric(r_m, kind)
        vlm_conf = _score_for_metric(v_m, kind)
        
        # Determine primary source
        if rag_conf == 0.0 and vlm_conf == 0.0:
            primary = "none"
        else:
            primary = "rag" if rag_conf >= vlm_conf else "vlm"
            
        # Relative fusion weights
        total_conf = rag_conf + vlm_conf
        if total_conf > 0:
            rag_weight = rag_conf / total_conf
            vlm_weight = vlm_conf / total_conf
        else:
            rag_weight = 0.5
            vlm_weight = 0.5
            
        weights[param] = {
            "kind": kind,
            "rag_confidence": round(rag_conf, 4),
            "vlm_confidence": round(vlm_conf, 4),
            "primary_source": primary,
            "fusion_weight_rag": round(rag_weight, 4),
            "fusion_weight_vlm": round(vlm_weight, 4),
        }

    return {
        "metadata": {
            "description": "Dynamic reliability weights derived from Ground Truth error analysis.",
            "n_buildings_evaluated": error_report.get("n_buildings_evaluated", 0)
        },
        "parameter_weights": weights
    }
