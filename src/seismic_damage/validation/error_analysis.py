"""Comprehensive error analysis module for comparing RAG and VLM outputs against ground truth."""

import json
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from seismic_damage.validation.cross_validation import _index_by_building, _value
from seismic_damage.validation.metrics import ALL_PARAMETERS, metrics_for, is_missing
from seismic_damage.io_utils import ensure_parent


def _build_differential_analysis(
    gt_index: dict[str, Mapping[str, Any]],
    rag_index: dict[str, Mapping[str, Any]],
    vlm_index: dict[str, Mapping[str, Any]],
    shared_bids: list[str],
) -> dict[str, Any]:
    """Perform per-parameter, per-building differential analysis."""
    diff_report: dict[str, Any] = {}

    for param in sorted(ALL_PARAMETERS):
        param_diff = {
            "rag_better": 0,
            "vlm_better": 0,
            "both_correct": 0,
            "both_wrong": 0,
            "both_missing": 0,
            "details": []
        }

        for bid in shared_bids:
            gt_val = _value(gt_index[bid], param)
            rag_val = _value(rag_index[bid], param)
            vlm_val = _value(vlm_index[bid], param)

            if is_missing(gt_val):
                continue
                
            gt_str = str(gt_val).lower().strip()
            rag_str = str(rag_val).lower().strip() if not is_missing(rag_val) else None
            vlm_str = str(vlm_val).lower().strip() if not is_missing(vlm_val) else None
            
            rag_correct = rag_str == gt_str
            vlm_correct = vlm_str == gt_str

            detail = {
                "building_id": bid,
                "ground_truth": gt_val,
                "rag": rag_val,
                "vlm": vlm_val,
                "rag_correct": rag_correct,
                "vlm_correct": vlm_correct,
            }

            if rag_correct and vlm_correct:
                param_diff["both_correct"] += 1
                detail["category"] = "both_correct"
            elif not rag_correct and not vlm_correct:
                if is_missing(rag_val) and is_missing(vlm_val):
                    param_diff["both_missing"] += 1
                    detail["category"] = "both_missing"
                else:
                    param_diff["both_wrong"] += 1
                    detail["category"] = "both_wrong"
            elif rag_correct and not vlm_correct:
                param_diff["rag_better"] += 1
                detail["category"] = "rag_better"
            elif vlm_correct and not rag_correct:
                param_diff["vlm_better"] += 1
                detail["category"] = "vlm_better"
                
            param_diff["details"].append(detail)
            
        diff_report[param] = param_diff

    return diff_report


def run_error_analysis(
    gt_records: list[Mapping[str, Any]],
    rag_records: list[Mapping[str, Any]],
    vlm_records: list[Mapping[str, Any]],
    output_dir: Path | str | None = None,
) -> dict[str, Any]:
    """Generate comprehensive error report comparing RAG and VLM to GT."""
    gt_idx = _index_by_building(gt_records)
    rag_idx = _index_by_building(rag_records)
    vlm_idx = _index_by_building(vlm_records)

    # Intersection of all three
    shared = sorted(set(gt_idx) & set(rag_idx) & set(vlm_idx))

    # Compute metrics for RAG
    rag_metrics = {}
    vlm_metrics = {}

    for param in ALL_PARAMETERS:
        gt_vals = [_value(gt_idx[bid], param) for bid in shared]
        rag_vals = [_value(rag_idx[bid], param) for bid in shared]
        vlm_vals = [_value(vlm_idx[bid], param) for bid in shared]

        rag_metrics[param] = metrics_for(param, gt_vals, rag_vals)
        vlm_metrics[param] = metrics_for(param, gt_vals, vlm_vals)

    # Build DataFrame
    df_rows = []
    for param in ALL_PARAMETERS:
        r_m = rag_metrics[param]
        v_m = vlm_metrics[param]
        
        kind = r_m.get("kind", "unknown")
        
        def _row(modality, m):
            return {
                "parameter": param,
                "kind": kind,
                "modality": modality,
                "n": m.get("n", 0),
                "accuracy": m.get("accuracy", m.get("exact_match")),
                "precision": m.get("precision"),
                "recall": m.get("recall"),
                "f1": m.get("f1", m.get("macro_f1")),
                "mae": m.get("mae"),
                "rmse": m.get("rmse"),
                "qwk": m.get("qwk")
            }
            
        df_rows.append(_row("rag", r_m))
        df_rows.append(_row("vlm", v_m))

    df = pd.DataFrame(df_rows)

    # Differential Analysis
    diff_report = _build_differential_analysis(gt_idx, rag_idx, vlm_idx, shared)

    full_report = {
        "n_buildings_evaluated": len(shared),
        "rag_metrics": rag_metrics,
        "vlm_metrics": vlm_metrics,
        "differential": diff_report,
    }

    if output_dir:
        base = ensure_parent(Path(output_dir) / ".gitkeep").parent
        base.mkdir(parents=True, exist_ok=True)
        
        # Save JSON
        with open(base / "error_analysis_report.json", "w") as f:
            json.dump(full_report, f, indent=2)
            
        # Save CSV
        df.to_csv(base / "error_analysis_summary.csv", index=False)

    return full_report
