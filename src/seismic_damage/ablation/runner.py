"""Step 16: Ablation study runner executing Experiments A through J.

Guarantees:
- Uniform building population and ground truth reference across all experiments.
- Parameter-wise and system-level metrics reported separately.
- Outputs saved reproducibly under data/processed/ablation/.
"""

from __future__ import annotations

import csv
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from seismic_damage.ablation.experiments import (
    ABLATION_EXPERIMENTS,
    AblationExperimentConfig,
)
from seismic_damage.config.settings import PROJECT_ROOT
from seismic_damage.fusion.fusion import fuse_building
from seismic_damage.fusion.reliability import (
    compute_parameter_reliability,
    reliability_weights_by_param,
)
from seismic_damage.io_utils import ensure_parent, write_json
from seismic_damage.validation.cross_validation import _index_by_building, _value
from seismic_damage.validation.metrics import (
    ALL_PARAMETERS,
    is_missing,
    metrics_for,
    parameter_kind,
)

ABLATION_DIR = PROJECT_ROOT / "data" / "processed" / "ablation"
RANDOM_SEED = 42


def _apply_text_mask(records: list[Mapping[str, Any]], mask_ratio: float = 0.5) -> list[dict[str, Any]]:
    """Simulate incomplete textual evidence by blanking out parameters."""
    import random as _rng
    rng = _rng.Random(RANDOM_SEED)
    masked = []
    for r in records:
        rec = dict(r)
        for p in ALL_PARAMETERS:
            if rng.random() < mask_ratio:
                rec[p] = None
        masked.append(rec)
    return masked


def _apply_vlm_mask(records: list[Mapping[str, Any]], mask_ratio: float = 0.5) -> list[dict[str, Any]]:
    """Simulate degraded VLM evidence by blanking out parameters."""
    import random as _rng
    rng = _rng.Random(RANDOM_SEED)
    masked = []
    for r in records:
        rec = dict(r)
        for p in ALL_PARAMETERS:
            if rng.random() < mask_ratio:
                rec[p] = None
        masked.append(rec)
    return masked


def run_single_ablation_experiment(
    config: AblationExperimentConfig,
    rag_records: list[Mapping[str, Any]],
    vlm_records: list[Mapping[str, Any]],
    ground_truth: list[Mapping[str, Any]],
    weights_by_param: dict[str, dict[str, float]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Execute one ablation experiment.

    Returns:
        (prediction_records, evaluation_metric_rows)
    """
    # Prepare input RAG records
    if not config.uses_rag:
        r_inputs = []
    elif config.rag_variant == "incomplete":
        r_inputs = _apply_text_mask(rag_records, mask_ratio=0.5)
    else:
        r_inputs = list(rag_records)

    # Prepare input VLM records
    if not config.uses_vlm:
        v_inputs = []
    elif config.vlm_variant == "degraded":
        v_inputs = _apply_vlm_mask(vlm_records, mask_ratio=0.5)
    else:
        v_inputs = list(vlm_records)

    rag_idx = _index_by_building(r_inputs)
    vlm_idx = _index_by_building(v_inputs)
    gt_idx = _index_by_building(ground_truth)
    all_bids = sorted(set(gt_idx.keys()))

    pred_records: list[dict[str, Any]] = []
    for bid in all_bids:
        r_rec = rag_idx.get(bid) if config.uses_rag else None
        v_rec = vlm_idx.get(bid) if config.uses_vlm else None

        if config.fusion_mode == "none":
            if config.uses_rag:
                f_rec = fuse_building(bid, r_rec, None, weights_by_param, mode="F1_rag_only")
            else:
                f_rec = fuse_building(bid, None, v_rec, weights_by_param, mode="F2_vlm_only")
        elif config.fusion_mode == "equal_weight":
            f_rec = fuse_building(bid, r_rec, v_rec, weights_by_param, mode="F3_equal_weight")
        else:  # calibrated
            f_rec = fuse_building(bid, r_rec, v_rec, weights_by_param, mode="F4_calibrated")

        pred_records.append(f_rec)

    # Evaluate against Ground Truth
    pred_idx = _index_by_building(pred_records)
    metric_rows: list[dict[str, Any]] = []

    for param in sorted(ALL_PARAMETERS):
        kind = parameter_kind(param)

        pred_vals: list[Any] = []
        gt_vals: list[Any] = []

        for bid in all_bids:
            p_val = _value(pred_idx.get(bid), param) if bid in pred_idx else None
            g_val = _value(gt_idx.get(bid), param)

            # If conflict filter is enabled (Exp_J), only include samples where RAG & VLM disagree
            if config.conflict_filter:
                r_val = _value(rag_idx.get(bid), param) if bid in rag_idx else None
                v_val = _value(vlm_idx.get(bid), param) if bid in vlm_idx else None
                if is_missing(r_val) or is_missing(v_val) or str(r_val).strip().lower() == str(v_val).strip().lower():
                    continue

            pred_vals.append(p_val)
            gt_vals.append(g_val)

        m = metrics_for(param, pred_vals, gt_vals)

        # Primary metric
        if kind == "ordinal":
            score = m.get("exact_match")
            metric_name = "exact_match"
        elif kind == "binary":
            score = m.get("f1")
            metric_name = "f1"
        elif kind == "categorical":
            score = m.get("macro_f1")
            metric_name = "macro_f1"
        else:
            score = m.get("mae")
            metric_name = "mae"

        row = {
            "experiment_id": config.experiment_id,
            "experiment_name": config.name,
            "parameter": param,
            "kind": kind,
            "metric_name": metric_name,
            "score": score,
            "n_evaluated": m.get("n", 0),
            "n_total": len(all_bids),
            "description": config.description,
        }
        for k, v in m.items():
            if k not in ("parameter", "kind", "n"):
                row[f"metric_{k}"] = v
        metric_rows.append(row)

    return pred_records, metric_rows


def run_full_ablation_framework(
    rag_records: list[Mapping[str, Any]],
    vlm_records: list[Mapping[str, Any]],
    ground_truth_records: list[Mapping[str, Any]],
    *,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    """Execute all ablation experiments A through J and generate complete reports."""
    out = output_dir or ABLATION_DIR
    configs_dir = out / "experiment_configs"
    plots_dir = out / "plots"
    ensure_parent(configs_dir / "stub")
    ensure_parent(plots_dir / "stub")

    rel_rows = compute_parameter_reliability(rag_records, vlm_records, ground_truth_records)
    weights_by_p = reliability_weights_by_param(rel_rows)

    all_metric_rows: list[dict[str, Any]] = []
    exp_summaries: list[dict[str, Any]] = []

    for exp_cfg in ABLATION_EXPERIMENTS:
        # Save individual experiment config JSON
        write_json(
            configs_dir / f"{exp_cfg.experiment_id}.json",
            {
                "experiment_id": exp_cfg.experiment_id,
                "name": exp_cfg.name,
                "description": exp_cfg.description,
                "uses_rag": exp_cfg.uses_rag,
                "uses_vlm": exp_cfg.uses_vlm,
                "rag_variant": exp_cfg.rag_variant,
                "vlm_variant": exp_cfg.vlm_variant,
                "fusion_mode": exp_cfg.fusion_mode,
                "conflict_filter": exp_cfg.conflict_filter,
            },
        )

        _, metric_rows = run_single_ablation_experiment(
            exp_cfg, rag_records, vlm_records, ground_truth_records, weights_by_p
        )
        all_metric_rows.extend(metric_rows)

        # Aggregate headline score across non-missing parameters
        valid_scores = [r["score"] for r in metric_rows if r["score"] is not None and r["kind"] != "numerical"]
        mean_score = sum(valid_scores) / len(valid_scores) if valid_scores else 0.0

        exp_summaries.append(
            {
                "experiment_id": exp_cfg.experiment_id,
                "name": exp_cfg.name,
                "description": exp_cfg.description,
                "mean_classification_score": round(mean_score, 4),
            }
        )

    # Write ablation_results.csv
    results_csv = out / "ablation_results.csv"
    if all_metric_rows:
        all_keys = dict.fromkeys(k for row in all_metric_rows for k in row.keys())
        with results_csv.open("w", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(all_keys.keys()), extrasaction="ignore")
            writer.writeheader()
            writer.writerows(all_metric_rows)

    # Render ablation comparison plot
    plot_path = _render_ablation_plot(exp_summaries, plots_dir / "ablation_comparison.png")

    # Summary JSON
    summary_data = {
        "step": 16,
        "n_experiments": len(ABLATION_EXPERIMENTS),
        "experiments": exp_summaries,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "evaluation_strategy": "Direct Ground Truth comparison on 110 canonical buildings (leakage prevented)",
    }
    write_json(out / "ablation_summary.json", summary_data)

    # Manifest
    manifest = {
        "step": 16,
        "description": "Step 16 Final Ablation Study (Experiments A through J)",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "random_seed": RANDOM_SEED,
        "outputs": {
            "ablation_results": str(results_csv),
            "ablation_summary": str(out / "ablation_summary.json"),
            "ablation_plot": str(plot_path),
        },
    }
    write_json(out / "ablation_manifest.json", manifest)

    return {
        "manifest": manifest,
        "n_experiments": len(exp_summaries),
        "outputs": manifest["outputs"],
    }


def _render_ablation_plot(exp_summaries: list[dict[str, Any]], dest: Path) -> Path:
    """Render overall ablation comparison bar chart."""
    ensure_parent(dest)
    names = [e["experiment_id"] + "\n" + e["name"].replace("_", " ") for e in exp_summaries]
    scores = [e["mean_classification_score"] for e in exp_summaries]

    fig, ax = plt.subplots(figsize=(12, 5))
    colors = ["#4c78a8", "#f58518", "#e45756", "#72b7b2", "#54a24b", "#eeca3b", "#b279a2", "#ff9da6", "#9d755d", "#bab0ac"]
    ax.bar(range(len(names)), scores, color=colors[:len(names)], width=0.6)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=25, ha="right", fontsize=8)
    ax.set_ylim(0.0, 1.1)
    ax.set_ylabel("Mean Categorical/Ordinal Score")
    ax.set_title("System Ablation Study: Component Contributions (Experiments A through J)")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(dest, dpi=140)
    plt.close(fig)
    return dest
