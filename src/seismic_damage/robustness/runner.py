"""Step 15: Robustness experiment execution comparing RAG-only, VLM-only, and Multimodal.

Evaluates how performance, coverage, and abstention behave across Scenarios A through J.
Missing VLM is explicitly flagged as vlm_available=False and never treated as zero.
"""

from __future__ import annotations

import csv
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from seismic_damage.config.settings import PROJECT_ROOT
from seismic_damage.fusion.fusion import fuse_building
from seismic_damage.fusion.reliability import (
    compute_parameter_reliability,
    reliability_weights_by_param,
)
from seismic_damage.io_utils import ensure_parent, write_json
from seismic_damage.robustness.scenarios import (
    RANDOM_SEED,
    SCENARIO_DESCRIPTIONS,
    build_scenario_image_map,
)
from seismic_damage.validation.cross_validation import _index_by_building, _value
from seismic_damage.validation.metrics import (
    ALL_PARAMETERS,
    is_missing,
    metrics_for,
    parameter_kind,
)

ROBUSTNESS_DIR = PROJECT_ROOT / "data" / "processed" / "robustness"


def _condition_vlm_records(
    vlm_records: list[Mapping[str, Any]],
    scenario: str,
    scenario_image_map: dict[str, list[str]],
    *,
    seed: int = RANDOM_SEED,
) -> list[dict[str, Any]]:
    """Apply scenario conditions to VLM records.

    For degraded scenarios (E, F), blank out a fraction of attributes to simulate
    information loss from pixelation/blur.
    For missing scenarios (G, or buildings with 0 images), return vlm_available=False.
    """
    import random as _rng
    rng = _rng.Random(seed)
    by_bid = {str(r.get("building_id") or ""): r for r in vlm_records}

    conditioned: list[dict[str, Any]] = []
    for bid, imgs in sorted(scenario_image_map.items()):
        base_rec = by_bid.get(bid)

        if not imgs or scenario == "G_0pct" or base_rec is None:
            # Explicitly missing VLM evidence
            rec: dict[str, Any] = {
                "building_id": bid,
                "vlm_available": False,
                "evidence_source": "vlm",
                "n_images_available": 0,
            }
            conditioned.append(rec)
            continue

        rec = dict(base_rec)
        rec["vlm_available"] = True
        rec["n_images_available"] = len(imgs)

        if scenario == "E_pixelated":
            # Severe pixelation: simulate loss of fine features (90% missing)
            for p in ALL_PARAMETERS:
                if rng.random() < 0.90:
                    rec[p] = None
                    if isinstance(rec.get("parameters"), dict):
                        rec["parameters"][p] = None
        elif scenario == "F_low_quality":
            # Low contrast / blur: simulate moderate loss (50% missing)
            for p in ALL_PARAMETERS:
                if rng.random() < 0.50:
                    rec[p] = None
                    if isinstance(rec.get("parameters"), dict):
                        rec["parameters"][p] = None

        conditioned.append(rec)
    return conditioned


def evaluate_modality_predictions(
    predictions: list[Mapping[str, Any]],
    ground_truth: list[Mapping[str, Any]],
    modality_name: str,
    scenario_name: str,
) -> list[dict[str, Any]]:
    """Compute per-parameter metrics, coverage, and abstention for a predictions set."""
    pred_idx = _index_by_building(predictions)
    gt_idx = _index_by_building(ground_truth)
    all_bids = sorted(set(gt_idx.keys()))

    results: list[dict[str, Any]] = []
    for param in sorted(ALL_PARAMETERS):
        kind = parameter_kind(param)

        pred_vals: list[Any] = []
        gt_vals: list[Any] = []
        n_predicted = 0

        for bid in all_bids:
            p_val = _value(pred_idx.get(bid), param) if bid in pred_idx else None
            g_val = _value(gt_idx.get(bid), param)

            if not is_missing(p_val):
                n_predicted += 1
            pred_vals.append(p_val)
            gt_vals.append(g_val)

        m = metrics_for(param, pred_vals, gt_vals)
        n_eval = m.get("n", 0)
        coverage = n_predicted / len(all_bids) if all_bids else 0.0
        abstention = 1.0 - coverage

        # Headline metric extraction
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
            "scenario": scenario_name,
            "modality": modality_name,
            "parameter": param,
            "kind": kind,
            "metric_name": metric_name,
            "score": score,
            "n_matched": n_eval,
            "coverage": round(coverage, 4),
            "abstention_rate": round(abstention, 4),
            "n_total_buildings": len(all_bids),
        }
        for k, v in m.items():
            if k not in ("parameter", "kind", "n"):
                row[f"metric_{k}"] = v
        results.append(row)

    return results


def run_full_robustness_framework(
    rag_records: list[Mapping[str, Any]],
    vlm_records: list[Mapping[str, Any]],
    ground_truth_records: list[Mapping[str, Any]],
    building_image_map: dict[str, list[str]],
    *,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    """Execute Step 15 comprehensive robustness framework across all scenarios.

    Compares:
      1. RAG-only
      2. VLM-only
      3. Calibrated Multimodal (Step 14 reliability weighting)
    """
    out = output_dir or ROBUSTNESS_DIR
    out.mkdir(parents=True, exist_ok=True)
    plots_dir = out / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    # Compute baseline parameter reliability for multimodal fusion
    rel_rows = compute_parameter_reliability(rag_records, vlm_records, ground_truth_records)
    weights_by_p = reliability_weights_by_param(rel_rows)

    all_metric_rows: list[dict[str, Any]] = []

    for scenario in SCENARIO_DESCRIPTIONS:
        scenario_img_map = build_scenario_image_map(building_image_map, scenario)
        cond_vlm = _condition_vlm_records(vlm_records, scenario, scenario_img_map)

        # 1. RAG-only (unaffected by visual degradation)
        rag_metrics = evaluate_modality_predictions(
            rag_records, ground_truth_records, modality_name="RAG_only", scenario_name=scenario
        )
        all_metric_rows.extend(rag_metrics)

        # 2. VLM-only (degraded by scenario)
        vlm_metrics = evaluate_modality_predictions(
            cond_vlm, ground_truth_records, modality_name="VLM_only", scenario_name=scenario
        )
        all_metric_rows.extend(vlm_metrics)

        # 3. Calibrated Multimodal fusion
        fused_records = []
        rag_idx = _index_by_building(rag_records)
        vlm_idx = _index_by_building(cond_vlm)
        all_bids = sorted(set(building_image_map.keys()))

        for bid in all_bids:
            r_rec = rag_idx.get(bid)
            v_rec = vlm_idx.get(bid)
            # If VLM is not available, pass None to fusion
            v_input = v_rec if (v_rec and v_rec.get("vlm_available", True)) else None
            f_rec = fuse_building(
                building_id=bid,
                rag_rec=r_rec,
                vlm_rec=v_input,
                weights_by_param=weights_by_p,
                mode="F4_calibrated",
            )
            fused_records.append(f_rec)

        fused_metrics = evaluate_modality_predictions(
            fused_records, ground_truth_records, modality_name="Calibrated_Multimodal", scenario_name=scenario
        )
        all_metric_rows.extend(fused_metrics)

    # Write robustness_results.csv
    results_csv = out / "robustness_results.csv"
    if all_metric_rows:
        all_keys = dict.fromkeys(k for row in all_metric_rows for k in row.keys())
        with results_csv.open("w", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(all_keys.keys()), extrasaction="ignore")
            writer.writeheader()
            writer.writerows(all_metric_rows)

    # Render plots
    plot_paths = _render_robustness_plots(all_metric_rows, plots_dir)

    # Summary JSON
    summary_data = {
        "step": 15,
        "n_scenarios": len(SCENARIO_DESCRIPTIONS),
        "scenarios": SCENARIO_DESCRIPTIONS,
        "n_buildings": len(building_image_map),
        "modalities_compared": ["RAG_only", "VLM_only", "Calibrated_Multimodal"],
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    write_json(out / "robustness_summary.json", summary_data)

    # Manifest
    manifest = {
        "step": 15,
        "description": "Step 15 Robustness / Missing & Poor VLM Analysis",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "random_seed": RANDOM_SEED,
        "outputs": {
            "robustness_results": str(results_csv),
            "robustness_summary": str(out / "robustness_summary.json"),
            "plots": {k: str(v) for k, v in plot_paths.items()},
        },
    }
    write_json(out / "robustness_manifest.json", manifest)

    return {
        "manifest": manifest,
        "n_rows": len(all_metric_rows),
        "outputs": manifest["outputs"],
    }


def _render_robustness_plots(metric_rows: list[dict[str, Any]], dest: Path) -> dict[str, Path]:
    """Generate publication-quality figures for Step 15."""
    paths: dict[str, Path] = {}
    avail_scenarios = ["A_100pct", "B_75pct", "C_50pct", "D_25pct", "G_0pct"]

    # 1. Performance vs Image Availability (damage_grade)
    fig, ax = plt.subplots(figsize=(8, 5))
    for mod, color, style in [
        ("VLM_only", "#f58518", "--o"),
        ("Calibrated_Multimodal", "#2ca02c", "-s"),
        ("RAG_only", "#4c78a8", ":^"),
    ]:
        scores = []
        for sc in avail_scenarios:
            match = next(
                (r for r in metric_rows if r["scenario"] == sc and r["modality"] == mod and r["parameter"] == "damage_grade"),
                None,
            )
            scores.append(match["score"] if match and match["score"] is not None else float("nan"))
        ax.plot(range(len(avail_scenarios)), scores, style, label=mod.replace("_", " "), color=color, linewidth=2)

    ax.set_xticks(range(len(avail_scenarios)))
    ax.set_xticklabels(["100%", "75%", "50%", "25%", "0%"])
    ax.set_ylim(0.0, 1.05)
    ax.set_xlabel("Image Availability")
    ax.set_ylabel("Exact Match Score (damage_grade)")
    ax.set_title("Damage Grade Assessment vs Visual Evidence Availability")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    p1 = dest / "performance_vs_image_availability.png"
    fig.savefig(p1, dpi=140)
    plt.close(fig)
    paths["performance_vs_image_availability"] = p1

    # 2. Coverage vs Image Availability
    fig, ax = plt.subplots(figsize=(8, 5))
    for mod, color in [("VLM_only", "#f58518"), ("Calibrated_Multimodal", "#2ca02c")]:
        covs = []
        for sc in avail_scenarios:
            match = next(
                (r for r in metric_rows if r["scenario"] == sc and r["modality"] == mod and r["parameter"] == "damage_grade"),
                None,
            )
            covs.append(match["coverage"] if match else 0.0)
        ax.plot(range(len(avail_scenarios)), covs, marker="o", label=mod.replace("_", " "), color=color, linewidth=2)

    ax.set_xticks(range(len(avail_scenarios)))
    ax.set_xticklabels(["100%", "75%", "50%", "25%", "0%"])
    ax.set_ylim(0.0, 1.05)
    ax.set_xlabel("Image Availability")
    ax.set_ylabel("Building Parameter Coverage")
    ax.set_title("VLM Coverage Retention under Missing Evidence")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    p2 = dest / "coverage_vs_image_availability.png"
    fig.savefig(p2, dpi=140)
    plt.close(fig)
    paths["coverage_vs_image_availability"] = p2

    # 3. Performance vs Quality Degradation
    quality_scenarios = ["A_100pct", "F_low_quality", "E_pixelated", "G_0pct"]
    fig, ax = plt.subplots(figsize=(8, 5))
    for mod, color in [("VLM_only", "#f58518"), ("Calibrated_Multimodal", "#2ca02c")]:
        scores = []
        for sc in quality_scenarios:
            match = next(
                (r for r in metric_rows if r["scenario"] == sc and r["modality"] == mod and r["parameter"] == "damage_grade"),
                None,
            )
            scores.append(match["score"] if match and match["score"] is not None else 0.0)
        ax.bar([i + (0.2 if mod == "VLM_only" else -0.2) for i in range(len(quality_scenarios))], scores, width=0.35, label=mod.replace("_", " "), color=color)

    ax.set_xticks(range(len(quality_scenarios)))
    ax.set_xticklabels(["Original (100%)", "Low Quality (Blur)", "Pixelated", "No Image"], rotation=15)
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("Exact Match Score")
    ax.set_title("System Performance Under Visual Quality Degradation")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    p3 = dest / "performance_vs_image_quality.png"
    fig.savefig(p3, dpi=140)
    plt.close(fig)
    paths["performance_vs_image_quality"] = p3

    return paths
