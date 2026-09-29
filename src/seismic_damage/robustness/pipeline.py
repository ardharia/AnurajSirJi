"""Step 11 robustness pipeline.

Runs all availability and degradation conditions, evaluates each against
ground truth using the Step 7 metrics framework, and writes outputs to
data/processed/robustness/.

RAG pipeline is never invoked here. VLM and RAG remain independent.
Missing VLM evidence is represented as None, never as zero.
"""

from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from seismic_damage.config.settings import PROJECT_ROOT
from seismic_damage.io_utils import ensure_parent, write_json
from seismic_damage.robustness.degradation import describe_degradation_condition
from seismic_damage.robustness.evaluation import (
    aggregate_condition_metrics,
    build_robustness_summary,
    evaluate_condition,
)
from seismic_damage.robustness.sampling import (
    AVAILABILITY_LEVELS,
    RANDOM_SEED,
    build_availability_sets,
    describe_availability_set,
)
from seismic_damage.validation.metrics import ALL_PARAMETERS, parameter_kind

ROBUSTNESS_DIR = PROJECT_ROOT / "data" / "processed" / "robustness"


def _stub_vlm_for_condition(
    full_vlm_records: list[Mapping[str, Any]],
    available_bids: dict[str, list[str]],
) -> list[dict[str, Any]]:
    """Return VLM records with parameters set to None for buildings with no images.

    For buildings with ≥1 available image, keep original VLM record.
    For buildings with 0 available images, return an explicit missing record.
    Missing = None, never zero.
    """
    by_bid: dict[str, Mapping[str, Any]] = {
        str(r.get("building_id", "")): r for r in full_vlm_records
    }
    result: list[dict[str, Any]] = []
    for bid, imgs in sorted(available_bids.items()):
        if imgs:
            # At least one image available — use existing VLM record
            rec = dict(by_bid.get(bid, {"building_id": bid}))
            rec["building_id"] = bid
        else:
            # No usable images → all parameters explicitly missing (None)
            rec = {
                "building_id": bid,
                "vlm_missing": True,
                # All canonical parameters are absent (not set to 0)
            }
        result.append(rec)
    return result


def run_availability_experiments(
    vlm_records: list[Mapping[str, Any]],
    ground_truth_records: list[Mapping[str, Any]],
    building_image_map: dict[str, list[str]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Run A–D and G availability conditions.

    Returns:
        (per_building_rows, condition_metric_rows)
    """
    all_rows: list[dict[str, Any]] = []
    all_metrics: list[dict[str, Any]] = []

    for condition_name, fraction in AVAILABILITY_LEVELS.items():
        available_bids = build_availability_sets(
            building_image_map, fraction, seed=RANDOM_SEED
        )
        avail_meta = describe_availability_set(available_bids, building_image_map)
        conditioned_vlm = _stub_vlm_for_condition(vlm_records, available_bids)

        rows = evaluate_condition(
            condition_name,
            conditioned_vlm,
            ground_truth_records,
            availability_meta=avail_meta,
        )
        all_rows.extend(rows)
        metrics = aggregate_condition_metrics(rows, condition_name)
        all_metrics.extend(metrics)

    return all_rows, all_metrics


def run_degradation_experiments(
    vlm_records: list[Mapping[str, Any]],
    ground_truth_records: list[Mapping[str, Any]],
    building_image_map: dict[str, list[str]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Run E (pixelated) and F (low quality) degradation conditions.

    Because we can't actually re-run a real VLM in a test environment,
    this experiment uses a proxy approach:
    - Condition E: assume degraded images produce no usable VLM output
      (represents worst-case degradation → all predictions missing).
    - Condition F: assume 50% of parameters become missing due to quality loss.

    In a real deployment these would feed degraded images through the VLM backend.
    The proxy is clearly labelled in the manifest.
    """
    all_rows: list[dict[str, Any]] = []
    all_metrics: list[dict[str, Any]] = []

    degradation_conditions = {
        "E_pixelated": {"missing_rate": 1.0, "proxy": True},
        "F_low_quality": {"missing_rate": 0.5, "proxy": True},
    }

    for condition_name, params in degradation_conditions.items():
        missing_rate = params["missing_rate"]
        conditioned_vlm = _apply_degradation_proxy(
            vlm_records, building_image_map, missing_rate=missing_rate
        )
        avail_meta = [
            {
                "building_id": bid,
                "image_count_total": len(imgs),
                "image_count_available": len(imgs),
                "availability_fraction": 1.0 if imgs else 0.0,
            }
            for bid, imgs in sorted(building_image_map.items())
        ]
        rows = evaluate_condition(
            condition_name,
            conditioned_vlm,
            ground_truth_records,
            availability_meta=avail_meta,
        )
        all_rows.extend(rows)
        metrics = aggregate_condition_metrics(rows, condition_name)
        all_metrics.extend(metrics)

    return all_rows, all_metrics


def _apply_degradation_proxy(
    vlm_records: list[Mapping[str, Any]],
    building_image_map: dict[str, list[str]],
    *,
    missing_rate: float,
) -> list[dict[str, Any]]:
    """Proxy: blank out `missing_rate` fraction of VLM parameters per building.

    missing_rate=1.0 → all parameters missing (condition E: severe pixelation).
    missing_rate=0.5 → half of parameters missing (condition F: low quality).
    """
    import random as _random

    rng = _random.Random(RANDOM_SEED)
    all_params = sorted(ALL_PARAMETERS)

    by_bid: dict[str, Mapping[str, Any]] = {
        str(r.get("building_id", "")): r for r in vlm_records
    }
    result: list[dict[str, Any]] = []
    for bid in sorted(building_image_map):
        orig = dict(by_bid.get(bid, {"building_id": bid}))
        orig["building_id"] = bid

        if missing_rate >= 1.0:
            # Blank everything
            for p in all_params:
                orig.pop(p, None)
                nested = orig.get("parameters")
                if isinstance(nested, dict):
                    nested.pop(p, None)
        else:
            n_blank = round(len(all_params) * missing_rate)
            to_blank = rng.sample(all_params, n_blank)
            for p in to_blank:
                orig.pop(p, None)
                nested = orig.get("parameters")
                if isinstance(nested, dict):
                    nested.pop(p, None)

        result.append(orig)
    return result


def _write_csv(rows: list[dict[str, Any]], path: Path) -> Path:
    ensure_parent(path)
    if not rows:
        path.write_text("", encoding="utf-8")
        return path
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return path


def render_availability_plot(
    summary_rows: list[dict[str, Any]],
    output_path: Path,
) -> Path:
    """Plot primary metric vs availability for ordinal damage_grade."""
    ensure_parent(output_path)

    # Filter to damage_grade ordinal — most illustrative parameter
    param = "damage_grade"
    cond_order = ["A_100pct", "B_75pct", "C_50pct", "D_25pct", "G_0pct"]
    values: list[float | None] = []
    for c in cond_order:
        found = next(
            (r for r in summary_rows if r["condition"] == c and r["parameter"] == param),
            None,
        )
        values.append(found.get("primary_metric_value") if found else None)

    x = list(range(len(cond_order)))
    y = [v if v is not None else float("nan") for v in values]
    labels = [c.replace("pct", "%").replace("_", " ") for c in cond_order]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(x, y, marker="o", linewidth=2, color="#1f77b4")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("Image Availability Condition")
    ax.set_ylabel("Exact Match (damage_grade)")
    ax.set_title("VLM Performance vs Image Availability\n(damage_grade exact match)")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=140)
    plt.close(fig)
    return output_path


def render_degradation_plot(
    summary_rows: list[dict[str, Any]],
    output_path: Path,
) -> Path:
    """Bar chart: primary metric under each degradation condition for key parameters."""
    ensure_parent(output_path)

    params_of_interest = ["damage_grade", "collapse_mode", "crack_severity"]
    all_conditions = sorted({r["condition"] for r in summary_rows})

    fig, axes = plt.subplots(1, len(params_of_interest), figsize=(14, 5), sharey=False)
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2"]

    for ax, param in zip(axes, params_of_interest):
        vals = []
        labels = []
        for c in all_conditions:
            found = next(
                (r for r in summary_rows if r["condition"] == c and r["parameter"] == param),
                None,
            )
            v = found.get("primary_metric_value") if found else None
            vals.append(v if v is not None else 0.0)
            labels.append(c.replace("pct", "%").replace("_", " "))

        ax.bar(range(len(labels)), vals, color=colors[: len(labels)])
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
        ax.set_ylim(0.0, 1.0)
        ax.set_title(param.replace("_", " "), fontsize=10)
        kind = parameter_kind(param)
        from seismic_damage.robustness.evaluation import _PRIMARY_METRIC
        ax.set_ylabel(_PRIMARY_METRIC.get(kind, "accuracy"), fontsize=8)

    fig.suptitle("VLM Performance Under Degradation Conditions", fontsize=12)
    fig.tight_layout()
    fig.savefig(output_path, dpi=140)
    plt.close(fig)
    return output_path


def run_robustness_pipeline(
    vlm_records: list[Mapping[str, Any]],
    ground_truth_records: list[Mapping[str, Any]],
    building_image_map: dict[str, list[str]],
    *,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    """Run the complete Step 11 robustness evaluation.

    Args:
        vlm_records: Building-level VLM records (100% availability baseline).
        ground_truth_records: Ground truth building records.
        building_image_map: {building_id -> [image_ids]} for all buildings.
        output_dir: Override for the robustness output directory.

    Returns:
        Summary dict with paths to all generated files.
    """
    out = output_dir or ROBUSTNESS_DIR
    out.mkdir(parents=True, exist_ok=True)

    # Availability experiments (conditions A–G)
    avail_rows, avail_metrics = run_availability_experiments(
        vlm_records, ground_truth_records, building_image_map
    )

    # Degradation experiments (conditions E, F)
    degrad_rows, degrad_metrics = run_degradation_experiments(
        vlm_records, ground_truth_records, building_image_map
    )

    # Combined robustness summary
    all_metrics = avail_metrics + degrad_metrics
    summary = build_robustness_summary(all_metrics, baseline_condition="A_100pct")

    # Write CSVs
    avail_csv = _write_csv(avail_rows, out / "vlm_availability_results.csv")
    degrad_csv = _write_csv(degrad_rows, out / "vlm_degradation_results.csv")
    summary_csv = _write_csv(summary, out / "vlm_robustness_summary.csv")

    # Plots
    avail_plot = render_availability_plot(
        avail_metrics + [r for r in summary if r["condition"] == "A_100pct"],
        out / "vlm_performance_vs_image_availability.png",
    )
    degrad_plot = render_degradation_plot(
        all_metrics,
        out / "vlm_performance_degradation.png",
    )

    # Manifest
    manifest = {
        "step": 11,
        "description": "VLM Missing/Poor-Image Robustness Experiments",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "random_seed": RANDOM_SEED,
        "availability_conditions": list(AVAILABILITY_LEVELS.keys()),
        "degradation_conditions": ["E_pixelated", "F_low_quality"],
        "degradation_note": (
            "Conditions E and F use a proxy (parameter blanking) to simulate "
            "VLM accuracy degradation. In a real deployment, degraded images "
            "would be fed through the VLM backend directly."
        ),
        "missing_representation": "None (never zero)",
        "n_buildings": len(building_image_map),
        "outputs": {
            "vlm_availability_results": str(avail_csv),
            "vlm_degradation_results": str(degrad_csv),
            "vlm_robustness_summary": str(summary_csv),
            "vlm_performance_vs_image_availability": str(avail_plot),
            "vlm_performance_degradation": str(degrad_plot),
        },
    }
    write_json(out / "robustness_manifest.json", manifest)

    return {
        "manifest": manifest,
        "n_availability_rows": len(avail_rows),
        "n_degradation_rows": len(degrad_rows),
        "n_summary_rows": len(summary),
        "outputs": manifest["outputs"],
    }
