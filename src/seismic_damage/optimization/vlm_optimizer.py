"""Step 13: Independent VLM optimization.

Explores image selection, prompt, and aggregation configurations independently of RAG.
Ground truth is the only evaluation target — never VLM vs RAG agreement.
VLM output is never fed into RAG and RAG output is never fed into VLM.

Because no real VLM API key is typically available in CI/offline environments,
this optimizer operates against the existing pre-computed VLM records. The
experiment matrix varies *which* records to use (image subset, confidence filters)
and *how* parameters are aggregated from image-level to building-level.

The limitation is recorded clearly in the manifest.
"""

from __future__ import annotations

import csv
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from seismic_damage.io_utils import ensure_parent, write_json
from seismic_damage.optimization.config import (
    RANDOM_SEED,
    ExperimentResult,
    evaluate_against_ground_truth,
    experiment_results_to_rows,
    select_best_experiment,
)
from seismic_damage.validation.cross_validation import _index_by_building
from seismic_damage.validation.metrics import ALL_PARAMETERS, is_missing
from seismic_damage.config.settings import PROJECT_ROOT

VLM_OPT_DIR = PROJECT_ROOT / "data" / "processed" / "vlm_optimization"


# ---------------------------------------------------------------------------
# Experiment matrix
# ---------------------------------------------------------------------------

VLM_EXPERIMENT_MATRIX: list[dict[str, Any]] = [
    # V1: baseline — all images, no confidence filter, no aggregation change
    {
        "experiment_id": "V1_baseline",
        "confidence_threshold": 0.0,
        "image_selection": "all",
        "aggregation": "agreement_only",
        "notes": "Baseline: all images, agreement-only aggregation (Step 6 behaviour)",
    },
    # V2: confidence filter — only include observations above threshold
    {
        "experiment_id": "V2_confidence_filter_05",
        "confidence_threshold": 0.5,
        "image_selection": "all",
        "aggregation": "agreement_only",
        "notes": "Filter: only retain high-confidence observations (>=0.5)",
    },
    # V3: image selection — first image only (representative single-image)
    {
        "experiment_id": "V3_first_image_only",
        "confidence_threshold": 0.0,
        "image_selection": "first_only",
        "aggregation": "agreement_only",
        "notes": "Single-image: only use the first (index-0) image per building",
    },
    # V4: majority vote aggregation instead of agreement-only
    {
        "experiment_id": "V4_majority_vote",
        "confidence_threshold": 0.0,
        "image_selection": "all",
        "aggregation": "majority_vote",
        "notes": "Majority vote aggregation (relaxes conflict → missing rule)",
    },
    # V5: combined — confidence filter + first image
    {
        "experiment_id": "V5_combined",
        "confidence_threshold": 0.5,
        "image_selection": "first_only",
        "aggregation": "agreement_only",
        "notes": "Combined: first image + confidence filter",
    },
]


def _apply_confidence_filter(
    vlm_records: list[Mapping[str, Any]],
    threshold: float,
) -> list[dict[str, Any]]:
    """Zero out parameters in records below confidence threshold.

    Missing → None; never zero.
    """
    result: list[dict[str, Any]] = []
    for rec in vlm_records:
        conf = float(rec.get("confidence") or 0.0)
        if conf < threshold:
            # Confidence is too low — treat all parameters as missing
            filtered: dict[str, Any] = {
                "building_id": rec.get("building_id"),
                "evidence_source": "vlm",
                "confidence": conf,
                "vlm_low_confidence": True,
            }
        else:
            filtered = dict(rec)
        result.append(filtered)
    return result


def _apply_image_selection(
    vlm_records: list[Mapping[str, Any]],
    selection: str,
    image_level_records: list[Mapping[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Apply image selection strategy.

    For 'first_only', restrict to buildings' first image.
    For 'all', no change.
    """
    if selection == "all" or not image_level_records:
        return [dict(r) for r in vlm_records]

    if selection == "first_only":
        # Group image-level records by building, keep only first
        from collections import defaultdict as _dd
        by_bid: dict[str, list[Mapping[str, Any]]] = _dd(list)
        for rec in image_level_records:
            bid = str(rec.get("building_id") or "")
            if bid:
                by_bid[bid].append(rec)

        result: list[dict[str, Any]] = []
        for rec in vlm_records:
            bid = str(rec.get("building_id") or "")
            images = by_bid.get(bid, [])
            if not images:
                result.append(dict(rec))
                continue
            # Use only the first image-level observation
            first = dict(images[0])
            first["building_id"] = bid
            result.append(first)
        return result

    return [dict(r) for r in vlm_records]


def _apply_majority_vote(
    vlm_records: list[Mapping[str, Any]],
    image_level_records: list[Mapping[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Aggregate image-level observations using majority vote instead of agreement-only.

    Agreement-only (Step 6 default): parameters with any conflict become missing.
    Majority vote: the most common non-missing value wins.
    """
    if not image_level_records:
        return [dict(r) for r in vlm_records]

    from collections import Counter, defaultdict as _dd

    by_bid: dict[str, list[Mapping[str, Any]]] = _dd(list)
    for rec in image_level_records:
        bid = str(rec.get("building_id") or "")
        if bid:
            by_bid[bid].append(rec)

    result: list[dict[str, Any]] = []
    for rec in vlm_records:
        bid = str(rec.get("building_id") or "")
        images = by_bid.get(bid, [])
        if not images:
            result.append(dict(rec))
            continue

        merged: dict[str, Any] = {"building_id": bid, "evidence_source": "vlm"}
        for param in ALL_PARAMETERS:
            values = [
                img.get(param) for img in images
                if not is_missing(img.get(param))
            ]
            if not values:
                merged[param] = None  # Still None, not zero
            else:
                # Majority vote: most common value
                counter: Counter[str] = Counter(str(v).strip().lower() for v in values)
                winner_str = counter.most_common(1)[0][0]
                # Return original typed value matching the winner
                for v in values:
                    if str(v).strip().lower() == winner_str:
                        merged[param] = v
                        break
        result.append(merged)
    return result


def run_vlm_optimization(
    vlm_records: list[Mapping[str, Any]],
    ground_truth_records: list[Mapping[str, Any]],
    *,
    image_level_records: list[Mapping[str, Any]] | None = None,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    """Run all VLM optimization experiments and select the best configuration.

    Args:
        vlm_records: Pre-computed building-level VLM records (baseline).
        ground_truth_records: Ground truth building records (evaluation target).
        image_level_records: Optional image-level VLM observations (for aggregation experiments).
        output_dir: Override for output directory.

    Returns:
        Summary with experiment results and best configuration.
    """
    out = output_dir or VLM_OPT_DIR
    out.mkdir(parents=True, exist_ok=True)
    baseline_dir = out / "baseline"
    baseline_dir.mkdir(parents=True, exist_ok=True)

    results: list[ExperimentResult] = []

    for config in VLM_EXPERIMENT_MATRIX:
        exp_id = config["experiment_id"]
        confidence_threshold = float(config.get("confidence_threshold", 0.0))
        image_selection = str(config.get("image_selection", "all"))
        aggregation = str(config.get("aggregation", "agreement_only"))

        # Apply experiment transformations (VLM-only, no RAG dependency)
        working_records: list[Mapping[str, Any]] = list(vlm_records)

        if aggregation == "majority_vote" and image_level_records:
            working_records = _apply_majority_vote(working_records, image_level_records)
        elif image_selection != "all":
            working_records = _apply_image_selection(
                working_records, image_selection, image_level_records
            )

        if confidence_threshold > 0.0:
            working_records = _apply_confidence_filter(working_records, confidence_threshold)

        param_metrics, n = evaluate_against_ground_truth(
            working_records, list(ground_truth_records)
        )

        result = ExperimentResult(
            experiment_id=exp_id,
            modality="vlm",
            configuration={
                k: v for k, v in config.items() if k not in ("experiment_id", "notes")
            },
            parameter_metrics=param_metrics,
            n_buildings=n,
            notes=config.get("notes", ""),
        )
        results.append(result)

        if exp_id == "V1_baseline":
            _write_csv(
                experiment_results_to_rows([result]),
                baseline_dir / "baseline_results.csv",
            )

    best = select_best_experiment(results)

    all_rows = experiment_results_to_rows(results)
    _write_csv(all_rows, out / "experiment_results.csv")

    write_json(
        out / "best_configuration.json",
        {
            "experiment_id": best.experiment_id,
            "configuration": best.configuration,
            "aggregate_score": best.aggregate_score(),
            "notes": best.notes,
        },
    )

    manifest = {
        "step": 13,
        "description": "Independent VLM Optimization",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "random_seed": RANDOM_SEED,
        "n_experiments": len(results),
        "experiments": [r.experiment_id for r in results],
        "best_experiment": best.experiment_id,
        "best_aggregate_score": round(best.aggregate_score(), 6),
        "selection_metric": "aggregate_score (mean primary metric across parameters)",
        "evaluation_target": "ground_truth (never VLM vs RAG)",
        "rag_dependency": False,
        "limitation": (
            "VLM optimization uses pre-computed building-level records rather than "
            "re-running the full VLM pipeline (no API key required). Experiments vary "
            "confidence thresholds, image selection, and aggregation strategy. "
            "The Bhuj dataset (110 buildings) is too small for train/val/test split."
        ),
        "outputs": {
            "baseline_results": str(baseline_dir / "baseline_results.csv"),
            "experiment_results": str(out / "experiment_results.csv"),
            "best_configuration": str(out / "best_configuration.json"),
        },
    }
    write_json(out / "optimization_manifest.json", manifest)

    return {
        "manifest": manifest,
        "n_experiments": len(results),
        "best_experiment_id": best.experiment_id,
        "best_aggregate_score": best.aggregate_score(),
        "results": [r.to_dict() for r in results],
    }


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
