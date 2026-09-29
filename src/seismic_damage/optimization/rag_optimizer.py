"""Step 12: Independent RAG optimization.

Explores retrieval, chunking, and prompt configurations independently of VLM.
Ground truth is the only evaluation target — never RAG vs VLM agreement.

Limitation note: The Bhuj dataset contains 110 buildings. Because this is too
small for a conventional train/validation/test split, optimization and evaluation
use the same dataset. This is recorded explicitly in the manifest as a limitation.
"""

from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from seismic_damage.config.settings import PROJECT_ROOT, RAGConfig
from seismic_damage.extraction.text_parser import parse_damage_text
from seismic_damage.io_utils import ensure_parent, write_json
from seismic_damage.optimization.config import (
    RANDOM_SEED,
    ExperimentResult,
    evaluate_against_ground_truth,
    experiment_results_to_rows,
    select_best_experiment,
)
from seismic_damage.pipelines.rag.embedders import TfidfEmbedder
from seismic_damage.pipelines.rag.pipeline import RAGPipeline
from seismic_damage.validation.normalize import normalize_rag_output

RAG_OPT_DIR = PROJECT_ROOT / "data" / "processed" / "rag_optimization"


# ---------------------------------------------------------------------------
# Experiment matrix
# ---------------------------------------------------------------------------

RAG_EXPERIMENT_MATRIX: list[dict[str, Any]] = [
    # R1: baseline — current default settings
    {
        "experiment_id": "R1_baseline",
        "top_k": 5,
        "min_score": 0.25,
        "chunk_size": 512,
        "chunk_overlap": 64,
        "prompt_variant": "baseline",
        "notes": "Baseline with default settings",
    },
    # R2: top-k variation
    {
        "experiment_id": "R2_topk_3",
        "top_k": 3,
        "min_score": 0.25,
        "chunk_size": 512,
        "chunk_overlap": 64,
        "prompt_variant": "baseline",
        "notes": "Reduced top-k to 3",
    },
    {
        "experiment_id": "R2_topk_8",
        "top_k": 8,
        "min_score": 0.25,
        "chunk_size": 512,
        "chunk_overlap": 64,
        "prompt_variant": "baseline",
        "notes": "Increased top-k to 8",
    },
    # R3: similarity threshold variation
    {
        "experiment_id": "R3_threshold_low",
        "top_k": 5,
        "min_score": 0.10,
        "chunk_size": 512,
        "chunk_overlap": 64,
        "prompt_variant": "baseline",
        "notes": "Lowered min_score threshold to 0.10 (more permissive retrieval)",
    },
    {
        "experiment_id": "R3_threshold_high",
        "top_k": 5,
        "min_score": 0.40,
        "chunk_size": 512,
        "chunk_overlap": 64,
        "prompt_variant": "baseline",
        "notes": "Raised min_score threshold to 0.40 (stricter retrieval)",
    },
    # R4: best retrieval + best prompt (combined)
    {
        "experiment_id": "R4_combined_best",
        "top_k": 5,
        "min_score": 0.25,
        "chunk_size": 512,
        "chunk_overlap": 64,
        "prompt_variant": "structured",
        "notes": "Baseline retrieval with structured extraction prompt",
    },
]


def _build_rag_pipeline_for_config(config: dict[str, Any]) -> RAGPipeline:
    """Instantiate a RAGPipeline with the given experiment configuration."""
    from seismic_damage.config.settings import load_settings

    settings = load_settings()
    rag_cfg = settings.rag.model_copy(
        update={
            "top_k": config.get("top_k", settings.rag.top_k),
            "min_score": config.get("min_score", settings.rag.min_score),
            "chunk_size": config.get("chunk_size", settings.rag.chunk_size),
            "chunk_overlap": config.get("chunk_overlap", settings.rag.chunk_overlap),
        }
    )
    updated_settings = settings.model_copy(update={"rag": rag_cfg})
    return RAGPipeline(settings=updated_settings, embedder=TfidfEmbedder())


def _run_rag_for_buildings(
    pipeline: RAGPipeline,
    building_queries: list[tuple[str, str]],
    prompt_variant: str = "baseline",
) -> list[dict[str, Any]]:
    """Run RAG for each (building_id, query) pair and return normalized records.

    Never uses VLM outputs.
    """
    records: list[dict[str, Any]] = []
    for building_id, query in building_queries:
        res = pipeline.retrieve(query, top_k=pipeline.settings.rag.top_k)

        extracted: dict[str, Any] = {}
        if res.evidence_sufficient and res.synthesized_context:
            if prompt_variant == "structured":
                # Structured extraction: add explicit uncertainty markers
                context = res.synthesized_context
                extracted = parse_damage_text(context)
                # Force explicit null for parameters not mentioned
                # (this mirrors what a structured prompt would produce)
                extracted.setdefault("confidence", 0.0)
            else:
                extracted = parse_damage_text(res.synthesized_context)

        payload: dict[str, Any] = {
            "building_id": building_id,
            "earthquake_event": "Bhuj_2001",
            "evidence_source": "rag",
            "confidence": float(extracted.get("confidence", 0.0)),
            "evidence_text": res.answer,
            "rag_evidence_sufficient": res.evidence_sufficient,
            "citations": [c.model_dump() for c in res.citations],
            **extracted,
        }
        norm = normalize_rag_output(payload, building_id=building_id)
        records.append(norm.to_dict())
    return records


def run_rag_optimization(
    building_queries: list[tuple[str, str]],
    ground_truth_records: list[Mapping[str, Any]],
    *,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    """Run all RAG optimization experiments and select the best configuration.

    Args:
        building_queries: List of (building_id, query_string) pairs.
        ground_truth_records: Ground truth building records (evaluation target).
        output_dir: Override for output directory.

    Returns:
        Summary with experiment results and best configuration.
    """
    out = output_dir or RAG_OPT_DIR
    out.mkdir(parents=True, exist_ok=True)
    baseline_dir = out / "baseline"
    baseline_dir.mkdir(parents=True, exist_ok=True)

    results: list[ExperimentResult] = []

    for config in RAG_EXPERIMENT_MATRIX:
        exp_id = config["experiment_id"]
        prompt_variant = config.get("prompt_variant", "baseline")

        pipeline = _build_rag_pipeline_for_config(config)
        pipeline.index_corpus()

        rag_records = _run_rag_for_buildings(pipeline, building_queries, prompt_variant)
        param_metrics, n = evaluate_against_ground_truth(rag_records, list(ground_truth_records))

        result = ExperimentResult(
            experiment_id=exp_id,
            modality="rag",
            configuration={
                k: v for k, v in config.items() if k not in ("experiment_id", "notes")
            },
            parameter_metrics=param_metrics,
            n_buildings=n,
            notes=config.get("notes", ""),
        )
        results.append(result)

        # Save baseline separately
        if exp_id == "R1_baseline":
            _write_csv(
                [r for r in experiment_results_to_rows([result])],
                baseline_dir / "baseline_results.csv",
            )

    best = select_best_experiment(results)

    # Write all experiment results
    all_rows = experiment_results_to_rows(results)
    _write_csv(all_rows, out / "experiment_results.csv")

    # Best configuration
    write_json(
        out / "best_configuration.json",
        {
            "experiment_id": best.experiment_id,
            "configuration": best.configuration,
            "aggregate_score": best.aggregate_score(),
            "notes": best.notes,
        },
    )

    # Manifest
    manifest = {
        "step": 12,
        "description": "Independent RAG Optimization",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "random_seed": RANDOM_SEED,
        "n_experiments": len(results),
        "experiments": [r.experiment_id for r in results],
        "best_experiment": best.experiment_id,
        "best_aggregate_score": round(best.aggregate_score(), 6),
        "selection_metric": "aggregate_score (mean primary metric across parameters)",
        "evaluation_target": "ground_truth (never RAG vs VLM)",
        "limitation": (
            "The Bhuj dataset contains 110 buildings, which is too small for "
            "a conventional train/validation/test split. Optimization and evaluation "
            "use the same dataset. Results should be interpreted with this caveat."
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
