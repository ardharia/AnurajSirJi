"""Evaluation utilities for Steps 12 & 13 optimization.

Evaluates candidate configurations strictly against Ground Truth.
RAG and VLM are evaluated in total isolation — never against each other.
"""

from __future__ import annotations

from seismic_damage.optimization.config import (
    PRIMARY_METRIC_KEY,
    ExperimentResult,
    evaluate_against_ground_truth,
    experiment_results_to_rows,
    select_best_experiment,
)

__all__ = [
    "PRIMARY_METRIC_KEY",
    "ExperimentResult",
    "evaluate_against_ground_truth",
    "experiment_results_to_rows",
    "select_best_experiment",
]
