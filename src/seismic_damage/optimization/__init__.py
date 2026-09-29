"""Steps 12 & 13: Independent RAG and VLM optimization package.

RAG is optimized without any VLM outputs.
VLM is optimized without any RAG outputs.
Both are evaluated against ground truth only.
"""

from seismic_damage.optimization.config import (
    ExperimentResult,
    evaluate_against_ground_truth,
    experiment_results_to_rows,
    select_best_experiment,
)
from seismic_damage.optimization.rag_optimizer import (
    RAG_EXPERIMENT_MATRIX,
    run_rag_optimization,
)
from seismic_damage.optimization.vlm_optimizer import (
    VLM_EXPERIMENT_MATRIX,
    run_vlm_optimization,
)

__all__ = [
    "ExperimentResult",
    "RAG_EXPERIMENT_MATRIX",
    "VLM_EXPERIMENT_MATRIX",
    "evaluate_against_ground_truth",
    "experiment_results_to_rows",
    "run_rag_optimization",
    "run_vlm_optimization",
    "select_best_experiment",
]
