"""Step 11: VLM Robustness Experiments.

Evaluates system behaviour when visual evidence is incomplete, degraded, or unavailable.
RAG and VLM remain fully independent — no fusion occurs here.
"""

from seismic_damage.robustness.degradation import (
    degrade_image_blur_low_contrast,
    degrade_image_pixelate,
    describe_degradation_condition,
)
from seismic_damage.robustness.evaluation import (
    aggregate_condition_metrics,
    build_robustness_summary,
    evaluate_condition,
)
from seismic_damage.robustness.pipeline import (
    ROBUSTNESS_DIR,
    run_availability_experiments,
    run_degradation_experiments,
    run_robustness_pipeline,
)
from seismic_damage.robustness.sampling import (
    AVAILABILITY_LEVELS,
    RANDOM_SEED,
    build_availability_sets,
    describe_availability_set,
    deterministic_sample,
)

__all__ = [
    "AVAILABILITY_LEVELS",
    "RANDOM_SEED",
    "ROBUSTNESS_DIR",
    "aggregate_condition_metrics",
    "build_availability_sets",
    "build_robustness_summary",
    "degrade_image_blur_low_contrast",
    "degrade_image_pixelate",
    "describe_availability_set",
    "describe_degradation_condition",
    "deterministic_sample",
    "evaluate_condition",
    "run_availability_experiments",
    "run_degradation_experiments",
    "run_robustness_pipeline",
]
