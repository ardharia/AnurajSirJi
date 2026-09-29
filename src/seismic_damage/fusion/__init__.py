"""Step 14: Calibrated RAG + VLM Fusion package.

This is the first and only stage where RAG and VLM are allowed to interact.
Steps 11–13 remain completely independent.
"""

from seismic_damage.fusion.evaluation import (
    build_fusion_comparison,
    evaluate_records_against_gt,
    run_fusion_pipeline,
)
from seismic_damage.fusion.fusion import (
    FUSION_MODES,
    fuse_building,
    generate_fused_dataset,
    write_predictions_csv,
)
from seismic_damage.fusion.reliability import (
    compute_parameter_reliability,
    reliability_weights_by_param,
    write_reliability_csv,
)
from seismic_damage.fusion.weighting import (
    fuse_binary,
    fuse_categorical,
    fuse_numerical,
    fuse_ordinal,
    fuse_parameter,
)

__all__ = [
    "FUSION_MODES",
    "build_fusion_comparison",
    "compute_parameter_reliability",
    "evaluate_records_against_gt",
    "fuse_binary",
    "fuse_building",
    "fuse_categorical",
    "fuse_numerical",
    "fuse_ordinal",
    "fuse_parameter",
    "generate_fused_dataset",
    "reliability_weights_by_param",
    "run_fusion_pipeline",
    "write_predictions_csv",
    "write_reliability_csv",
]
