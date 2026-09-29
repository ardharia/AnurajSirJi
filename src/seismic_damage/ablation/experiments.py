"""Step 16: Ablation experiment matrix definitions.

Experiments:
  A: RAG only (baseline RAG extraction)
  B: VLM only (baseline VLM extraction)
  C: Equal-weight RAG + VLM (50:50 fusion)
  D: Calibrated RAG + VLM (Step 14 empirical reliability weighting)
  E: Optimized RAG (Step 12 best retrieval configuration)
  F: Optimized VLM (Step 13 best visual aggregation configuration)
  G: Optimized + Calibrated Multimodal (Optimized RAG + Optimized VLM + Calibrated fusion)
  H: Missing/poor VLM ablation (Calibrated fusion under 50% degraded VLM)
  I: Incomplete text ablation (Calibrated fusion with 50% text evidence blanked)
  J: Conflicting evidence ablation (Evaluation isolating instances where RAG & VLM disagree)

Prevents data leakage:
- Same canonical buildings across all experiments.
- Same Ground Truth evaluation records.
- Explicit declaration of dataset partition/limitation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class AblationExperimentConfig:
    experiment_id: str
    name: str
    description: str
    uses_rag: bool
    uses_vlm: bool
    rag_variant: str  # "baseline", "optimized", "incomplete"
    vlm_variant: str  # "baseline", "optimized", "degraded"
    fusion_mode: str  # "none", "equal_weight", "calibrated"
    conflict_filter: bool = False  # If True, evaluate only conflicting cases


ABLATION_EXPERIMENTS: list[AblationExperimentConfig] = [
    AblationExperimentConfig(
        experiment_id="Exp_A",
        name="RAG_only",
        description="Standalone baseline text RAG pipeline without visual inputs",
        uses_rag=True,
        uses_vlm=False,
        rag_variant="baseline",
        vlm_variant="none",
        fusion_mode="none",
    ),
    AblationExperimentConfig(
        experiment_id="Exp_B",
        name="VLM_only",
        description="Standalone baseline visual VLM pipeline without text inputs",
        uses_rag=False,
        uses_vlm=True,
        rag_variant="none",
        vlm_variant="baseline",
        fusion_mode="none",
    ),
    AblationExperimentConfig(
        experiment_id="Exp_C",
        name="Equal_Weight_Multimodal",
        description="Equal 50:50 weighting of baseline RAG and baseline VLM",
        uses_rag=True,
        uses_vlm=True,
        rag_variant="baseline",
        vlm_variant="baseline",
        fusion_mode="equal_weight",
    ),
    AblationExperimentConfig(
        experiment_id="Exp_D",
        name="Calibrated_Multimodal",
        description="Empirically calibrated source reliability fusion of baseline models",
        uses_rag=True,
        uses_vlm=True,
        rag_variant="baseline",
        vlm_variant="baseline",
        fusion_mode="calibrated",
    ),
    AblationExperimentConfig(
        experiment_id="Exp_E",
        name="Optimized_RAG_only",
        description="Optimized RAG configuration (Step 12 selected parameters)",
        uses_rag=True,
        uses_vlm=False,
        rag_variant="optimized",
        vlm_variant="none",
        fusion_mode="none",
    ),
    AblationExperimentConfig(
        experiment_id="Exp_F",
        name="Optimized_VLM_only",
        description="Optimized VLM configuration (Step 13 selected aggregation)",
        uses_rag=False,
        uses_vlm=True,
        rag_variant="none",
        vlm_variant="optimized",
        fusion_mode="none",
    ),
    AblationExperimentConfig(
        experiment_id="Exp_G",
        name="Optimized_Calibrated_Multimodal",
        description="Combined optimized RAG + optimized VLM with calibrated reliability fusion",
        uses_rag=True,
        uses_vlm=True,
        rag_variant="optimized",
        vlm_variant="optimized",
        fusion_mode="calibrated",
    ),
    AblationExperimentConfig(
        experiment_id="Exp_H",
        name="Degraded_VLM_Multimodal",
        description="Calibrated multimodal performance under degraded / 50% missing VLM evidence",
        uses_rag=True,
        uses_vlm=True,
        rag_variant="baseline",
        vlm_variant="degraded",
        fusion_mode="calibrated",
    ),
    AblationExperimentConfig(
        experiment_id="Exp_I",
        name="Incomplete_Text_Multimodal",
        description="Calibrated multimodal performance under 50% incomplete text evidence",
        uses_rag=True,
        uses_vlm=True,
        rag_variant="incomplete",
        vlm_variant="baseline",
        fusion_mode="calibrated",
    ),
    AblationExperimentConfig(
        experiment_id="Exp_J",
        name="Conflicting_Evidence_Ablation",
        description="Evaluation isolating parameters where RAG and VLM provide conflicting values",
        uses_rag=True,
        uses_vlm=True,
        rag_variant="baseline",
        vlm_variant="baseline",
        fusion_mode="calibrated",
        conflict_filter=True,
    ),
]
