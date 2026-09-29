# Step 16: Final Ablation Study

## Objective
Quantitatively determine the contribution of each system component across 10 unified experiments (Exp A through Exp J).

## Experiments Matrix
- **Exp A (RAG only)**: Baseline textual RAG extraction without visual inputs.
- **Exp B (VLM only)**: Baseline visual VLM extraction without text inputs.
- **Exp C (Equal-Weight Multimodal)**: 50:50 equal weighting of RAG and VLM.
- **Exp D (Calibrated Multimodal)**: Empirical Ground Truth reliability weighting.
- **Exp E (Optimized RAG)**: Step 12 optimized RAG retrieval configuration.
- **Exp F (Optimized VLM)**: Step 13 optimized VLM aggregation configuration.
- **Exp G (Optimized + Calibrated Multimodal)**: Best RAG + Best VLM with calibrated fusion.
- **Exp H (Degraded VLM Multimodal)**: Multimodal performance with 50% visual degradation.
- **Exp I (Incomplete Text Multimodal)**: Multimodal performance with 50% text masking.
- **Exp J (Conflicting Evidence Ablation)**: Performance isolating instances where RAG and VLM disagree.

## Outputs
Saved under `data/processed/ablation/` with `ablation_results.csv`, `ablation_summary.json`, `ablation_manifest.json`, and `plots/ablation_comparison.png`.
