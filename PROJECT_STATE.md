# PROJECT_STATE.md — Multimodal Seismic Damage & Risk Assessment

> Last updated: 2026-09-29

## Current System Summary

Comprehensive Python research framework for post-earthquake damage assessment, pre-earthquake vulnerability estimation, and seismic risk evaluation for the **Bhuj 2001 Earthquake**. Combines **RAG** (retrieval-augmented generation), **VLM** (vision-language model) pipelines, and calibrated multimodal fusion, with modular extensions for pre-earthquake vulnerability, risk, and ancient heritage monuments.

---

## Implemented Modules & Architecture

| Module | Status | Description |
|---|---|---|
| `config/` | ✅ Working | Pydantic Settings with `.env` and YAML configurations |
| `schemas/` | ✅ Working | Canonical schemas across ingestion, normalization, extraction, fragility, vulnerability, risk, and heritage |
| `pipelines/rag/` | ✅ Working | FAISS vector store, TF-IDF / transformer embedders, citation extraction, evidence sufficiency gating |
| `pipelines/vlm/` | ✅ Working | Structured VLM observation backends (Mock, OpenAICompatible, Fallback) |
| `linking/` | ✅ Working | 110 canonical buildings, 119 image-to-building evidence links with deduplication |
| `validation/` | ✅ Working | Controlled vocabulary mapping, normalization, cross-validation metrics (exact match, QWK, Macro-F1, F1, MAE) |
| `stats/` | ✅ Working | Parameter frequencies, rankings, distributions, and Random Forest feature importance |
| `robustness/` (Steps 11, 15) | ✅ Working | Missing & poor visual data robustness analysis across 10 scenarios (A–J: 100%, 75%, 50%, 25%, 0%, pixelation, blur, random removal, primary drop, single image) |
| `optimization/` (Steps 12, 13) | ✅ Working | Independent RAG and VLM optimization matrices evaluated strictly against Ground Truth |
| `fusion/` (Step 14) | ✅ Working | Empirical source reliability weighting ($F_1$–$F_4$ baselines), type-aware value fusion, missing-source semantics |
| `ablation/` (Step 16) | ✅ Working | Formal 10-experiment ablation study (Exp A through Exp J) without data leakage |
| `vulnerability/` (Step 17) | ✅ Working | Pre-earthquake vulnerability model (NDMA RVS scores $S_0, \Delta S, S$, EMS-98 Vulnerability Classes A–F, vulnerability index $V_I \in [0, 1]$) strictly isolated from post-earthquake damage |
| `risk/` (Step 18) | ✅ Working | Integrated Seismic Risk Engine: $\text{Risk} = \text{Hazard Demand} \times \text{Site Amplification} \times \text{Vulnerability} \times \text{Exposure}$ |
| `heritage/` (Step 19) | ✅ Working | Modular ancient monument and vernacular heritage assessment (sandstone temples, forts, traditional Bhungas, historic decay modifiers) |

---

## Test Suite Status

- **67 unit tests** — all passing (`pytest tests/`)
- Test coverage across:
  - Ground truth generation and linking
  - Data normalization and metrics
  - Pipeline graphs and settings
  - Step 11 & 15: Robustness and missing visual evidence handling
  - Step 12: Independent RAG optimization
  - Step 13: Independent VLM optimization
  - Step 14: Calibrated multimodal fusion
  - Step 16: Final ablation study
  - Step 17: Pre-earthquake vulnerability isolation
  - Step 18: Integrated seismic risk framework
  - Step 19: Heritage asset assessment

---

## Key Output Directories

- `data/processed/reference/`: `building_ground_truth.csv`, `building_ground_truth.jsonl`
- `data/processed/results/`: Consolidated RAG/VLM summaries and cross-validation reports
- `data/processed/robustness/`: Robustness results across scenarios A–J, manifests, and plots
- `data/processed/rag_optimization/`: RAG experiment matrix results and best configuration
- `data/processed/vlm_optimization/`: VLM experiment matrix results and best configuration
- `data/processed/fusion/`: `parameter_reliability.csv`, `fusion_predictions.csv`, `fusion_comparison.csv`
- `data/processed/ablation/`: Ablation experiment configs, results CSV, summary JSON, and comparison plots
- `data/processed/risk/`: `risk_assessment.csv`, `hazard_summary.csv`, `vulnerability_summary.csv`, `risk_manifest.json`
- `reference/`: Standard source registries, damage grade mappings, EMS-98 crosswalks, and heritage parameter dictionaries
