# PROJECT_STATE.md — Multimodal Seismic Damage Assessment

> Last updated: 2026-09-17

## Current System Summary

Modular Python scaffold (v0.1.0) for post-earthquake building damage assessment
combining **RAG** (retrieval-augmented generation) and **VLM** (vision-language model)
pipelines. Both pipelines are **independent** and strictly not fused.

### Working Components

| Module | Status | Description |
|--------|--------|-------------|
| `config/` + `Settings` | ✅ Working | YAML defaults + `.env` overrides via Pydantic Settings |
| `schemas/` | ✅ Working | `BuildingParameters`, `DamageAssessment`, `ExtractionResult`, `NormalizedAssessmentRecord`, `TypologyAttributes`, `VulnerabilityAttributes`, `DamageAttributes`, `ProvenanceInfo`, `RAGResult`, `VLMResult`, `TextChunk`, `BuildingRecord`, `EvidenceLink` |
| `pipelines/rag/` | ✅ Working | FAISS/NumPy vector store, TF-IDF/SentenceTransformer embedders, extractive answer with citations, evidence sufficiency gate, refusal |
| `pipelines/vlm/` | ✅ Working | MockVLM (tests), OpenAICompatibleVLM (real), UnavailableVLM (fallback); structured JSON observation schema |
| `pipelines/consolidation.py` | ✅ Working | Consolidates building-level RAG and VLM summaries; exports CSV/JSONL; generates modality plots |
| `ingestion/` | ✅ Working | PDF extraction (PyMuPDF), text cleaning, page-level chunking (1,732 chunks from 11 Bhuj reports) |
| `extraction/` | ✅ Working | Rule-based `parse_damage_text()` for captions/report text |
| `linking/` | ✅ Working | Caption-group building catalog (110 canonical buildings, 119 evidence links); `data/metadata/building_links.csv` |
| `validation/normalize.py` | ✅ Working | Controlled vocabulary mapping for materials, structural systems, damage grades (0-5), crack patterns, collapse modes |
| `validation/cross_validation.py` | ✅ Working | Parameter-level metrics (ordinal/binary/numerical/categorical), QWK; GT vs RAG vs VLM comparisons |
| `stats/` | ✅ Working | Frequency tables, parameter ranking, error distributions, Matplotlib figures |
| `fragility/` | ✅ Stubbed | Lognormal fragility interface defined; returns `not_implemented` |
| `orchestration/` | ✅ Stubbed | End-to-end `run_assessment()` composition |

### Test Suite

- **22 unit tests** — all passing (pytest 9.1.1, Python 3.11.0)
- Tests cover: damage assessment schema, building parameters, pipeline stubs, settings, ground truth builder, normalization, cross-validation integration, building summaries consolidation, and visualizations.

### Indian Reference Framework (Steps 1–2)

Located in `reference/`:
- `parameter_dictionary.csv`: 53 parameters across typology, vulnerability, and post-earthquake damage with citations.
- `damage_grade_mapping.csv`: Complete EMS-98 and IS 13935:2009 damage grade scale (0–5) with masonry vs RC indicators.
- `ems98_crosswalk.csv`: 15 typologies mapped across EMS-98, IS 13935, and NDMA RVS Primer codes.

### Ground & Reference Dataset (Step 3)

Located in `data/processed/reference/`:
- `building_ground_truth.csv` and `building_ground_truth.jsonl`: 110 canonical buildings with explicit separation of pre-earthquake vulnerability, post-earthquake damage, and building typology.
- `data/metadata/building_links.csv`: 119 evidence links mapping images to canonical buildings.

### Consolidated RAG & VLM Summaries (Steps 5–6)

Located in `data/processed/results/`:
- `rag_building_summary.csv` and `rag_building_summary.jsonl`: 110 buildings (54 damage grades observed).
- `vlm_building_summary.csv` and `vlm_building_summary.jsonl`: 110 buildings (48 damage grades observed).
- `cross_validation_report.json`: Full 3-way evaluation metrics (GT vs RAG vs VLM).
- Separate visualizations in `data/processed/results/figures/`:
  - `rag_damage_grade_frequency.png`
  - `rag_parameter_coverage.png`
  - `vlm_damage_grade_frequency.png`
  - `vlm_parameter_coverage.png`
