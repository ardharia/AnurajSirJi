# FINAL VALIDATION REPORT
## Bhuj 2001 AI-Assisted Multimodal Earthquake Vulnerability Assessment
**Date:** 2026-09-29 | **Python:** 3.13.15 | **Pytest:** 9.1.1 | **FAISS:** 1.15.1

---

## 1. EXECUTIVE SUMMARY

This report documents the results of a **complete engineering audit and actual pipeline execution** against the Bhuj 2001 dataset (Steps 1–19). The audit was performed against the real dataset, not against code presence alone.

### Critical Findings

| # | Finding | Severity |
|---|---------|----------|
| F1 | **RAG pipeline was never executed against real PDF data.** All 110 RAG building records have confidence=0.0 and all parameters NULL. | **CRITICAL** |
| F2 | **VLM "output" is NOT from a real VLM model.** It is the same caption-parsed annotation as the ground truth → 100% accuracy on all metrics — textbook data leakage. | **CRITICAL** |
| F3 | **PyMuPDF was not installed** → all 11 PDF ingestion attempts silently failed before this audit. | **HIGH** |
| F4 | **OpenAI API key not configured.** VLM falls back to UnavailableVLM. Zero real visual inferences. | **HIGH** |
| F5 | **sentence-transformers not installed** prior to audit. RAG used TF-IDF fallback. | **MEDIUM** |
| F6 | **0 real damage photographs exist in data/raw/.** 119 images are URL references only; download script never executed. | **HIGH** |
| F7 | **Reliability weights are all RAG=0.0, VLM=1.0** — artifact of F1/F2, not a real result. | **HIGH** |
| F8 | Ablation Exp_A (RAG-only)=0.0, Exp_B–J=1.0 — trivially because VLM output equals GT. | **HIGH** |
| F9 | Pre-earthquake vulnerability pipeline is executable but returns "insufficient_evidence" for most buildings due to low caption coverage. | **MEDIUM** |
| F10 | Heritage module is implemented but not connected to any data pipeline. | **LOW** |

### What Is Actually Working

- PDF ingestion: 11 PDFs → 432 pages → 1,732 text chunks (after fixing F3)
- RAG FAISS index: 1,732 vectors, retrieval works (TF-IDF fallback)
- Ground truth builder: 110 buildings, correct schema, proper pre/post-EQ separation
- Building-level aggregation: 119 image entries → 110 unique buildings (correct)
- Cross-validation framework: metrics code correct (ordinal/categorical/binary/numerical)
- Reliability weight formula: parameter-specific, NOT 50:50 default (correctly implemented)
- Vulnerability/Risk modules: importable and executable with provided data
- All 67 unit tests pass in 1.74s

---

## 2. DATASET VERIFICATION

### 2.1 Physical Files

| Asset | Expected | Actual on Disk |
|-------|----------|----------------|
| PDF reports | 11 | **11** (data/reports/bhuj_2001_github/) |
| Damage photographs | 119 | **0** — URL references only |
| Building links CSV | 1 | **1** (119 rows + header) |
| Annotation CSVs | 4 | **4** |

**The 119 images exist as URL references only. `scripts/download_bhuj_images.py` exists but was never run.**

### 2.2 Building-Level Statistics

| Metric | Value |
|--------|-------|
| Total unique buildings | 110 |
| Total image records | 119 |
| Images per building | min=1, max=3, mean=1.08 |
| Source: IITK_NICEE_Bhuj_RC | 24 buildings |
| Source: GEER_Bhuj_2001 | 53 buildings |
| Source: SlideShare_Bhuj_2001 | 33 buildings |

### 2.3 Ground Truth Parameter Coverage (from caption parsing)

| Parameter | Non-Null | Coverage |
|-----------|---------|---------|
| damage_grade | 48/110 | 44% |
| collapse_mode | 39/110 | 35% |
| material_type | 29/110 | 26% |
| structural_system | 14/110 | 13% |
| number_of_stories | 11/110 | 10% |
| crack_severity | 10/110 | 9% |
| soft_story_failure | 8/110 | 7% |
| wall_failure | 6/110 | 5% |
| crack_pattern | 3/110 | 3% |
| soft_story (pre-EQ) | 0/110 | 0% |
| soil_type | 0/110 | 0% |
| pga | 0/110 | 0% |

### 2.4 Damage Grade Distribution

| DG | Count | Description |
|----|-------|-------------|
| 5 | 9 | Total destruction |
| 4 | 29 | Heavy structural damage |
| 3 | 1 | Moderate-heavy |
| 2 | 7 | Moderate |
| 1 | 2 | Slight |
| Missing | 62 | Indeterminate from caption |

---

## 3. STEP 1–19 IMPLEMENTATION AUDIT

| Step | Description | Implemented | Executable | Actual Result | Issues |
|------|-------------|------------|-----------|--------------|--------|
| 1 | PDF ingestion | YES | YES (after fix) | 432 pages, 1732 chunks | PyMuPDF missing |
| 2 | FAISS RAG pipeline | YES | YES (TF-IDF) | 1732 vectors indexed | sentence-transformers missing |
| 3 | VLM pipeline | YES | NO | UnavailableVLM, 0 inferences | No API key; 0 images |
| 4 | Evidence linking | YES | YES | 110 unique buildings | Correct |
| 5 | Ground truth records | YES | YES | 110 buildings, caption-parsed | NOT from real structural surveys |
| 6 | RAG parameter extraction | YES (framework) | NO | All parameters NULL | LLM extractor is a no-op stub |
| 7 | VLM parameter extraction | YES (framework) | NO | Caption-copy = GT leakage | VLM output equals ground truth |
| 8 | Cross-validation framework | YES | YES | Metrics computed correctly | Cannot produce valid numbers |
| 9 | Statistical frequency analysis | YES | YES | Figures generated | Based on leakage data |
| 10 | Building-level aggregation | YES | YES | Correct | Working correctly |
| 11 | Missing/poor VLM robustness | YES | PARTIAL | Scenarios A-J defined; synthetic | No real images |
| 12 | RAG optimization | YES | PARTIAL | TF-IDF configs tested | No LLM extractor |
| 13 | VLM optimization | YES | PARTIAL | Configs defined | No real VLM |
| 14 | Calibrated fusion | YES | PARTIAL | Fusion logic correct | Weights from leakage data |
| 15 | Robustness experiments | YES | PARTIAL | 10 scenarios run | Synthetic data only |
| 16 | Ablation study | YES | PARTIAL | 10 experiments run | Exp_A=0.0, rest=1.0 (leakage) |
| 17 | Pre-earthquake vulnerability | YES | YES | 110 assessments | Most "insufficient_evidence" |
| 18 | Seismic risk framework | YES | YES | 110 risk records | Correct H/S/V/E separation |
| 19 | Heritage extension | YES | YES | Module importable | Not connected to data pipeline |

---

## 4. RAG EXECUTION RESULTS

### PDF Ingestion (executed after installing PyMuPDF)

| Document | Status | Pages | Chunks |
|----------|--------|-------|--------|
| bhuj_github_001 (EEFIT) | ok | 120 | 486 |
| bhuj_github_002 (EERI) | ok | 16 | 133 |
| bhuj_github_003 (EQRBhuj) | ok | 60 | 85 |
| bhuj_github_004 (Gujarat EQ) | ok | 148 | 518 |
| bhuj_github_005–011 | ok | 88 | 510 |
| **TOTAL** | 0 skipped | **432** | **1,732** |

### RAG Retrieval (working)
- Embedder: TF-IDF 4096-dim (sentence-transformers was unavailable)
- Index: 1,732 vectors
- Top-k=5 retrieval scores: 0.55–0.86
- evidence_sufficient=True for all tested buildings

### RAG Parameter Extraction — NOT EXECUTED
The LLM-based structured extractor (`extractor(context)` call in `pipeline.py:213`) is a no-op stub returning `{}`. All 110 RAG records: confidence=0.0, all 12 parameters NULL.

---

## 5. VLM EXECUTION RESULTS

### Status
- Provider: openai | API key: NOT CONFIGURED
- Fallback backend: UnavailableVLM
- Images on disk: 0
- Real VLM inferences: 0

### Data Leakage Finding
`data/processed/results/vlm_building_summary.jsonl` was generated by the same `parse_damage_text()` function as the ground truth. `evidence_source="vlm"` is a label, not an indicator of actual VLM model inference.

**Proof — VLM vs Ground Truth cross-validation:**

| Parameter | n | accuracy | F1/MAE |
|-----------|---|----------|--------|
| damage_grade | 48 | — | MAE=0.0, QWK=1.0 |
| material_type | 29 | 1.0 | F1=1.0 |
| structural_system | 14 | 1.0 | F1=1.0 |
| collapse_mode | 39 | 1.0 | F1=1.0 |
| crack_severity | 10 | 1.0 | F1=1.0 |
| number_of_stories | 11 | — | MAE=0.0 |
| soft_story_failure | 8 | 1.0 | F1=1.0 |
| wall_failure | 6 | 1.0 | F1=1.0 |

**These are not valid research metrics. They are confirmation that both datasets share the same origin.**

---

## 6. GROUND TRUTH VERIFICATION

### Provenance

| Layer | Source | Valid? |
|-------|--------|--------|
| Raw evidence | IITK NICEE / GEER captions + URLs | YES |
| Adjudicated reference | parse_damage_text() on captions | YES (weakly supervised) |
| RAG prediction | Empty stubs | NO |
| VLM prediction | Same caption parsing as GT | NO — leakage |
| Fused prediction | Derived from leakage | NO |

### Pre/Post-EQ Separation
Correctly implemented:
- Pre-EQ typology: material_type, structural_system, number_of_stories, occupancy
- Pre-EQ vulnerability: soft_story, plan_irregularity (keyword-based)
- Post-EQ damage: damage_grade, collapse_mode, crack_severity, etc.

---

## 7. RAG PERFORMANCE
**NOT EVALUABLE** — all RAG records have zero parameter coverage.

---

## 8. VLM PERFORMANCE
**INVALID** — data leakage (VLM output = ground truth). See Section 5.

---

## 9. FUSION PERFORMANCE
**INVALID** — derived from leakage data. Fusion logic is methodologically correct.

---

## 10. RELIABILITY WEIGHTS

| Parameter | RAG weight | VLM weight | Note |
|-----------|-----------|-----------|------|
| All evaluable parameters | 0.0 | 1.0 | RAG n=0 (stub), VLM n=GT (leakage) |
| magnitude, pga, soil_type | 0.0 | 0.0 | Insufficient from both |

**Formula is correct** (parameter-specific, not 50:50 default). Current values are artifacts of F1/F2.

---

## 11. PARAMETER IMPORTANCE
Correctly separated from source reliability in code. No valid data to compute rankings.

---

## 12. MISSING/POOR VLM ROBUSTNESS
Framework correct. 10 scenarios defined. Missing VLM → None, not 0 (verified in tests).
Cannot execute real image degradation without downloaded photographs.

---

## 13. ABLATION STUDY

| Exp | Name | Score | Validity |
|-----|------|-------|---------|
| A | RAG only | 0.0 | INVALID — RAG is null |
| B | VLM only | 1.0 | INVALID — VLM = GT |
| C-J | All others | 1.0 | INVALID — same root cause |

---

## 14. PRE-EARTHQUAKE VULNERABILITY PIPELINE

Executable. Results on 110 buildings:
- Most: vulnerability_class="insufficient_evidence" (no typology in caption)
- Buildings with material_type+structural_system: get full RVS scores + EMS-98 class
- Hazard / site / exposure / vulnerability / damage: correctly separated across risk/ module

---

## 15. ERRORS AND ISSUES

| Error | Fixed? |
|-------|--------|
| pymupdf not installed | YES — pip3 install pymupdf |
| sentence-transformers not installed | YES — pip3 install sentence-transformers |
| openai not installed | NO — no API key configured |
| langchain not installed | NO — not required for current functionality |
| RAG LLM extractor is a no-op stub | NO — requires LLM API |
| VLM records = GT (data leakage) | NO — requires real VLM |
| 0 images on disk | NO — requires download |

---

## 16. FIXES APPLIED

1. `pip3 install pymupdf` → PDF ingestion now works
2. `pip3 install sentence-transformers` → dense embeddings available
3. Verified FAISS retrieval against real Bhuj text chunks

---

## 17. REMAINING LIMITATIONS

| Limitation | Resolution |
|------------|-----------|
| No LLM API key | Set LLM_API_KEY in .env |
| No VLM API key | Set VLM_API_KEY in .env |
| 0 images downloaded | python3 scripts/download_bhuj_images.py |
| Caption GT low coverage (3–44%) | Manual annotation needed for full evaluation |
| RAG extractor stub | Implement LLM extraction prompt in pipeline.py |
| Robustness/ablation synthetic | Requires real VLM + images |
| Heritage not wired | Wire heritage/ to data pipeline |

---

## 18. REPRODUCIBILITY INSTRUCTIONS

```bash
cd /Users/anujpal/Downloads/deploy/AnurajSirJi

# Install missing dependencies
pip3 install pymupdf sentence-transformers
pip3 install -e .

# Download images
python3 scripts/download_bhuj_images.py

# Configure API keys
cp .env.example .env
# Edit .env: set VLM_API_KEY and LLM_API_KEY

# Ingest PDFs
python3 -c "from seismic_damage.ingestion.pdf import ingest_pdf_reports; r=ingest_pdf_reports(); print(r.n_documents, r.n_pages, r.n_chunks)"

# Build RAG index
python3 -c "from seismic_damage.pipelines.rag.pipeline import RAGPipeline; rag=RAGPipeline(); print(rag.index_corpus())"

# Build ground truth
python3 -c "from seismic_damage.linking.ground_truth import build_ground_truth_dataset; r,_,_=build_ground_truth_dataset(); print(len(r))"

# Run tests
python3 -m pytest tests/ -v

# Run vulnerability + risk
python3 -c "
from seismic_damage.linking.ground_truth import build_ground_truth_dataset
from seismic_damage.vulnerability.assessment import assess_all_buildings_vulnerability
from seismic_damage.risk.assessment import assess_all_buildings_risk
records,_,_ = build_ground_truth_dataset()
print('Vuln:', len(assess_all_buildings_vulnerability(records)))
print('Risk:', len(assess_all_buildings_risk(records)))
"
```

---

## APPENDIX: PARAMETER FRAMEWORK AUDIT

### NDMA RVS Basic Scores (Implemented)

| Typology | Score | Reference |
|---------|-------|-----------|
| rubble_masonry | 1.5 | NDMA RVS Zone V |
| burnt_clay_brick | 2.2 | NDMA RVS |
| unreinforced_masonry | 2.0 | NDMA RVS |
| rc_frame_open_ground | 2.2 | NDMA RVS (soft-story) |
| rc_frame_non_ductile | 2.5 | NDMA RVS (OMRF) |
| rc_frame_ductile | 3.5 | NDMA RVS (SMRF) |

### RVS Modifiers (Implemented)

| Modifier | Delta | Reference |
|---------|-------|-----------|
| soft_story | -0.8 | NDMA RVS |
| plan_irregularity | -0.4 | NDMA RVS |
| vertical_irregularity | -0.4 | NDMA RVS |
| short_columns | -0.5 | IS 1893:2016 |
| floating_columns | -0.6 | IS 1893:2016 |

### Evaluation Parameters (metrics.py)

- ORDINAL: damage_grade
- BINARY: soft_story_failure, wall_failure
- NUMERICAL: number_of_stories, pga, magnitude
- CATEGORICAL: structural_system, material_type, crack_severity, crack_pattern, collapse_mode, soil_type

**Assessment:** Framework aligns with Indian seismic context. No invented parameters. Consistent naming. Source reliability and parameter importance are correctly separated in code.

---

*Report generated: 2026-09-29 | Bhuj 2001 Engineering Validation Audit*
