# Multimodal Seismic Damage Assessment

Modular Python scaffold for post-earthquake damage assessment that combines
**retrieval-augmented generation (RAG)** and **vision-language models (VLM)**
with structured parameter extraction, validation, and probabilistic fragility analysis.

> Status: **scaffold only** — pipelines and analysis are stubbed with typed interfaces
> and Pydantic schemas. Full model/provider wiring is intentionally deferred.

## Architecture

```
                    ┌─────────────────────┐
                    │  AssessmentRequest  │
                    └──────────┬──────────┘
                               │
              ┌────────────────┴────────────────┐
              ▼                                 ▼
     ┌─────────────────┐               ┌─────────────────┐
     │  RAG Pipeline   │               │  VLM Pipeline   │
     │  (independent)  │               │  (independent)  │
     └────────┬────────┘               └────────┬────────┘
              │                                 │
              └────────────────┬────────────────┘
                               ▼
                    ┌─────────────────────┐
                    │ Parameter Extraction│
                    └──────────┬──────────┘
                               ▼
                    ┌─────────────────────┐
                    │     Validation      │
                    └──────────┬──────────┘
                               ▼
                    ┌─────────────────────┐
                    │ Fragility Analysis  │
                    └─────────────────────┘
```

| Module | Role |
|--------|------|
| `pipelines.rag` | Retrieve seismic codes, fragility tables, case reports |
| `pipelines.vlm` | Inspect building imagery for damage cues and attributes |
| `extraction` | Fuse multimodal hints into structured parameters |
| `validation` | Enforce required fields, confidence, cross-modal checks |
| `fragility` | Probabilistic damage-state exceedance given IM |
| `orchestration` | Optional end-to-end composition (pipelines stay callable alone) |

## Project layout

```
earthquake/
├── config/
│   └── defaults.yaml          # Default YAML configuration
├── src/seismic_damage/
│   ├── config/                # Pydantic Settings loader
│   ├── schemas/               # Shared Pydantic models
│   ├── pipelines/
│   │   ├── rag/
│   │   └── vlm/
│   ├── extraction/
│   ├── validation/
│   ├── fragility/
│   └── orchestration/
├── data/
│   ├── raw/
│   ├── processed/
│   └── knowledge/             # RAG corpus
├── tests/
├── scripts/
├── notebooks/
├── .env.example
├── requirements.txt
├── pyproject.toml
└── README.md
```

## Setup

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate

pip install -e ".[dev]"
# or
pip install -r requirements.txt
pip install -e .

copy .env.example .env   # Windows
# cp .env.example .env   # macOS / Linux
```

## Configuration

Configuration is layered:

1. `config/defaults.yaml` — baseline values
2. `.env` / environment variables — secrets and overrides
3. Programmatic `Settings` / `load_settings(path)` — runtime control

```python
from seismic_damage.config import get_settings, load_settings

settings = get_settings()
print(settings.rag.embedding_model)
print(settings.fragility.method)

# Custom YAML
custom = load_settings("config/defaults.yaml")
```

Important environment variables (see `.env.example`):

- `VLM_API_KEY` / `LLM_API_KEY`
- `RAG_ENABLED` / `VLM_ENABLED` (via YAML; flat toggles planned)
- `FRAGILITY_METHOD`, `FRAGILITY_N_SAMPLES`

## Usage (scaffold APIs)

```python
from seismic_damage.pipelines.rag import run_rag
from seismic_damage.pipelines.vlm import run_vlm
from seismic_damage.orchestration import run_assessment
from seismic_damage.schemas.pipeline import AssessmentRequest

# Independent pipelines
rag = run_rag("RC moment frame soft-story fragility PGA")
vlm = run_vlm(["data/raw/example.jpg"])

# Optional composed workflow
result = run_assessment(
    AssessmentRequest(
        text_query="three-story RC residential, soft story",
        image_paths=["data/raw/example.jpg"],
        intensity_value=0.35,
    )
)
print(result.validation_passed, result.validation_messages)
```

## Schemas

Core Pydantic models live under `seismic_damage.schemas`:

- `BuildingParameters` — taxonomy / structural attributes
- `ExtractionResult` / `ParameterValue` — structured extraction with provenance
- `RAGResult` / `VLMResult` — modality-specific outputs
- `FragilityInput` / `FragilityResult` — probabilistic analysis I/O
- `AssessmentRequest` / `AssessmentResult` — orchestration contract

## Roadmap

1. Wire RAG indexing + vector retrieval
2. Wire VLM provider client and damage prompt schemas
3. Implement LLM-backed structured extraction and conflict resolution
4. Implement lognormal fragility evaluation and sampling
5. Add integration tests and example notebooks

## License

Proprietary / TBD.
