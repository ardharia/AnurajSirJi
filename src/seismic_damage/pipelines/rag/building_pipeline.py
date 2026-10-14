# src/seismic_damage/pipelines/rag/building_pipeline.py
"""Pipeline to extract buildings, assign deterministic UUIDs, classify typology, and gather evidence.

Outputs (CSV) are written to the repository's ``outputs`` directory:
- ``building_registry.csv`` – one row per unique building with provenance.
- ``building_typology.csv`` – mapping of building UUID to typology.
- ``parameter_evidence.csv`` – evidence status per building‑parameter.
- ``typology_parameter_frequencies.csv`` – frequency of each parameter across unique buildings.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, List, Tuple

from seismic_damage.utils.uuid_helper import build_building_uuid
from seismic_damage.pipelines.rag.building_extractor import (
    get_unique_building_ids,
    get_building_typology,
)
from seismic_damage.pipelines.rag.pipeline import RAGPipeline

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
OUTPUT_DIR = Path(__file__).resolve().parents[3] / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TYPOLOGY_PARAMS_DIR = Path(__file__).resolve().parents[4] / "data" / "typology_parameters"

def load_typology_parameters() -> Dict[str, List[str]]:
    """Return a mapping typology -> list of parameter strings."""
    mapping: Dict[str, List[str]] = {}
    for json_file in TYPOLOGY_PARAMS_DIR.glob("*.json"):
        typology = json_file.stem.upper()
        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        mapping[typology] = [item["parameter"] for item in data]
    return mapping

def build_fallback_map(typology_params: Dict[str, List[str]]) -> Dict[str, str]:
    """Simple keyword‑to‑typology fallback derived from parameters."""
    fallback: Dict[str, str] = {}
    for typ, params in typology_params.items():
        for p in params:
            kw = p.split()[0].lower()
            fallback[kw] = typ
    return fallback

def get_building_details(building_id: str, rag: RAGPipeline) -> Tuple[Path, int, str]:
    """Query RAG for provenance JSON (report_path, page_number, description)."""
    prompt = (
        f"Provide a JSON object with keys 'report_path', 'page_number' and 'description' for building identifier {building_id} as mentioned in the Bhuj 2001 reports. "
        "The description should be the exact excerpt that mentions the building's earthquake response. Do not add any additional commentary."
    )
    result = rag.run(prompt)
    try:
        info = json.loads(result.answer)
        return Path(info["report_path"]).resolve(), int(info["page_number"]), info["description"]
    except Exception:
        return Path(""), -1, result.answer.strip()

def main() -> None:
    rag = RAGPipeline()
    if rag.store is None:
        rag.index_corpus()

    building_ids = get_unique_building_ids()
    typology_params = load_typology_parameters()
    fallback_map = build_fallback_map(typology_params)

    registry_rows: List[Dict[str, str]] = []
    typology_rows: List[Dict[str, str]] = []
    evidence_rows: List[Dict[str, str]] = []
    freq_counter: Dict[Tuple[str, str], int] = {}

    for bid in building_ids:
        report_path, page, description = get_building_details(bid, rag)
        building_uuid = build_building_uuid(report_path, page, description)
        registry_rows.append({
            "building_uuid": building_uuid,
            "building_id": bid,
            "report_path": str(report_path),
            "page_number": str(page),
            "description": description,
        })
        _, typology = get_building_typology(bid)
        if not typology or typology == "UNKNOWN":
            # fallback via keyword map
            for word in description.lower().split():
                if word in fallback_map:
                    typology = fallback_map[word]
                    break
            else:
                typology = "UNKNOWN"
        typology_rows.append({"building_uuid": building_uuid, "typology": typology})
        params = typology_params.get(typology, [])
        for param in params:
            query = f"Is the following parameter mentioned for building {bid} in the Bhuj 2001 reports? Parameter: {param}"
            answer = rag.run(query).answer or ""
            answer_lc = answer.lower()
            if "yes" in answer_lc:
                status = "PRESENT"
            elif "no" in answer_lc:
                status = "ABSENT"
            elif answer.strip() == "":
                status = "NOT_REPORTED"
            else:
                status = "UNKNOWN"
            evidence_rows.append({
                "building_uuid": building_uuid,
                "typology": typology,
                "parameter": param,
                "status": status,
            })
            if status == "PRESENT":
                key = (typology, param)
                freq_counter[key] = freq_counter.get(key, 0) + 1

    def write_csv(path: Path, fieldnames: List[str], rows: List[Dict[str, str]]) -> None:
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    write_csv(OUTPUT_DIR / "building_registry.csv",
              ["building_uuid", "building_id", "report_path", "page_number", "description"],
              registry_rows)
    write_csv(OUTPUT_DIR / "building_typology.csv",
              ["building_uuid", "typology"],
              typology_rows)
    write_csv(OUTPUT_DIR / "parameter_evidence.csv",
              ["building_uuid", "typology", "parameter", "status"],
              evidence_rows)
    freq_rows = [{"typology": typ, "parameter": param, "frequency": str(cnt)}
                 for (typ, param), cnt in freq_counter.items()]
    write_csv(OUTPUT_DIR / "typology_parameter_frequencies.csv",
              ["typology", "parameter", "frequency"],
              freq_rows)
    print(f"✅ Pipeline completed. CSV artefacts written to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
