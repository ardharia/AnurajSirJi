"""Typology-specific parameter matcher connecting NDMA RVS Primer and Earthquake Reports.

Purely focuses on:
Earthquake Reports -> Building IDs -> Building Typology -> RVS Parameter Matching -> Building-Level Frequency -> Additional Observed Parameters
(NO Red/Yellow/Green scoring, no vulnerability scores, no weights, no ratings).
"""

from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import Any

from seismic_damage.config.settings import PROJECT_ROOT
from seismic_damage.linking.registry import build_building_registry
from seismic_damage.pipelines.rag.embedders import TfidfEmbedder
from seismic_damage.pipelines.rag.pipeline import RAGPipeline

RVS_DICT_CSV = PROJECT_ROOT / "data" / "reference" / "rvs_parameter_dictionary.csv"
BUILDING_OBS_CSV = PROJECT_ROOT / "data" / "analysis" / "building_parameter_observations.csv"
FREQUENCY_CSV = PROJECT_ROOT / "data" / "analysis" / "typology_parameter_frequency.csv"


def load_rvs_dictionary(csv_path: Path = RVS_DICT_CSV) -> list[dict[str, str]]:
    """Load NDMA RVS parameter dictionary."""
    if not csv_path.exists():
        from scripts.extract_ndma_rvs_dictionary import extract_rvs_dictionary
        return extract_rvs_dictionary(output_csv=csv_path)

    with csv_path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


# Additional structurally meaningful parameters observed in earthquake reports not in RVS Primer
ADDITIONAL_PARAMETERS_CATALOG = [
    {
        "parameter_id": "ADD_RC_01",
        "parameter_text": "Inadequate lap length in longitudinal column reinforcement",
        "pattern": r"inadequate lap lengths? in (?:the )?column reinforcement",
        "typology": "RC",
    },
    {
        "parameter_id": "ADD_RC_02",
        "parameter_text": "Lack of core confinement / 90-degree ties with excessive longitudinal spacing",
        "pattern": r"6mm diameter lateral ties|confinement to the core concrete|lateral ties.*90o hooks",
        "typology": "RC",
    },
    {
        "parameter_id": "ADD_RC_03",
        "parameter_text": "Rooftop swimming pool causing extreme mass irregularity and asymmetric torsional collapse",
        "pattern": r"swimming pool on (?:the )?roof",
        "typology": "RC",
    },
    {
        "parameter_id": "ADD_RC_04",
        "parameter_text": "Pancake collapse of multi-storey educational frame structure",
        "pattern": r"pancake collapse of \d+-storey.*school building",
        "typology": "RC",
    },
    {
        "parameter_id": "ADD_RC_05",
        "parameter_text": "Collapse of intermediate storey in commercial RC frame",
        "pattern": r"collapse of intermediate storey in a \d+-storey rc frame",
        "typology": "RC",
    },
    {
        "parameter_id": "ADD_RC_06",
        "parameter_text": "Dislodging of weakly connected overhead water tank appendages",
        "pattern": r"dislodging of (?:the )?weakly connected overhead storage water tanks|dislodging of water tank appendages",
        "typology": "RC",
    },
    {
        "parameter_id": "ADD_RC_07",
        "parameter_text": "Temporary steel section or wooden post propping of damaged open ground storeys",
        "pattern": r"propped.*by structural steel sections and wooded posts",
        "typology": "RC",
    },
    {
        "parameter_id": "ADD_RC_08",
        "parameter_text": "Improper post-earthquake column jacketing not extended to foundations or beam joints",
        "pattern": r"improper jacketing of columns|not extended till the foundation|not carried to the foundation",
        "typology": "RC",
    },
    {
        "parameter_id": "ADD_RC_09",
        "parameter_text": "Shear cracks in cantilever stub beam supporting floating columns",
        "pattern": r"shear cracks in the cantilever stub beam supporting a floating column",
        "typology": "RC",
    },
    {
        "parameter_id": "ADD_RC_10",
        "parameter_text": "Insufficient connection between RC elevator core wall and adjacent framing",
        "pattern": r"insufficient connection between the rc elevator core and rest of the building",
        "typology": "RC",
    },
    {
        "parameter_id": "ADD_RC_11",
        "parameter_text": "Dislodgement of precast roof panels atop precast frame due to inadequate connections",
        "pattern": r"roof panels were dislodged atop the precast frame/wall system due to inadequate connections",
        "typology": "RC",
    },
    {
        "parameter_id": "ADD_GEN_01",
        "parameter_text": "Common shared wall between adjacent buildings with differential response",
        "pattern": r"shared a common wall",
        "typology": "ALL",
    },
    {
        "parameter_id": "ADD_GEN_02",
        "parameter_text": "Failure of welded tie connections between prestressed concrete trusses",
        "pattern": r"failure of (?:the )?connection welding between the tie and the truss|failure of ties between prestressed concrete trusses",
        "typology": "ALL",
    },
    {
        "parameter_id": "ADD_RRSM_01",
        "parameter_text": "Out-of-plane collapse of stone masonry wall",
        "pattern": r"out-of-plane collapse of masonry wall",
        "typology": "RRSM",
    },
    {
        "parameter_id": "ADD_RRSM_02",
        "parameter_text": "Collapse of staircase and building corner in multi-storey stone masonry",
        "pattern": r"collapse of staircase and collapse of the corner of.*stone masonry",
        "typology": "RRSM",
    },
    {
        "parameter_id": "ADD_BCBM_01",
        "parameter_text": "Cyclic shear cracks in industrial brick masonry chimney",
        "pattern": r"cyclic shear cracks in the upper portion of.*brick chimney",
        "typology": "BCBM",
    },
]


def match_rvs_parameters(
    building: dict[str, Any],
    applicable_params: list[dict[str, str]],
    rag_context: str = "",
) -> list[dict[str, Any]]:
    """Match earthquake-report evidence for a building against applicable RVS parameters."""
    ev_text = building.get("evidence_text", "")
    desc = building.get("building_description", "")
    full_text = f"{ev_text} {desc} {rag_context}".strip()
    lowered = full_text.lower()

    matches: list[dict[str, Any]] = []

    for p in applicable_params:
        pid = p["parameter_id"]
        matched = False
        evidence_snippet = ""
        confidence = 0.85

        # RC Typology Parameter Rules
        if pid in ["RC_RVS_3.4", "RC_POST_3.3"]:
            if any(k in lowered for k in ["open ground storey", "open ground story", "open storey", "soft storey", "soft story"]):
                matched = True
                evidence_snippet = "Open ground storey / soft storey collapse or configuration evidenced."
        elif pid in ["RC_RVS_3.5", "RC_POST_3.4"]:
            if "floating column" in lowered or "floating columns" in lowered:
                matched = True
                evidence_snippet = "Floating columns / stub cantilever beam observed."
        elif pid in ["RC_POST_5.2", "RC_RVS_8.4"]:
            if "water tank" in lowered or "overhead storage" in lowered:
                matched = True
                evidence_snippet = "Overhead storage water tanks dislodged or unanchored on roof."
        elif pid == "RC_POST_3.5":
            if "shear crack" in lowered and "column" in lowered:
                matched = True
                evidence_snippet = "Main load resisting columns with shear cracks."
        elif pid == "RC_POST_3.8":
            if "spalling" in lowered:
                matched = True
                evidence_snippet = "Spalling of concrete cover in main columns."
        elif pid == "RC_RVS_9.1":
            if any(k in lowered for k in ["cracks in the cantilever", "shear cracks", "longitudinal cracks", "spalling in beams"]):
                matched = True
                evidence_snippet = "Cracks/shear cracks in beams/columns."
        elif pid == "RC_POST_2.3":
            if "liquefaction" in lowered or "liquefied" in lowered:
                matched = True
                evidence_snippet = "Liquefaction near building."
        elif pid == "RC_POST_3.9":
            if any(k in lowered for k in ["cracking in the masonry infill", "infill wall", "collapse of infill", "x cracking"]):
                matched = True
                evidence_snippet = "Cracking or damage in masonry infills."
        elif pid == "RC_RVS_8.2":
            if "elevator core" in lowered:
                matched = True
                evidence_snippet = "RC elevator core wall providing unsymmetrical stiffness."
        elif pid == "RC_RVS_7.3":
            if "cantilever" in lowered:
                matched = True
                evidence_snippet = "Cantilever stub beam / overhang projections."
        elif pid == "RC_RVS_3.6":
            if any(k in lowered for k in ["l-shaped", "irregular plan", "plan aspect"]):
                matched = True
                evidence_snippet = "L-shaped plan configuration."
        elif pid == "RC_RVS_9.2":
            if any(k in lowered for k in ["poor detailing", "inadequate spacing", "poor construction", "inadequate lap"]):
                matched = True
                evidence_snippet = "Poor detailing / inadequate spacing of lateral ties."
        elif pid == "RC_POST_5.1":
            if any(k in lowered for k in ["collapse of cantilever", "sunshade", "balconies"]):
                matched = True
                evidence_snippet = "Collapse of cantilever appendages."

        # RRSM Typology Parameter Rules
        elif pid == "RRSM_RVS_4.3":
            if "without lintel band" in lowered or "no continuous lintel" in lowered:
                matched = True
                evidence_snippet = "Stone masonry houses constructed without continuous lintel bands."
        elif pid in ["RRSM_RVS_4.5", "RRSM_POST_5.7"]:
            if "corner" in lowered and any(k in lowered for k in ["collapse", "separation", "separated"]):
                matched = True
                evidence_snippet = "Collapse/separation of masonry wall junctions at corners."
        elif pid in ["RRSM_RVS_4.6", "RRSM_RVS_5.1"]:
            if "mud mortar" in lowered:
                matched = True
                evidence_snippet = "Stone masonry constructed with mud mortar."
        elif pid == "RRSM_RVS_4.12":
            if "crack" in lowered:
                matched = True
                evidence_snippet = "Cracks observed in stone masonry walls."
        elif pid == "RRSM_POST_5.4":
            if "staircase" in lowered and "collapse" in lowered:
                matched = True
                evidence_snippet = "Collapse of staircase in stone masonry."
        elif pid == "RRSM_POST_5.8":
            if "out-of-plane" in lowered or "out of plumb" in lowered:
                matched = True
                evidence_snippet = "Out-of-plane wall displacement."

        # BCBM Typology Parameter Rules
        elif pid == "BCBM_POST_5.1":
            if "chimney" in lowered and any(k in lowered for k in ["collapse", "partial collapse"]):
                matched = True
                evidence_snippet = "Partial collapse of brick chimney."
        elif pid == "BCBM_POST_3.9":
            if "cyclic shear cracks" in lowered or "shear crack" in lowered:
                matched = True
                evidence_snippet = "Diagonal cyclic shear cracking in masonry wall / chimney."
        elif pid == "BCBM_RVS_4.9":
            if "crack" in lowered:
                matched = True
                evidence_snippet = "Cracks observed in brick masonry walls."
        elif pid == "BCBM_POST_3.3":
            if "railroad station" in lowered and "collapse" in lowered:
                matched = True
                evidence_snippet = "Collapse of brick masonry building at plinth/ground."

        if matched:
            matches.append({
                "building_id": building["building_id"],
                "event_id": building["event_id"],
                "typology": building["building_typology"],
                "rvs_parameter_id": pid,
                "rvs_parameter": p["parameter_text"],
                "parameter_category": p["parameter_category"],
                "parameter_source": "RVS_PRIMER",
                "observed": 1,
                "evidence_text": evidence_snippet or ev_text[:200],
                "source_document": building["source_document"],
                "page_number": building["page_number"],
                "match_confidence": f"{confidence:.2f}",
            })

    return matches


def match_additional_parameters(
    building: dict[str, Any],
    rag_context: str = "",
) -> list[dict[str, Any]]:
    """Identify structurally meaningful characteristics not present in the RVS Primer."""
    bldg_typ = building.get("building_typology", "UNKNOWN")
    ev_text = building.get("evidence_text", "")
    desc = building.get("building_description", "")
    full_text = f"{ev_text} {desc} {rag_context}".strip()

    matches: list[dict[str, Any]] = []

    for item in ADDITIONAL_PARAMETERS_CATALOG:
        # Check typology applicability
        if item["typology"] != "ALL" and item["typology"] != bldg_typ and bldg_typ != "UNKNOWN":
            continue

        if re.search(item["pattern"], full_text, re.I):
            matches.append({
                "building_id": building["building_id"],
                "event_id": building["event_id"],
                "typology": bldg_typ,
                "rvs_parameter_id": item["parameter_id"],
                "rvs_parameter": item["parameter_text"],
                "parameter_category": "Additional Earthquake Observation",
                "parameter_source": "ADDITIONAL_EARTHQUAKE_OBSERVED",
                "observed": 1,
                "evidence_text": desc or ev_text[:200],
                "source_document": building["source_document"],
                "page_number": building["page_number"],
                "match_confidence": "0.90",
            })

    return matches


def run_parameter_frequency_analysis(
    output_obs_path: Path = BUILDING_OBS_CSV,
    output_freq_path: Path = FREQUENCY_CSV,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Execute complete building-level frequency analysis pipeline."""
    # 1. Building registry
    registry = build_building_registry()
    print(f"Loaded {len(registry)} buildings from registry.")

    # 2. RVS parameter dictionary
    rvs_dictionary = load_rvs_dictionary()
    dict_by_typology: dict[str, list[dict[str, str]]] = {}
    for row in rvs_dictionary:
        dict_by_typology.setdefault(row["typology"], []).append(row)

    # 3. RAG pipeline for retrieval augmentation
    rag_pipe = RAGPipeline(embedder=TfidfEmbedder())

    # 4. Building-level observations
    # Strict deduplication key: (building_id, parameter_id)
    seen_observations: set[tuple[str, str]] = set()
    observations: list[dict[str, Any]] = []

    # Count unique building IDs per typology (excluding NON_BUILDING)
    typology_building_set: dict[str, set[str]] = {}
    for b in registry:
        typ = b["building_typology"]
        if typ != "NON_BUILDING":
            typology_building_set.setdefault(typ, set()).add(b["building_id"])

    for building in registry:
        bid = building["building_id"]
        typ = building["building_typology"]

        if typ == "NON_BUILDING":
            continue

        # Retrieve RAG context for this building
        query = building["evidence_text"] or f"Building {bid} Bhuj 2001 damage"
        rag_res = rag_pipe.retrieve(query, top_k=2)
        rag_context = rag_res.synthesized_context or ""

        # Retrieve citations to fill source document/page if not yet set
        if building["page_number"] == "N/A" and rag_res.citations:
            top_cit = rag_res.citations[0]
            building["page_number"] = str(top_cit.page_number or "N/A")
            if building["source_document"] == "bhuj_2001_reports":
                building["source_document"] = top_cit.document_id

        # A. Load ONLY RVS parameters applicable to this typology
        applicable_rvs = dict_by_typology.get(typ, [])
        rvs_matches = match_rvs_parameters(building, applicable_rvs, rag_context)

        for match in rvs_matches:
            key = (bid, match["rvs_parameter_id"])
            if key not in seen_observations:
                seen_observations.add(key)
                observations.append(match)

        # B. Match additional earthquake-observed parameters
        add_matches = match_additional_parameters(building, rag_context)
        for match in add_matches:
            key = (bid, match["rvs_parameter_id"])
            if key not in seen_observations:
                seen_observations.add(key)
                observations.append(match)

    # Save building-level observations CSV
    output_obs_path.parent.mkdir(parents=True, exist_ok=True)
    with output_obs_path.open("w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "building_id",
            "event_id",
            "typology",
            "parameter_id",
            "parameter_text",
            "parameter_source",
            "observed",
            "evidence_text",
            "source_document",
            "page_number",
            "match_confidence",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for o in observations:
            row = {
                "building_id": o["building_id"],
                "event_id": o["event_id"],
                "typology": o["typology"],
                "parameter_id": o["rvs_parameter_id"],
                "parameter_text": o["rvs_parameter"],
                "parameter_source": o["parameter_source"],
                "observed": o["observed"],
                "evidence_text": o["evidence_text"],
                "source_document": o["source_document"],
                "page_number": o["page_number"],
                "match_confidence": o["match_confidence"],
            }
            writer.writerow(row)

    print(f"Saved {len(observations)} building-level observations to {output_obs_path}")

    # 5. Calculate Building-Level Frequency
    # Frequency = number of unique buildings exhibiting the parameter
    frequency_map: dict[tuple[str, str, str, str, str], set[str]] = {}
    for o in observations:
        key = (
            o["event_id"],
            o["typology"],
            o["parameter_source"],
            o["rvs_parameter_id"],
            o["rvs_parameter"],
        )
        frequency_map.setdefault(key, set()).add(o["building_id"])

    frequency_rows: list[dict[str, Any]] = []
    for (event_id, typ, p_source, param_id, param_text), bldg_set in frequency_map.items():
        freq = len(bldg_set)
        total_bldgs = len(typology_building_set.get(typ, set()))
        freq_pct = (freq / total_bldgs) * 100.0 if total_bldgs > 0 else 0.0

        frequency_rows.append({
            "event_id": event_id,
            "typology": typ,
            "total_buildings": total_bldgs,
            "parameter_source": p_source,
            "parameter_id": param_id,
            "parameter": param_text,
            "frequency": freq,
            "frequency_percentage": f"{freq_pct:.1f}%",
        })

    # Sort frequency summary by: event_id, typology, frequency DESC
    frequency_rows.sort(key=lambda r: (r["event_id"], r["typology"], -r["frequency"]))

    output_freq_path.parent.mkdir(parents=True, exist_ok=True)
    with output_freq_path.open("w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "event_id",
            "typology",
            "total_buildings",
            "parameter_source",
            "parameter_id",
            "parameter",
            "frequency",
            "frequency_percentage",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(frequency_rows)

    print(f"Saved {len(frequency_rows)} frequency summary rows to {output_freq_path}")

    return observations, frequency_rows


def format_typology_summary_text(frequency_rows: list[dict[str, Any]]) -> str:
    """Format final user-facing summary strictly matching Section 12 specifications."""
    lines: list[str] = [
        "========================================================",
        "EARTHQUAKE EVENT: BHUJ 2001",
        "========================================================",
        "",
    ]

    # Group by typology
    by_typology: dict[str, list[dict[str, Any]]] = {}
    for r in frequency_rows:
        by_typology.setdefault(r["typology"], []).append(r)

    # Order typologies: RC, RRSM, BCBM, UNKNOWN, etc.
    preferred_order = ["RC", "RRSM", "BCBM", "CM", "MH", "DD", "EH", "UNKNOWN"]
    sorted_typologies = sorted(
        by_typology.keys(),
        key=lambda t: preferred_order.index(t) if t in preferred_order else 99,
    )

    for typ in sorted_typologies:
        rows = by_typology[typ]
        total_bldgs = rows[0]["total_buildings"]

        lines.append(f"TYPOLOGY: {typ}")
        lines.append(f"Number of buildings identified: {total_bldgs}")
        lines.append("")

        # RVS parameters
        rvs_rows = [r for r in rows if r["parameter_source"] == "RVS_PRIMER"]
        rvs_rows.sort(key=lambda x: -x["frequency"])

        lines.append("RVS PARAMETERS OBSERVED")
        lines.append("----------------------------------------------------------------------------------------------------")
        lines.append(f"{'Parameter':<78} {'Frequency':>10} {'% Typology':>10}")
        lines.append("----------------------------------------------------------------------------------------------------")
        if rvs_rows:
            for r in rvs_rows:
                p_text = r["parameter"][:76]
                lines.append(f"{p_text:<78} {r['frequency']:>10} {r['frequency_percentage']:>10}")
        else:
            lines.append("None observed.")
        lines.append("----------------------------------------------------------------------------------------------------")
        lines.append("")

        # Additional parameters
        add_rows = [r for r in rows if r["parameter_source"] == "ADDITIONAL_EARTHQUAKE_OBSERVED"]
        add_rows.sort(key=lambda x: -x["frequency"])

        lines.append("ADDITIONAL EARTHQUAKE-OBSERVED PARAMETERS")
        lines.append("----------------------------------------------------------------------------------------------------")
        lines.append(f"{'Parameter':<78} {'Frequency':>10} {'% Typology':>10}")
        lines.append("----------------------------------------------------------------------------------------------------")
        if add_rows:
            for r in add_rows:
                p_text = r["parameter"][:76]
                lines.append(f"{p_text:<78} {r['frequency']:>10} {r['frequency_percentage']:>10}")
        else:
            lines.append("None observed.")
        lines.append("----------------------------------------------------------------------------------------------------")
        lines.append("")

    return "\n".join(lines)


if __name__ == "__main__":
    obs, freqs = run_parameter_frequency_analysis()
    print("\n" + format_typology_summary_text(freqs))
