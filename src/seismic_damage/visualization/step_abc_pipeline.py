import re
import pathlib
import pandas as pd
from pathlib import Path
import json

# RAG imports
from seismic_damage.pipelines.rag.pipeline import RAGPipeline

# PDF parsing (PyPDF2)
try:
    from PyPDF2 import PdfReader
except ImportError:
    raise ImportError("PyPDF2 is required to parse NDMA_RVS_Primer PDF. Install it first.")

# ---------------------------------------------------------------------------
# STEP A – Retrieve all unique building IDs from the Bhuj PDFs via RAG
# ---------------------------------------------------------------------------

def get_building_ids(rag: RAGPipeline) -> list[str]:
    query = "List every unique building identifier mentioned in these Bhuj 2001 reports."
    result = rag.run(query)
    # The answer is a plain text list – extract alphanumeric IDs (e.g., BHJ_B001)
    ids = re.findall(r"[A-Z]{3,}_B\d{3,}", result.answer)
    # Fallback: split on commas / whitespace if pattern fails
    if not ids:
        ids = [tok.strip() for tok in re.split(r"[,;\s]+", result.answer) if tok]
    return sorted(set(ids))

# ---------------------------------------------------------------------------
# STEP B – Load typology‑wise NDMA‑RVS parameter list from the Primer PDF
# ---------------------------------------------------------------------------

def load_ndma_parameter_dict(pdf_path: Path) -> dict[str, set[str]]:
    """Return a mapping typology -> set of parameter IDs (e.g., RC_RVS_3.4).

    The Primer PDF contains tables like:
        RC – RC_RVS_3.4, RC_RVS_3.6, ADD_RC_01, …
        RRSM – RRSM_RVS_4.2, …
        BCBM – BCBM_RVS_4.9, …
    We use a simple regex to pull out tokens that match the pattern
    <TY>_RVS_\d+\.\d+ or ADD_<TY>_\d+ etc.
    """
    reader = PdfReader(str(pdf_path))
    text = "\n".join(page.extract_text() or "" for page in reader.pages)
    # Normalize whitespace
    text = re.sub(r"\s+", " ", text)
    typology_map: dict[str, set[str]] = {"RC": set(), "RRSM": set(), "BCBM": set()}
    # Look for lines that start with a typology name followed by a dash or colon
    for typ in typology_map.keys():
        # Grab a chunk of text after the typology name up to the next typology name
        pattern = rf"{typ}[\s\-:]*([A-Za-z0-9_,\s]+)"
        matches = re.findall(pattern, text)
        for m in matches:
            # Extract parameter ids from the captured group
            params = re.findall(r"[A-Z]{2,}_RVS_\d+\.\d+|ADD_[A-Z]{2,}_\d+|[A-Z]{2,}_\w+", m)
            typology_map[typ].update(params)
    return typology_map

# ---------------------------------------------------------------------------
# STEP B – For every building ID, retrieve all sentences that mention it and
# extract any parameter‑like token. Then map to typology using the dictionary.
# ---------------------------------------------------------------------------

def extract_parameters_for_buildings(rag: RAGPipeline, building_ids: list[str], ndma_dict: dict[str, set[str]]) -> pd.DataFrame:
    rows = []
    param_pattern = re.compile(r"[A-Z]{2,}_RVS_\d+\.\d+|ADD_[A-Z]{2,}_\d+|[A-Z]{2,}_\w+")
    for b_id in building_ids:
        # Retrieve up to 20 relevant chunks for the building ID
        query = f"Give me all sentences that mention building {b_id} and list any parameter identifiers (e.g., RC_RVS_3.4)."
        result = rag.run(query)
        # The RAG answer may be a long string; extract param tokens
        found_params = set(param_pattern.findall(result.answer))
        # Determine typology for each param using NDMA dict (first match wins)
        for p in found_params:
            typ = None
            for t, pool in ndma_dict.items():
                if p in pool:
                    typ = t
                    break
            # If not found in NDMA list, keep typology as "UNKNOWN"
            typ = typ or "UNKNOWN"
            rows.append({"building_id": b_id, "parameter": p, "typology": typ})
    return pd.DataFrame(rows)

# ---------------------------------------------------------------------------
# STEP C – Compute frequency per typology and plot (reuse existing plot script)
# ---------------------------------------------------------------------------

def compute_and_save_frequency(df: pd.DataFrame, out_csv: Path):
    # Count distinct buildings per (typology, parameter)
    freq = (
        df.groupby(["typology", "parameter", "building_id"])
        .size()
        .reset_index(name="count")
    )
    # Collapse to building count per parameter
    freq2 = (
        freq.groupby(["typology", "parameter"]).agg({"building_id": "nunique", "count": "sum"})
        .rename(columns={"building_id": "building_count", "count": "mentions"})
        .reset_index()
    )
    freq2.to_csv(out_csv, index=False)
    print(f"✅ Frequency CSV written to {out_csv}")
    return freq2

def main():
    # Paths
    pdf_path = Path("reference/source_documents/NDMA_RVS_Primer-43-73.pdf")
    out_csv = Path("data/analysis/typology_parameter_frequency_full.csv")

    rag = RAGPipeline()
    # Ensure vector store is available (index if needed)
    if rag.store is None:
        rag.index_corpus()

    print("Step A - retrieving building IDs ...")
    building_ids = get_building_ids(rag)
    print(f"Found {len(building_ids)} building IDs.")

    print("Step B - loading NDMA parameter dictionary ...")
    ndma_dict = load_ndma_parameter_dict(pdf_path)
    for t, vals in ndma_dict.items():
        print(f"{t}: {len(vals)} parameters loaded.")

    print("🔎 Extracting parameters for each building …")
    building_params_df = extract_parameters_for_buildings(rag, building_ids, ndma_dict)
    print(f"Extracted {len(building_params_df)} building‑parameter rows.")

    print("Step C - computing frequency table ...")
    freq_df = compute_and_save_frequency(building_params_df, out_csv)
    print(freq_df.head())

    # Optional: reuse existing plot script to visualise (RC, RRSM, BCBM)
    try:
        import subprocess
        plot_script = Path("src/seismic_damage/visualization/plot_parameter_frequency.py")
        for typ in ["RC", "RRSM", "BCBM"]:
            cmd = [
                "python",
                str(plot_script),
                "--csv",
                str(out_csv),
                "--typology",
                typ,
                "--out-dir",
                "data/analysis/plots",
                "--mode",
                "count",
            ]
            print(f"Generating plot for {typ} …")
            subprocess.run(cmd, check=True)
    except Exception as e:
        print("Plot generation failed:", e)

if __name__ == "__main__":
    main()
