"""Run the complete NDMA RVS Primer-aligned parameter frequency analysis pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path

from seismic_damage.config.settings import PROJECT_ROOT
from seismic_damage.extraction.frequency_pipeline import (
    format_typology_summary_text,
    run_parameter_frequency_analysis,
)
from seismic_damage.linking.registry import build_building_registry
import sys
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
from extract_ndma_rvs_dictionary import extract_rvs_dictionary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run NDMA RVS Parameter Frequency Pipeline")
    parser.add_argument("--rebuild-dict", action="store_true", help="Re-extract RVS dictionary from PDF")
    args = parser.parse_args()

    print("=" * 60)
    print("NDMA RVS PRIMER PARAMETER EXTRACTION & FREQUENCY PIPELINE")
    print("=" * 60)

    # 1. RVS Parameter Dictionary
    dict_path = PROJECT_ROOT / "data" / "reference" / "rvs_parameter_dictionary.csv"
    if args.rebuild_dict or not dict_path.exists():
        print("\n[Step 1] Extracting NDMA RVS Parameter Dictionary from Primer PDF...")
        extract_rvs_dictionary(output_csv=dict_path)
    else:
        print(f"\n[Step 1] Using existing NDMA RVS Dictionary at {dict_path}")

    # 2. Building Registry
    print("\n[Step 2] Building / Verifying Unique Building Registry...")
    registry = build_building_registry()
    print(f"Registry generated: {len(registry)} buildings recorded.")

    # 3. Frequency Analysis
    print("\n[Step 3] Running Typology-Specific RVS Matching & Frequency Analysis...")
    obs, freqs = run_parameter_frequency_analysis()

    # 4. Final Output Display
    print("\n" + format_typology_summary_text(freqs))


if __name__ == "__main__":
    main()
