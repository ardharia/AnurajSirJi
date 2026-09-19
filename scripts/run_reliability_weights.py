"""CLI script to compute dynamic reliability weights for RAG and VLM."""

import argparse
import json
from pathlib import Path
import sys

from seismic_damage.fusion.weights import compute_reliability_weights
from seismic_damage.io_utils import ensure_parent


def main():
    parser = argparse.ArgumentParser(description="Compute RAG/VLM reliability weights.")
    parser.add_argument(
        "--report-path",
        type=Path,
        default=Path("data/processed/results/error_analysis_report.json"),
        help="Path to the error analysis JSON report"
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path("data/processed/models/reliability_weights.json"),
        help="Path to save the reliability weights registry"
    )
    args = parser.parse_args()

    if not args.report_path.exists():
        print(f"Error: Error analysis report not found at {args.report_path}")
        print("Please run 'python scripts/run_error_analysis.py' first.")
        sys.exit(1)

    print(f"Loading error report from {args.report_path}...")
    with open(args.report_path, "r", encoding="utf-8") as f:
        report = json.load(f)

    print("Computing reliability weights...")
    weights = compute_reliability_weights(report)

    ensure_parent(args.output_path)
    with open(args.output_path, "w", encoding="utf-8") as f:
        json.dump(weights, f, indent=2)

    print(f"Reliability weights computed for {len(weights['parameter_weights'])} parameters.")
    print(f"Weights saved to {args.output_path}")

if __name__ == "__main__":
    main()
