"""CLI script to run feature importance analysis on the building registry."""

import argparse
from pathlib import Path
import sys

from seismic_damage.linking.ground_truth import build_ground_truth_dataset
from seismic_damage.stats.feature_importance import run_vulnerability_analysis


def main():
    parser = argparse.ArgumentParser(description="Run feature importance analysis to identify critical damage parameters.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed/statistics/"),
        help="Directory to save analysis report"
    )
    args = parser.parse_args()

    print("Loading Ground Truth dataset for analysis...")
    gt_records, _, _ = build_ground_truth_dataset()
    
    if not gt_records:
        print("Error: Ground truth dataset is empty.")
        sys.exit(1)

    print(f"Loaded {len(gt_records)} buildings. Running Random Forest feature importance...")
    
    report = run_vulnerability_analysis(
        gt_records=gt_records,
        output_dir=args.output_dir
    )

    if report.get("feature_importances"):
        print("\nTop 10 parameters most strongly dictating structural damage (damage_grade):")
        print("-" * 60)
        for i, f in enumerate(report["feature_importances"][:10], 1):
            print(f"{i:2d}. {f['parameter']:<30} {f['importance_score']:.4f}")
        print("-" * 60)
        print(f"\nAnalysis complete. Results saved to {args.output_dir}")
    else:
        print("Feature importance analysis could not be completed (likely missing scikit-learn).")

if __name__ == "__main__":
    main()
