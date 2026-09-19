"""CLI script to run ground-truth error analysis comparing RAG and VLM outputs."""

import argparse
from pathlib import Path
import sys

from seismic_damage.linking.ground_truth import build_ground_truth_dataset
from seismic_damage.validation.error_analysis import run_error_analysis
from seismic_damage.io_utils import read_jsonl


def main():
    parser = argparse.ArgumentParser(description="Run ground-truth error analysis.")
    parser.add_argument(
        "--rag-path",
        type=Path,
        default=Path("data/processed/results/rag_graph_output.jsonl"),
        help="Path to RAG graph JSONL output"
    )
    parser.add_argument(
        "--vlm-path",
        type=Path,
        default=Path("data/processed/results/vlm_graph_output.jsonl"),
        help="Path to VLM graph JSONL output"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed/results/"),
        help="Directory to save analysis report"
    )
    args = parser.parse_args()

    if not args.rag_path.exists():
        print(f"Error: RAG graph output not found at {args.rag_path}")
        print("Please run 'python scripts/run_graphs.py --modality rag' first.")
        sys.exit(1)

    if not args.vlm_path.exists():
        print(f"Error: VLM graph output not found at {args.vlm_path}")
        print("Please run 'python scripts/run_graphs.py --modality vlm' first.")
        sys.exit(1)

    print("Loading Ground Truth dataset...")
    # build_ground_truth_dataset() writes its own csv/jsonl by default and returns (records, csv, jsonl)
    gt_records, _, _ = build_ground_truth_dataset()
    print(f"Loaded {len(gt_records)} ground truth records.")

    print(f"Loading RAG outputs from {args.rag_path}...")
    rag_records = read_jsonl(args.rag_path)
    
    print(f"Loading VLM outputs from {args.vlm_path}...")
    vlm_records = read_jsonl(args.vlm_path)

    print("Running error analysis...")
    report = run_error_analysis(
        gt_records=gt_records,
        rag_records=rag_records,
        vlm_records=vlm_records,
        output_dir=args.output_dir
    )

    print(f"Analysis complete for {report['n_buildings_evaluated']} buildings.")
    print(f"Reports saved to {args.output_dir}")

if __name__ == "__main__":
    main()
