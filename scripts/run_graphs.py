"""Execute the RAG and VLM graphs independently."""

import argparse
from pathlib import Path

from seismic_damage.config.settings import Settings
from seismic_damage.pipelines.rag.rag_graph import run_rag_graph
from seismic_damage.pipelines.vlm.vlm_graph import run_vlm_graph

def main():
    parser = argparse.ArgumentParser(description="Run independent RAG and VLM execution graphs.")
    parser.add_argument(
        "--modality",
        choices=["all", "rag", "vlm"],
        default="all",
        help="Which pipeline to run"
    )
    args = parser.parse_args()

    settings = Settings()
    
    # RAG Graph
    if args.modality in ("all", "rag"):
        print("Starting RAG graph execution...")
        output_path = Path("data/processed/results/rag_graph_output.jsonl")
        rag_res = run_rag_graph(output_path=output_path, settings=settings)
        print(f"RAG complete: processed {len(rag_res.records)} buildings in {rag_res.elapsed_seconds:.2f}s.")
        print(f"Results saved to {output_path}")
        
    # VLM Graph
    if args.modality in ("all", "vlm"):
        print("Starting VLM graph execution...")
        output_path = Path("data/processed/results/vlm_graph_output.jsonl")
        vlm_res = run_vlm_graph(output_path=output_path, settings=settings)
        print(f"VLM complete: processed {len(vlm_res.records)} buildings in {vlm_res.elapsed_seconds:.2f}s.")
        print(f"Results saved to {output_path}")

if __name__ == "__main__":
    main()
