"""Independent execution graph for the VLM (image) module."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from seismic_damage.config.settings import Settings
from seismic_damage.linking.catalog import build_catalog, load_image_rows, BuildingRecord
from seismic_damage.pipelines.vlm.pipeline import VLMPipeline
from seismic_damage.validation.normalize import normalize_vlm_output
from seismic_damage.validation.cross_validation import records_from_vlm_links
from seismic_damage.schemas.normalized import NormalizedAssessmentRecord
from seismic_damage.io_utils import ensure_parent, write_jsonl


@dataclass
class VLMGraphResult:
    records: list[NormalizedAssessmentRecord]
    metadata: dict[str, Any]
    elapsed_seconds: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "metadata": self.metadata,
            "elapsed_seconds": self.elapsed_seconds,
            "records": [r.to_dict() for r in self.records],
        }

    def to_jsonl(self, path: Path | str) -> None:
        write_jsonl(path, [r.to_dict() for r in self.records])


class VLMGraph:
    """Execution graph for generating independent visual building assessments."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or Settings()
        self.pipeline = VLMPipeline(settings=self.settings)

    def run(
        self,
        catalog: list[BuildingRecord] | None = None,
        image_rows: list[dict[str, Any]] | None = None
    ) -> VLMGraphResult:
        """Execute the VLM graph pipeline."""
        start_t = time.time()
        
        if image_rows is None:
            image_rows = load_image_rows()
        if catalog is None:
            catalog = build_catalog(image_rows)

        # 1. Process all images through VLM
        vlm_links: list[dict[str, Any]] = []
        for row in image_rows:
            iid = row.get("image_id") or row.get("filename")
            caption = row.get("caption") or ""
            filename = row.get("filename", "")
            image_path = f"data/raw/images/{filename}" if filename else ""

            # Find matching building
            bid = None
            for b in catalog:
                if iid and iid in b.image_ids:
                    bid = b.building_id
                    break
            if bid is None and caption:
                for b in catalog:
                    if b.caption and b.caption.strip() == caption.strip():
                        bid = b.building_id
                        break

            # Analyze image using VLM Pipeline
            obs = self.pipeline.analyze_image(
                image_path=image_path,
                caption=caption
            )
            
            # Map observations to payload
            payload = {
                "image_id": iid,
                "image_path": image_path,
                "evidence": caption,
                "source_collection": row.get("source_collection") or row.get("_collection"),
                "confidence": obs.confidence,
            }
            if obs.damage:
                payload["damage_grade"] = obs.damage
            if obs.inferred_attributes:
                payload.update(obs.inferred_attributes)
                
            vlm_links.append({"building_id": bid, "payload": payload})

        # 2. Consolidate image links into building records without voting
        collapsed = records_from_vlm_links(vlm_links)
        by_bid = {str(r["building_id"]): r for r in collapsed if r.get("building_id")}

        records: list[NormalizedAssessmentRecord] = []
        for building in catalog:
            bid = building.building_id
            raw_building = by_bid.get(bid, {"building_id": bid, "evidence_source": "vlm", "confidence": 0.0})
            
            # Add image paths
            raw_building["image_paths"] = [
                f"data/raw/images/{iid}" for iid in building.image_ids
            ]
            
            # 3. Normalize
            norm = normalize_vlm_output(raw_building, building_id=bid)
            records.append(norm)

        elapsed = time.time() - start_t
        
        metadata = {
            "n_buildings": len(catalog),
            "n_records": len(records),
            "n_images_processed": len(image_rows),
            "modality": "vlm",
            "backend": getattr(self.pipeline.backend, "name", type(self.pipeline.backend).__name__)
        }

        return VLMGraphResult(
            records=records,
            metadata=metadata,
            elapsed_seconds=elapsed
        )


def run_vlm_graph(
    output_path: Path | str | None = None,
    settings: Settings | None = None
) -> VLMGraphResult:
    """Functional entry point to execute the VLM graph and persist results."""
    settings = settings or Settings()
    graph = VLMGraph(settings=settings)
    result = graph.run()
    
    if output_path:
        path = ensure_parent(output_path)
        result.to_jsonl(path)
        
    return result
