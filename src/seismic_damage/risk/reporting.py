"""Step 18: Risk reporting and artifact generation.

Produces:
- risk_assessment.csv
- hazard_summary.csv
- vulnerability_summary.csv
- risk_manifest.json
"""

from __future__ import annotations

import csv
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from seismic_damage.config.settings import PROJECT_ROOT
from seismic_damage.io_utils import ensure_parent, write_json
from seismic_damage.risk.assessment import IntegratedRiskRecord, assess_all_buildings_risk
from seismic_damage.risk.hazard import SeismicHazardRecord

RISK_DIR = PROJECT_ROOT / "data" / "processed" / "risk"


def export_risk_pipeline_artifacts(
    risk_records: list[IntegratedRiskRecord],
    hazard_record: SeismicHazardRecord,
    *,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    """Write all Step 18 outputs into data/processed/risk/."""
    out = output_dir or RISK_DIR
    out.mkdir(parents=True, exist_ok=True)

    # 1. risk_assessment.csv
    risk_rows = [r.to_dict() for r in risk_records]
    risk_csv = out / "risk_assessment.csv"
    if risk_rows:
        with risk_csv.open("w", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(risk_rows[0].keys()))
            writer.writeheader()
            writer.writerows(risk_rows)

    # 2. hazard_summary.csv
    hazard_csv = out / "hazard_summary.csv"
    with hazard_csv.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(hazard_record.to_dict().keys()))
        writer.writeheader()
        writer.writerow(hazard_record.to_dict())

    # 3. vulnerability_summary.csv
    vuln_rows = [r.vulnerability.to_dict() for r in risk_records]
    vuln_csv = out / "vulnerability_summary.csv"
    if vuln_rows:
        flat_vuln = [
            {
                "building_id": v["building_id"],
                "material_type": v["typology"]["material_type"],
                "structural_system": v["typology"]["structural_system"],
                "number_of_stories": v["typology"]["number_of_stories"],
                "soft_story_risk": v["vulnerability_indicators"]["soft_story_risk"],
                "plan_irregularity": v["vulnerability_indicators"]["plan_irregularity"],
                "soil_type": v["vulnerability_indicators"]["soil_type"],
                "vulnerability_class": v["vulnerability_class"],
                "vulnerability_index": v["vulnerability_index"],
                "rvs_final_score": v["rvs_final_score"],
                "rvs_high_vulnerability": v["rvs_high_vulnerability"],
                "confidence": v["confidence"],
            }
            for v in vuln_rows
        ]
        with vuln_csv.open("w", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(flat_vuln[0].keys()))
            writer.writeheader()
            writer.writerows(flat_vuln)

    # 4. Manifest
    manifest = {
        "step": 18,
        "description": "Step 18 Integrated Seismic Risk Framework",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "n_buildings_evaluated": len(risk_records),
        "hazard_demand": hazard_record.to_dict(),
        "outputs": {
            "risk_assessment": str(risk_csv),
            "hazard_summary": str(hazard_csv),
            "vulnerability_summary": str(vuln_csv),
        },
    }
    write_json(out / "risk_manifest.json", manifest)

    return {
        "manifest": manifest,
        "n_buildings": len(risk_records),
        "outputs": manifest["outputs"],
    }
