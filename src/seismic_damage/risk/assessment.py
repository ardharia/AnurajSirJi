"""Step 18: Integrated Seismic Risk Framework Engine.

Formulation:
  Risk = Hazard (Demand) × Site Amplification × Building Vulnerability × Exposure Consequence

Guarantees:
- Hazard, Exposure, Vulnerability, and Site Effects are tracked as separate traceable layers.
- Transparent relative/qualitative risk indices without fictitious monetary loss fabrication.
"""

from __future__ import annotations

from typing import Any, Mapping
from pydantic import BaseModel, Field

from seismic_damage.risk.exposure import BuildingExposureRecord, build_exposure_record
from seismic_damage.risk.hazard import SeismicHazardRecord
from seismic_damage.risk.site_effects import SiteEffectsRecord, calculate_site_effects
from seismic_damage.vulnerability.assessment import assess_building_vulnerability
from seismic_damage.vulnerability.schemas import BuildingVulnerabilityRecord


class IntegratedRiskRecord(BaseModel):
    """Integrated seismic risk assessment for a single building asset."""

    building_id: str
    hazard: SeismicHazardRecord
    site_effects: SiteEffectsRecord
    exposure: BuildingExposureRecord
    vulnerability: BuildingVulnerabilityRecord

    effective_ground_motion_pga_g: float = Field(
        ..., description="Site-amplified effective PGA demand in g"
    )
    relative_risk_score: float = Field(
        ..., ge=0.0, description="Normalized relative risk metric"
    )
    relative_risk_category: str = Field(
        ..., description="Low | Moderate | High | Very_High | Severe"
    )
    recommended_action: str = Field(
        ..., description="Prioritized mitigation / detailed engineering assessment guidance"
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "building_id": self.building_id,
            "earthquake_event": self.hazard.earthquake_event,
            "seismic_zone": self.hazard.seismic_zone,
            "regional_pga_g": self.hazard.regional_pga_g,
            "soil_type": self.site_effects.soil_type,
            "soil_amplification_factor": self.site_effects.soil_amplification_factor,
            "effective_pga_g": round(self.effective_ground_motion_pga_g, 3),
            "occupancy": self.exposure.occupancy.value,
            "exposure_weight": self.exposure.relative_exposure_weight,
            "typology_material": self.vulnerability.typology.material_type,
            "typology_system": self.vulnerability.typology.structural_system,
            "vulnerability_class": self.vulnerability.vulnerability_class.value,
            "vulnerability_index": self.vulnerability.vulnerability_index,
            "rvs_final_score": self.vulnerability.rvs_final_score,
            "rvs_high_vulnerability": self.vulnerability.rvs_high_vulnerability,
            "relative_risk_score": round(self.relative_risk_score, 4),
            "relative_risk_category": self.relative_risk_category,
            "recommended_action": self.recommended_action,
        }


def assess_building_risk(
    building_id: str,
    building_record: Mapping[str, Any],
    hazard_record: SeismicHazardRecord | None = None,
) -> IntegratedRiskRecord:
    """Compute integrated seismic risk from separate hazard, site, vulnerability, and exposure."""
    hazard = hazard_record or SeismicHazardRecord()

    # 1. Site effects
    soil_type_input = building_record.get("soil_type")
    if not soil_type_input and isinstance(building_record.get("vulnerability"), dict):
        soil_type_input = building_record["vulnerability"].get("soil_type")
    site = calculate_site_effects(building_id, soil_type_input)

    # 2. Building Exposure
    occ_input = building_record.get("occupancy")
    if not occ_input and isinstance(building_record.get("typology"), dict):
        occ_input = building_record["typology"].get("occupancy")
    stories = building_record.get("number_of_stories")
    exposure = build_exposure_record(building_id, occ_input, stories)

    # 3. Building Vulnerability (Step 17)
    vuln = assess_building_vulnerability(building_id, building_record)

    # 4. Integrated Risk Computation
    reg_pga = hazard.regional_pga_g or (hazard.zone_factor_z * 1.0)
    effective_pga = reg_pga * site.soil_amplification_factor * site.topographic_factor

    v_idx = vuln.vulnerability_index if vuln.vulnerability_index is not None else 0.5
    # Risk = Effective Demand × Vulnerability × Consequence
    raw_risk = effective_pga * v_idx * exposure.relative_exposure_weight

    # Risk categorization
    if raw_risk < 0.25:
        category = "Low"
        action = "Standard periodic maintenance; baseline compliance"
    elif raw_risk < 0.50:
        category = "Moderate"
        action = "Visual inspection; routine non-structural retrofit"
    elif raw_risk < 0.80:
        category = "High"
        action = "Detailed structural evaluation (IS 15988); seismic retrofit priority"
    elif raw_risk < 1.10:
        category = "Very_High"
        action = "Immediate structural retrofitting / jacketing; occupancy restriction review"
    else:
        category = "Severe"
        action = "Critical intervention required; mandatory evacuation or urgent structural overhaul"

    return IntegratedRiskRecord(
        building_id=building_id,
        hazard=hazard,
        site_effects=site,
        exposure=exposure,
        vulnerability=vuln,
        effective_ground_motion_pga_g=effective_pga,
        relative_risk_score=raw_risk,
        relative_risk_category=category,
        recommended_action=action,
    )


def assess_all_buildings_risk(
    building_records: list[Mapping[str, Any]],
    hazard_record: SeismicHazardRecord | None = None,
) -> list[IntegratedRiskRecord]:
    """Execute integrated risk framework across all building records."""
    hazard = hazard_record or SeismicHazardRecord()
    return [
        assess_building_risk(str(r.get("building_id") or f"bldg_{i}"), r, hazard)
        for i, r in enumerate(building_records)
    ]
