"""Step 18: Integrated Seismic Risk Framework package.

Composes separate layers: Hazard, Exposure, Vulnerability, and Site Amplification.
"""

from seismic_damage.risk.assessment import (
    IntegratedRiskRecord,
    assess_all_buildings_risk,
    assess_building_risk,
)
from seismic_damage.risk.exposure import (
    BuildingExposureRecord,
    OccupancyCategory,
    build_exposure_record,
)
from seismic_damage.risk.hazard import SeismicHazardRecord
from seismic_damage.risk.reporting import (
    RISK_DIR,
    export_risk_pipeline_artifacts,
)
from seismic_damage.risk.site_effects import (
    SiteEffectsRecord,
    calculate_site_effects,
)

__all__ = [
    "BuildingExposureRecord",
    "IntegratedRiskRecord",
    "OccupancyCategory",
    "RISK_DIR",
    "SeismicHazardRecord",
    "SiteEffectsRecord",
    "assess_all_buildings_risk",
    "assess_building_risk",
    "build_exposure_record",
    "calculate_site_effects",
    "export_risk_pipeline_artifacts",
]
