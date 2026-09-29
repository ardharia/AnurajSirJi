"""Step 19: Heritage and Ancient Monument Assessment Extension package.

Provides modular assessment for historic and vernacular cultural assets.
"""

from seismic_damage.heritage.assessment import assess_heritage_monument
from seismic_damage.heritage.mapping import (
    HERITAGE_KEYWORD_MAP,
    map_text_to_heritage_typology,
)
from seismic_damage.heritage.parameters import (
    HERITAGE_MODIFIERS,
    HERITAGE_TYPOLOGY_BASE_VULNERABILITY,
)
from seismic_damage.heritage.schemas import (
    HeritageBuildingRecord,
    HeritageMasonryBond,
    HeritageRoofVaultSystem,
    HeritageTypology,
    HeritageVulnerabilityAttributes,
)

__all__ = [
    "HERITAGE_KEYWORD_MAP",
    "HERITAGE_MODIFIERS",
    "HERITAGE_TYPOLOGY_BASE_VULNERABILITY",
    "HeritageBuildingRecord",
    "HeritageMasonryBond",
    "HeritageRoofVaultSystem",
    "HeritageTypology",
    "HeritageVulnerabilityAttributes",
    "assess_heritage_monument",
    "map_text_to_heritage_typology",
]
