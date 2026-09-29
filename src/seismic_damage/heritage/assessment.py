"""Step 19: Heritage Vulnerability and Conservation Assessment Engine.

Operates independently from ordinary modern building pipelines to prevent contamination
of standard RC/masonry datasets.
"""

from __future__ import annotations

from typing import Any, Mapping

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


def assess_heritage_monument(
    monument_id: str,
    record: Mapping[str, Any],
) -> HeritageBuildingRecord:
    """Assess vulnerability and conservation priority for a heritage asset."""
    name = str(record.get("monument_name") or record.get("name") or monument_id)
    typology_str = str(record.get("heritage_typology") or "other_heritage").strip().lower()

    # Match typology enum
    try:
        typology = HeritageTypology(typology_str)
    except ValueError:
        typology = HeritageTypology.OTHER_HERITAGE

    # Match masonry bond
    bond_str = str(record.get("masonry_bond") or "unknown").strip().lower()
    try:
        bond = HeritageMasonryBond(bond_str)
    except ValueError:
        bond = HeritageMasonryBond.UNKNOWN

    # Match roof system
    roof_str = str(record.get("roof_vault_system") or "unknown").strip().lower()
    try:
        roof = HeritageRoofVaultSystem(roof_str)
    except ValueError:
        roof = HeritageRoofVaultSystem.UNKNOWN

    # Vulnerability indicators
    attrs_raw = record.get("vulnerability_attributes") if isinstance(record.get("vulnerability_attributes"), dict) else record
    attrs = HeritageVulnerabilityAttributes(
        multi_leaf_wall_delamination_risk=attrs_raw.get("multi_leaf_wall_delamination_risk"),
        unrestrained_arch_thrust=attrs_raw.get("unrestrained_arch_thrust"),
        heavy_dome_pounding=attrs_raw.get("heavy_dome_pounding"),
        timber_rot_or_termite_decay=attrs_raw.get("timber_rot_or_termite_decay"),
        mortar_leaching_or_loss=attrs_raw.get("mortar_leaching_or_loss"),
        out_of_plumb_walls=attrs_raw.get("out_of_plumb_walls"),
        incompatible_modern_interventions=attrs_raw.get("incompatible_modern_interventions"),
        previous_earthquake_cracking=attrs_raw.get("previous_earthquake_cracking"),
    )

    # 1. Base vulnerability from typology
    base_v = HERITAGE_TYPOLOGY_BASE_VULNERABILITY.get(typology.value, 0.60)

    # 2. Modifiers
    mod_sum = 0.0
    guidelines: list[str] = []

    if attrs.multi_leaf_wall_delamination_risk:
        mod_sum += HERITAGE_MODIFIERS["multi_leaf_delamination"]
        guidelines.append("Inject compatible hydraulic lime grout to tie multi-leaf stone masonry.")
    if attrs.unrestrained_arch_thrust:
        mod_sum += HERITAGE_MODIFIERS["unrestrained_arch_thrust"]
        guidelines.append("Install stainless steel / bronze tie-rods across arch springings.")
    if attrs.incompatible_modern_interventions:
        mod_sum += HERITAGE_MODIFIERS["incompatible_modern_concrete"]
        guidelines.append("Deconstruct or decouple rigid RCC additions; restore flexible historical timber diaphragms.")
    if attrs.mortar_leaching_or_loss:
        mod_sum += HERITAGE_MODIFIERS["mortar_leaching"]
        guidelines.append("Rake out decayed joints and deep-repoint with breathable pozzolana-lime mortar.")
    if attrs.timber_rot_or_termite_decay:
        mod_sum += HERITAGE_MODIFIERS["timber_rot_or_decay"]
        guidelines.append("Treat historical timber with breathable biocides; sister rotted joist ends.")

    final_v = max(0.05, min(1.0, base_v + mod_sum))

    # Priority & class
    if final_v < 0.35:
        v_class = "Class_D_Vernacular_Resilient"
        priority = "Low"
    elif final_v < 0.60:
        v_class = "Class_C_Heritage_Moderate"
        priority = "Moderate"
    elif final_v < 0.80:
        v_class = "Class_B_Heritage_High"
        priority = "High"
    else:
        v_class = "Class_A_Heritage_Critical"
        priority = "Critical"

    if not guidelines:
        guidelines.append("Standard periodic heritage monument monitoring and non-destructive inspection.")

    return HeritageBuildingRecord(
        monument_id=monument_id,
        monument_name=name,
        location=str(record.get("location") or "Kutch_Gujarat"),
        heritage_typology=typology,
        masonry_bond=bond,
        roof_vault_system=roof,
        wall_thickness_m=float(record["wall_thickness_m"]) if record.get("wall_thickness_m") else None,
        height_m=float(record["height_m"]) if record.get("height_m") else None,
        number_of_stories=int(record.get("number_of_stories", 1)),
        construction_era=str(record.get("construction_era")) if record.get("construction_era") else None,
        vulnerability_attributes=attrs,
        heritage_vulnerability_index=round(final_v, 4),
        heritage_vulnerability_class=v_class,
        conservation_priority=priority,
        preservation_guidelines=guidelines,
    )
