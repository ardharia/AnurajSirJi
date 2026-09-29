"""Step 19: Heritage-specific vulnerability parameters and scoring tables.

References:
- IS 13935:2009 / IS 13828:1993 (Improving Earthquake Resistance of Low Strength Masonry)
- UNESCO / ICOMOS Guidelines for Seismic Retrofitting of Built Cultural Heritage
- INTACH Architectural Heritage Conservation Charter
- Archaeological Survey of India (ASI) Post-Bhuj Reconstruction Reports
"""

from __future__ import annotations

# Base Heritage Vulnerability Scores (0.0 = highly resilient vernacular like Bhunga, 1.0 = highly vulnerable unreinforced monument)
HERITAGE_TYPOLOGY_BASE_VULNERABILITY: dict[str, float] = {
    "traditional_bhunga": 0.25,             # Circular aerodynamic shape, lightweight thatch/wood roof (very resilient)
    "timber_laced_masonry": 0.40,           # Wood ring-beams / dhajji-dewari style energy absorption
    "ashlar_stone_temple": 0.75,            # Heavy corbelled domes, dry-stone interlocking joints susceptible to sliding
    "rubble_lime_fort_wall": 0.70,          # Multi-leaf delamination under ground shaking
    "unreinforced_lime_masonry": 0.65,      # High mass, weak lime mortar tensile capacity
    "heritage_monument_complex": 0.80,      # Complex asymmetric geometry and towering shikharas
    "other_heritage": 0.60,
}

# Modifiers for heritage decay and architectural vulnerabilities
HERITAGE_MODIFIERS: dict[str, float] = {
    "multi_leaf_delamination": +0.15,
    "unrestrained_arch_thrust": +0.12,
    "heavy_dome_pounding": +0.10,
    "timber_rot_or_decay": +0.15,
    "mortar_leaching": +0.10,
    "out_of_plumb_walls": +0.12,
    "incompatible_modern_concrete": +0.18,  # Heavy rigid RCC slab added on weak historical stone wall
    "previous_unrepaired_cracking": +0.15,
    "good_maintenance_lime_repointing": -0.10,
}
