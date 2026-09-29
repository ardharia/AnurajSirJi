"""Step 19: Heritage Taxonomy Mapping and Standard Crosswalks.

Maps heritage structures from local survey descriptions into the canonical
heritage typology schema.
"""

from __future__ import annotations

from typing import Any
from seismic_damage.heritage.schemas import HeritageTypology


HERITAGE_KEYWORD_MAP: list[tuple[list[str], HeritageTypology]] = [
    (["bhunga", "bunga"], HeritageTypology.TRADITIONAL_BHUNGA),
    (["temple", "mandir", "derasar", "shikhara", "chhatri"], HeritageTypology.ASHLAR_STONE_TEMPLE),
    (["fort", "bastion", "qila", "fortress", "citadel"], HeritageTypology.RUBBLE_LIME_FORT_WALL),
    (["haveli", "wood frame", "timber frame", "dhajji"], HeritageTypology.TIMBER_LACED_MASONRY),
    (["palace", "darbargadh", "court", "colonial"], HeritageTypology.UNREINFORCED_LIME_MASONRY),
]


def map_text_to_heritage_typology(text: str) -> HeritageTypology:
    """Classify free-form heritage description into structured typology."""
    t_lower = text.lower()
    for keywords, typ in HERITAGE_KEYWORD_MAP:
        if any(k in t_lower for k in keywords):
            return typ
    return HeritageTypology.OTHER_HERITAGE
