"""Canonical building identity linking."""

from seismic_damage.linking.catalog import build_catalog, load_image_rows
from seismic_damage.linking.ground_truth import build_ground_truth_dataset, export_building_links
from seismic_damage.linking.linker import link_all, link_chunk, link_image_row, link_vlm_observation
from seismic_damage.linking.registry import build_building_registry, classify_typology

__all__ = [
    "build_building_registry",
    "build_catalog",
    "build_ground_truth_dataset",
    "classify_typology",
    "export_building_links",
    "link_all",
    "link_chunk",
    "link_image_row",
    "link_vlm_observation",
    "load_image_rows",
]

