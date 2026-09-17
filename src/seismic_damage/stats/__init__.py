"""Statistical analysis of building-level damage parameters."""

from seismic_damage.stats.analysis import (
    deduplicate_buildings,
    frequency_tables,
    run_statistical_analysis,
)

__all__ = ["deduplicate_buildings", "frequency_tables", "run_statistical_analysis"]
