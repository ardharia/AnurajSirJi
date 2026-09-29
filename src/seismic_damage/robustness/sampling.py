"""Deterministic image sampling for VLM robustness experiments.

Rules:
- Sampling is always deterministic (fixed seed, sorted input).
- Missing evidence is represented as None, never as zero.
- Building identity is preserved throughout.
"""

from __future__ import annotations

import random
from typing import Any


RANDOM_SEED = 42

# Availability levels: fraction of images to keep per building.
# 1.0 = all images; 0.0 = no images (pure missing).
AVAILABILITY_LEVELS: dict[str, float] = {
    "A_100pct": 1.0,
    "B_75pct": 0.75,
    "C_50pct": 0.50,
    "D_25pct": 0.25,
    "G_0pct": 0.0,
}


def deterministic_sample(
    image_ids: list[str],
    fraction: float,
    *,
    seed: int = RANDOM_SEED,
) -> list[str]:
    """Return a deterministic subset of image_ids at a given availability fraction.

    Args:
        image_ids: Sorted list of image identifiers for a building.
        fraction: Fraction in [0.0, 1.0] of images to keep.
        seed: Random seed for reproducibility.

    Returns:
        Subset of image_ids, deterministically selected.
    """
    if fraction <= 0.0:
        return []
    if fraction >= 1.0:
        return sorted(image_ids)

    total = len(image_ids)
    n_keep = max(1, round(total * fraction))
    sorted_ids = sorted(image_ids)
    rng = random.Random(seed)
    # Shuffle a copy; deterministic because both sort and seed are fixed.
    shuffled = sorted_ids[:]
    rng.shuffle(shuffled)
    return sorted(shuffled[:n_keep])


def build_availability_sets(
    building_image_map: dict[str, list[str]],
    fraction: float,
    *,
    seed: int = RANDOM_SEED,
) -> dict[str, list[str]]:
    """Apply deterministic sampling to all buildings at a given fraction.

    Args:
        building_image_map: {building_id -> [image_ids]}.
        fraction: Availability fraction in [0.0, 1.0].
        seed: Random seed.

    Returns:
        {building_id -> sampled_image_ids}.
    """
    return {
        bid: deterministic_sample(images, fraction, seed=seed)
        for bid, images in sorted(building_image_map.items())
    }


def describe_availability_set(
    available: dict[str, list[str]],
    total_map: dict[str, list[str]],
) -> list[dict[str, Any]]:
    """Build per-building availability metadata rows.

    Returns:
        List of dicts with building_id, image_count_total, image_count_available,
        availability_fraction.
    """
    rows: list[dict[str, Any]] = []
    for bid in sorted(available):
        total = len(total_map.get(bid, []))
        available_n = len(available[bid])
        fraction = available_n / total if total > 0 else 0.0
        rows.append(
            {
                "building_id": bid,
                "image_count_total": total,
                "image_count_available": available_n,
                "availability_fraction": round(fraction, 4),
            }
        )
    return rows
