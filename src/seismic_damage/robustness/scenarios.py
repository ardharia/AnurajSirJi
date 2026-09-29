"""Step 15: Scenario definitions for VLM robustness analysis.

Scenarios:
  A: 100% image availability (all available images)
  B: 75% image availability (deterministic subset)
  C: 50% image availability (deterministic subset)
  D: 25% image availability (deterministic subset)
  E: Severely degraded / pixelated images
  F: Poor-quality / low-information images (heavy blur + contrast loss)
  G: No usable VLM evidence (0% availability)
  H: Random image removal (at least 1 image removed where count > 1)
  I: Removal of most informative / primary image (index 0)
  J: Single-image-per-building condition (only index 0 retained)

Rules:
- Operations are non-destructive (original files/metadata remain untouched).
- Sampling is deterministic with fixed random seeds.
- Missing evidence is represented as None, never 0.
"""

from __future__ import annotations

import random
from typing import Any, Mapping

RANDOM_SEED = 42

SCENARIO_DESCRIPTIONS: dict[str, str] = {
    "A_100pct": "100% image availability — full baseline visual evidence",
    "B_75pct": "75% image availability — deterministic 75% subset",
    "C_50pct": "50% image availability — deterministic 50% subset",
    "D_25pct": "25% image availability — deterministic 25% subset",
    "E_pixelated": "Severely degraded / pixelated images (downscale-upscale nearest)",
    "F_low_quality": "Poor quality / heavy blur and contrast reduction",
    "G_0pct": "Zero usable visual evidence (VLM unavailable)",
    "H_random_removal": "Random image removal (drop 1 random image for multi-image buildings)",
    "I_remove_primary": "Removal of primary/most informative image (drop index 0)",
    "J_single_image": "Single image per building (retain only index 0)",
}


def sample_for_scenario(
    image_ids: list[str],
    scenario: str,
    *,
    seed: int = RANDOM_SEED,
) -> list[str]:
    """Return the list of image IDs to keep for a building under a given scenario."""
    if not image_ids:
        return []

    sorted_ids = sorted(image_ids)
    n = len(sorted_ids)

    if scenario == "A_100pct":
        return sorted_ids

    if scenario == "B_75pct":
        k = max(1, round(n * 0.75))
        rng = random.Random(seed)
        shuffled = sorted_ids[:]
        rng.shuffle(shuffled)
        return sorted(shuffled[:k])

    if scenario == "C_50pct":
        k = max(1, round(n * 0.50))
        rng = random.Random(seed)
        shuffled = sorted_ids[:]
        rng.shuffle(shuffled)
        return sorted(shuffled[:k])

    if scenario == "D_25pct":
        k = max(1, round(n * 0.25))
        rng = random.Random(seed)
        shuffled = sorted_ids[:]
        rng.shuffle(shuffled)
        return sorted(shuffled[:k])

    if scenario in ("E_pixelated", "F_low_quality"):
        # Image count is 100%, but visual quality is degraded
        return sorted_ids

    if scenario == "G_0pct":
        return []

    if scenario == "H_random_removal":
        if n <= 1:
            return sorted_ids
        rng = random.Random(seed)
        dropped_idx = rng.randint(0, n - 1)
        return [img for idx, img in enumerate(sorted_ids) if idx != dropped_idx]

    if scenario == "I_remove_primary":
        if n <= 1:
            return []  # Only 1 image, removing primary leaves 0
        return sorted_ids[1:]

    if scenario == "J_single_image":
        return [sorted_ids[0]]

    raise ValueError(f"Unknown scenario: {scenario}")


def build_scenario_image_map(
    building_image_map: dict[str, list[str]],
    scenario: str,
    *,
    seed: int = RANDOM_SEED,
) -> dict[str, list[str]]:
    """Generate the {building_id -> [image_ids]} mapping under a specified scenario."""
    return {
        bid: sample_for_scenario(imgs, scenario, seed=seed)
        for bid, imgs in sorted(building_image_map.items())
    }
