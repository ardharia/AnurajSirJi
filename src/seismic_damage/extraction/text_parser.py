"""Conservative, rule-based parameter parsing from expert captions/report text."""

from __future__ import annotations

import re
from typing import Any

_STOREYS = re.compile(r"(\d+)\s*[- ]?\s*(?:storey|story|storied)\b", re.I)
_RC = re.compile(r"\b(?:reinforced concrete|\brc\b|r\.c\.)\b", re.I)


def parse_damage_text(text: str | None) -> dict[str, Any]:
    """Extract only fields supported by explicit wording. Unstated fields stay unset."""

    if not text or not text.strip():
        return {}
    original = text.strip()
    lowered = original.lower()
    out: dict[str, Any] = {"evidence_text": original}

    storeys = _STOREYS.search(original)
    if storeys:
        value = int(storeys.group(1))
        if value >= 1:
            out["number_of_stories"] = value

    if _RC.search(original) or "rc frame" in lowered:
        out["material_type"] = "reinforced_concrete"
    elif "brick masonry" in lowered or "masonry" in lowered:
        out["material_type"] = "masonry"
    elif "steel" in lowered:
        out["material_type"] = "steel"

    if "rc frame" in lowered or "reinforced concrete frame" in lowered:
        out["structural_system"] = "rc_frame"
    elif "shear wall" in lowered:
        out["structural_system"] = "shear_wall"
    elif "bearing wall" in lowered or "load bearing" in lowered:
        out["structural_system"] = "bearing_wall"
    elif "moment frame" in lowered:
        out["structural_system"] = "moment_frame"
    elif "precast" in lowered:
        out["structural_system"] = "precast"

    if any(term in lowered for term in ("pancake", "total collapse", "flattened", "reduced to rubble", "razed")):
        out["damage_grade"] = 5
        out["collapse_mode"] = "total"
    elif "partial collapse" in lowered or "collapse of a portion" in lowered or "collapse of one-half" in lowered:
        out["damage_grade"] = 4
        out["collapse_mode"] = "partial"
    elif "ground storey collapse" in lowered or "open ground storey" in lowered and "collapse" in lowered:
        out["damage_grade"] = 4
        out["collapse_mode"] = "soft_story"
        out["soft_story_failure"] = True
    elif "collapse" in lowered:
        out["damage_grade"] = 4
        out.setdefault("collapse_mode", "partial")
    elif "severe shear" in lowered or "severe" in lowered and "crack" in lowered:
        out["damage_grade"] = 3
        out["crack_severity"] = "severe"
    elif "minor cracking" in lowered or "minor crack" in lowered:
        out["damage_grade"] = 1
        out["crack_severity"] = "hairline"
    elif "crack" in lowered:
        out["damage_grade"] = 2
        out["crack_severity"] = "moderate"

    if "open ground storey" in lowered or "soft storey" in lowered or "soft story" in lowered:
        out["soft_story_failure"] = True
        out.setdefault("collapse_mode", "soft_story")

    if "shear crack" in lowered or "shear cracking" in lowered:
        out["crack_pattern"] = "shear"
    elif "x-shaped" in lowered or "x shaped" in lowered:
        out["crack_pattern"] = "x_shaped"
    elif "diagonal" in lowered:
        out["crack_pattern"] = "diagonal"
    elif "flexural" in lowered:
        out["crack_pattern"] = "flexural"

    if "wall" in lowered and any(term in lowered for term in ("fail", "collapse", "crack", "infill")):
        out["wall_failure"] = True

    filled = [key for key, value in out.items() if key != "evidence_text" and value is not None]
    out["confidence"] = min(0.9, 0.35 + 0.1 * len(filled)) if filled else 0.0
    return out
