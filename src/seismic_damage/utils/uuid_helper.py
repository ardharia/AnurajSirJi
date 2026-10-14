# src/seismic_damage/utils/uuid_helper.py
"""Utility functions for deterministic UUID generation.

The framework requires stable IDs for buildings based on source provenance (report
filename, page number) and the raw description text. This helper provides a thin
wrapper around :func:`uuid.uuid5` using the DNS namespace by default.
"""

from __future__ import annotations

import uuid
from pathlib import Path

# Using the DNS namespace UUID as a stable base; callers can provide a custom one.
DEFAULT_NAMESPACE = uuid.NAMESPACE_URL


def deterministic_uuid(name: str, namespace: uuid.UUID = DEFAULT_NAMESPACE) -> str:
    """Return a deterministic UUID string for *name* within *namespace*.

    The function uses :func:`uuid.uuid5` (SHA‑1 based) which guarantees the same
    UUID for the same ``name`` and ``namespace`` across runs. ``name`` should be a
    concise concatenation of the provenance identifiers (e.g. ``f"{filename}:{page}:{text}"``).
    """
    return str(uuid.uuid5(namespace, name))


def build_building_uuid(report_path: Path, page: int, description: str) -> str:
    """Create a UUID for a building based on its source report, page, and description.

    Parameters
    ----------
    report_path: Path
        Absolute path to the PDF report containing the building mention.
    page: int
        Page number where the building is referenced.
    description: str
        Raw excerpt describing the building’s earthquake response.
    """
    # Normalise the description to make the UUID stable for identical text.
    norm = " ".join(description.lower().split())
    name = f"{report_path}:{page}:{norm}"
    return deterministic_uuid(name)
