"""Evidence-to-building linking schemas."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class BuildingRecord(BaseModel):
    """Canonical building identity minted from explicit source grouping."""

    building_id: str
    image_ids: list[str] = Field(default_factory=list)
    document_ids: list[str] = Field(default_factory=list)
    aliases: list[str] = Field(default_factory=list)
    caption: str | None = None
    source_collection: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class EvidenceLink(BaseModel):
    """A single evidence item attached to a building_id or left unresolved."""

    evidence_id: str
    evidence_type: str
    building_id: str | None = None
    link_method: str | None = None
    reason: str
    payload: dict[str, Any] = Field(default_factory=dict)

    @property
    def resolved(self) -> bool:
        return self.building_id is not None
