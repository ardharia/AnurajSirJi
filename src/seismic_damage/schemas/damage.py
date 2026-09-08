"""Damage observation schemas."""

from __future__ import annotations

from pydantic import BaseModel, Field

from seismic_damage.config.settings import DamageState


class ObservedDamage(BaseModel):
    """Damage cues from imagery or reports."""

    damage_state: DamageState = DamageState.NONE
    description: str | None = None
    affected_components: list[str] = Field(default_factory=list)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
