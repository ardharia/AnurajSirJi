"""VLM backends. MockVLM is for tests only."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Protocol

from seismic_damage.schemas.vlm import StructuredVLMObservation


class VLMBackend(Protocol):
    name: str

    def analyze(
        self,
        image_path: Path,
        *,
        caption: str | None = None,
        extra: dict[str, Any] | None = None,
    ) -> StructuredVLMObservation:
        ...


class MockVLM:
    """Deterministic backend for unit tests. Does not invent damage grades."""

    name = "mock"

    def __init__(self, fixtures: dict[str, dict[str, Any]] | None = None) -> None:
        self.fixtures = fixtures or {}

    def analyze(
        self,
        image_path: Path,
        *,
        caption: str | None = None,
        extra: dict[str, Any] | None = None,
    ) -> StructuredVLMObservation:
        path = Path(image_path)
        keys = [str(path), path.name, path.stem]
        if extra and extra.get("image_id"):
            keys.insert(0, str(extra["image_id"]))
        for key in keys:
            if key in self.fixtures:
                payload = dict(self.fixtures[key])
                payload.setdefault("evidence", "mock_vlm_fixture")
                return StructuredVLMObservation.model_validate(payload)
        return StructuredVLMObservation(
            confidence=0.0,
            evidence="mock_vlm_no_fixture",
            notes="MockVLM returns empty observations unless a test fixture is provided.",
        )


class OpenAICompatibleVLM:
    """Multimodal chat backend (OpenAI-compatible) for real image analysis."""

    name = "openai"

    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        base_url: str | None = None,
        max_tokens: int = 2048,
        temperature: float = 0.0,
    ) -> None:
        from openai import OpenAI

        kwargs: dict[str, Any] = {"api_key": api_key}
        if base_url:
            kwargs["base_url"] = base_url
        self._client = OpenAI(**kwargs)
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature

    def analyze(
        self,
        image_path: Path,
        *,
        caption: str | None = None,
        extra: dict[str, Any] | None = None,
    ) -> StructuredVLMObservation:
        import base64

        path = Path(image_path)
        encoded = base64.b64encode(path.read_bytes()).decode("ascii")
        suffix = path.suffix.lower().lstrip(".") or "jpeg"
        mime = "jpeg" if suffix in {"jpg", "jpeg"} else suffix
        caption_block = f"Source caption (may be used as context, not as identity): {caption}" if caption else ""
        prompt = (
            "You are assessing post-earthquake building damage from a photograph. "
            "Return a JSON object with keys: damage_grade (0-5 or null), material, "
            "structural_system, number_of_stories, crack_severity, crack_pattern, "
            "collapse_mode, soft_story_failure, wall_failure, confidence (0-1), "
            "evidence, notes. Use null when the image does not support a field. "
            "Do not invent a building identity. "
            f"{caption_block}"
        )
        response = self._client.chat.completions.create(
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            response_format={"type": "json_object"},
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/{mime};base64,{encoded}"},
                        },
                    ],
                }
            ],
        )
        content = response.choices[0].message.content or "{}"
        payload = json.loads(content)
        payload.setdefault("evidence", "vlm_model_json")
        return StructuredVLMObservation.model_validate(payload)


class UnavailableVLM:
    """Used when no real backend is configured. Never fabricates damage labels."""

    name = "unavailable"

    def __init__(self, reason: str) -> None:
        self.reason = reason

    def analyze(
        self,
        image_path: Path,
        *,
        caption: str | None = None,
        extra: dict[str, Any] | None = None,
    ) -> StructuredVLMObservation:
        _ = (caption, extra)
        return StructuredVLMObservation(
            confidence=0.0,
            evidence="vlm_backend_unavailable",
            notes=f"{self.reason} Image: {image_path}",
        )
