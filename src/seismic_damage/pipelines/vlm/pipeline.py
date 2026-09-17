"""Independent vision-language model pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from seismic_damage.config import Settings, get_settings
from seismic_damage.pipelines.vlm.backends import (
    MockVLM,
    OpenAICompatibleVLM,
    UnavailableVLM,
    VLMBackend,
)
from seismic_damage.schemas.pipeline import VLMObservation, VLMResult
from seismic_damage.schemas.vlm import StructuredVLMObservation


class VLMPipeline:
    """Visual inspection pipeline for post-earthquake imagery.

    Designed to run independently of the RAG pipeline.
    """

    def __init__(
        self,
        settings: Settings | None = None,
        backend: VLMBackend | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.backend = backend if backend is not None else self._default_backend()

    def _default_backend(self) -> VLMBackend:
        provider = (self.settings.vlm.provider or "").strip().lower()
        if provider == "mock":
            return MockVLM()
        api_key = self.settings.vlm.api_key
        if provider in {"openai", "openai_compatible"} and api_key:
            return OpenAICompatibleVLM(
                api_key=api_key,
                model=self.settings.vlm.model,
                base_url=self.settings.vlm.base_url,
                max_tokens=self.settings.vlm.max_tokens,
                temperature=self.settings.vlm.temperature,
            )
        if not api_key:
            return UnavailableVLM(
                "No VLM API key configured. MockVLM is reserved for tests; "
                "set VLM_API_KEY or pass an explicit backend."
            )
        return UnavailableVLM(f"Unsupported VLM provider: {provider}")

    def _validate_image(self, path: Path) -> bool:
        suffix = path.suffix.lstrip(".").lower()
        return suffix in {fmt.lower() for fmt in self.settings.vlm.supported_formats}

    def analyze_image(
        self,
        image_path: Path | str,
        *,
        caption: str | None = None,
        extra: dict[str, Any] | None = None,
    ) -> VLMObservation:
        """Analyze a single image and return structured visual findings."""

        path = Path(image_path)
        if not self._validate_image(path):
            return VLMObservation(
                image_path=str(path),
                caption=caption,
                damage=None,
                inferred_attributes={"error": "unsupported_format"},
                confidence=0.0,
            )
        if not path.exists():
            return VLMObservation(
                image_path=str(path),
                caption=caption,
                damage=None,
                inferred_attributes={"error": "missing_file"},
                confidence=0.0,
            )
        structured: StructuredVLMObservation = self.backend.analyze(
            path, caption=caption, extra=extra
        )
        attributes = structured.as_inferred_attributes()
        attributes["backend"] = getattr(self.backend, "name", type(self.backend).__name__)
        if extra:
            attributes.update({f"source_{key}": value for key, value in extra.items()})
        return VLMObservation(
            image_path=str(path),
            caption=caption,
            damage=None,
            inferred_attributes=attributes,
            confidence=structured.confidence,
        )

    def run(
        self,
        image_paths: list[Path | str],
        *,
        captions: list[str | None] | None = None,
        extras: list[dict[str, Any] | None] | None = None,
    ) -> VLMResult:
        """Public entry point for the VLM pipeline."""

        if not self.settings.vlm.enabled:
            return VLMResult(summary="VLM disabled", extracted_hints={"status": "disabled"})
        observations: list[VLMObservation] = []
        for index, path in enumerate(image_paths):
            caption = captions[index] if captions and index < len(captions) else None
            extra = extras[index] if extras and index < len(extras) else None
            observations.append(self.analyze_image(path, caption=caption, extra=extra))
        backend_name = getattr(self.backend, "name", type(self.backend).__name__)
        return VLMResult(
            observations=observations,
            summary=f"{len(observations)} images analyzed with {backend_name}",
            extracted_hints={
                "n_images": len(observations),
                "backend": backend_name,
            },
        )


def run_vlm(
    image_paths: list[Path | str],
    settings: Settings | None = None,
    backend: VLMBackend | None = None,
) -> VLMResult:
    """Functional entry point for the independent VLM pipeline."""

    return VLMPipeline(settings=settings, backend=backend).run(image_paths)
