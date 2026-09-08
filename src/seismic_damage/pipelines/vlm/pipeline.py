"""Independent vision-language model pipeline (stub)."""

from __future__ import annotations

from pathlib import Path

from seismic_damage.config import Settings, get_settings
from seismic_damage.schemas.pipeline import VLMObservation, VLMResult


class VLMPipeline:
    """Visual inspection pipeline for post-earthquake imagery.

    Designed to run independently of the RAG pipeline.
    """

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()

    def _validate_image(self, path: Path) -> bool:
        suffix = path.suffix.lstrip(".").lower()
        return suffix in {fmt.lower() for fmt in self.settings.vlm.supported_formats}

    def analyze_image(self, image_path: Path | str) -> VLMObservation:
        """Analyze a single image and return structured visual findings."""
        path = Path(image_path)
        if not self._validate_image(path):
            return VLMObservation(
                image_path=str(path),
                caption=None,
                damage=None,
                inferred_attributes={"error": "unsupported_format"},
                confidence=0.0,
            )
        # Placeholder until VLM provider client is wired.
        return VLMObservation(
            image_path=str(path),
            caption=None,
            damage=None,
            inferred_attributes={"status": "not_implemented"},
            confidence=0.0,
        )

    def run(self, image_paths: list[Path | str]) -> VLMResult:
        """Public entry point for the VLM pipeline."""
        if not self.settings.vlm.enabled:
            return VLMResult(summary="VLM disabled", extracted_hints={"status": "disabled"})
        observations = [self.analyze_image(path) for path in image_paths]
        return VLMResult(
            observations=observations,
            summary=None,
            extracted_hints={"status": "not_implemented", "n_images": len(observations)},
        )


def run_vlm(
    image_paths: list[Path | str],
    settings: Settings | None = None,
) -> VLMResult:
    """Functional entry point for the independent VLM pipeline."""
    return VLMPipeline(settings=settings).run(image_paths)
