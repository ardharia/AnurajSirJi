"""Top-level assessment orchestration (stub)."""

from __future__ import annotations

import uuid

from seismic_damage.config import Settings, get_settings
from seismic_damage.extraction.parameters import extract_parameters
from seismic_damage.pipelines.rag.pipeline import RAGPipeline
from seismic_damage.pipelines.vlm.pipeline import VLMPipeline
from seismic_damage.schemas.pipeline import AssessmentRequest, AssessmentResult
from seismic_damage.validation.validators import validate_extraction


class SeismicDamageAssessor:
    """Coordinates RAG, VLM, extraction, validation, and fragility stages.

    Pipelines remain independently callable; this class only composes them.
    """

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.rag = RAGPipeline(self.settings)
        self.vlm = VLMPipeline(self.settings)

    def assess(self, request: AssessmentRequest) -> AssessmentResult:
        """Run the multimodal assessment workflow (stub composition)."""
        request_id = request.building_id or str(uuid.uuid4())

        rag_result = None
        if request.run_rag and self.settings.rag.enabled and request.text_query:
            rag_result = self.rag.run(request.text_query)

        vlm_result = None
        if request.run_vlm and self.settings.vlm.enabled and request.image_paths:
            vlm_result = self.vlm.run(list(request.image_paths))

        extraction = extract_parameters(
            rag_result=rag_result,
            vlm_result=vlm_result,
            known_parameters=request.known_parameters,
            settings=self.settings,
        )
        report = validate_extraction(extraction, settings=self.settings)

        return AssessmentResult(
            request_id=request_id,
            rag=rag_result,
            vlm=vlm_result,
            extraction=extraction,
            building=report.building,
            damage=None,
            fragility=None,
            validation_passed=report.passed,
            validation_messages=report.messages,
        )


def run_assessment(
    request: AssessmentRequest,
    settings: Settings | None = None,
) -> AssessmentResult:
    """Functional entry point for end-to-end assessment orchestration."""
    return SeismicDamageAssessor(settings=settings).assess(request)
