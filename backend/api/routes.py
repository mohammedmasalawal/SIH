"""HTTP endpoints -- thin wrappers only. All business logic lives in classifier.py."""

from __future__ import annotations

from fastapi import APIRouter

from backend.classification.classifier import classify_hotspot, model_is_available
from backend.models.schemas import ClassifyRequest, ClassifyResponse, HealthResponse

router = APIRouter(prefix="/api")


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(model_loaded=model_is_available())


@router.post("/classify", response_model=ClassifyResponse)
def classify(request: ClassifyRequest) -> ClassifyResponse:
    result = classify_hotspot(request.model_dump())
    return ClassifyResponse(
        label=result.label,
        label_source=result.label_source,
        probabilities=result.probabilities,
        model_version=result.model_version,
    )
