"""HTTP endpoints -- thin wrappers only. All business logic lives in classifier.py."""

from __future__ import annotations

import numpy as np
import pandas as pd
from fastapi import APIRouter

from backend.classification.classifier import classify_hotspot, model_is_available
from backend.classification.rules import CLASSES
from backend.config import CLASS_COLORS, DEMO_HOTSPOTS_PATH
from backend.models.schemas import (
    ClassesResponse,
    ClassifyRequest,
    ClassifyResponse,
    HealthResponse,
    HotspotRecord,
    HotspotsResponse,
)

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


@router.get("/classes", response_model=ClassesResponse)
def classes() -> ClassesResponse:
    return ClassesResponse(classes=list(CLASSES), colors=CLASS_COLORS)


@router.get("/hotspots", response_model=HotspotsResponse)
def hotspots() -> HotspotsResponse:
    """Demo hotspots for the frontend map -- data/sample/demo_hotspots.csv, not a real FIRMS pull.

    Swap this for a real labeled_hotspots.csv (or a live query) once one exists;
    the response shape is unchanged either way.
    """
    frame = pd.read_csv(DEMO_HOTSPOTS_PATH).replace({np.nan: None})
    records = [HotspotRecord(**row) for row in frame.to_dict(orient="records")]
    return HotspotsResponse(hotspots=records, source="demo")
