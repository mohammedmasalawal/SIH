"""HTTP endpoints -- thin wrappers only. Classification logic lives in classifier.py,
national-map data access in backend/map_store.py."""

from __future__ import annotations

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse

from backend.classification.classifier import classify_hotspot, model_is_available
from backend.classification.rules import CLASSES
from backend.config import CLASS_COLORS, CLASS_COLORS_DARK, DEMO_HOTSPOTS_PATH
from backend.map_store import MapDataUnavailable, map_store
from backend.models.schemas import (
    ClassesResponse,
    ClassifyRequest,
    ClassifyResponse,
    DetectionRecord,
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
    return ClassesResponse(classes=list(CLASSES), colors=CLASS_COLORS, colors_dark=CLASS_COLORS_DARK)


@router.get("/hotspots", response_model=HotspotsResponse)
def hotspots() -> HotspotsResponse:
    """Hotspots for the Leaflet page -- the 21-row synthetic demo CSV. The national
    map (frontend/index.html, served at /) reads /api/map/* instead."""
    frame = pd.read_csv(DEMO_HOTSPOTS_PATH).replace({np.nan: None})
    records = [HotspotRecord(**row) for row in frame.to_dict(orient="records")]
    return HotspotsResponse(hotspots=records, source=DEMO_HOTSPOTS_PATH.name)


def _map_meta() -> dict:
    try:
        return map_store.meta()
    except MapDataUnavailable as error:
        raise HTTPException(status_code=503, detail=str(error)) from error


@router.get("/map/meta")
def map_meta() -> dict:
    meta = _map_meta()
    return {key: value for key, value in meta.items() if key != "sources"}


@router.get("/map/points.bin")
def map_points(request: Request) -> FileResponse:
    _map_meta()
    # revalidate via ETag -- the file changes whenever it is repacked
    headers = {"Cache-Control": "no-cache", "Vary": "Accept-Encoding"}
    if "gzip" in request.headers.get("accept-encoding", "") and map_store.gz_path.exists():
        headers["Content-Encoding"] = "gzip"
        return FileResponse(map_store.gz_path, media_type="application/octet-stream", headers=headers)
    return FileResponse(map_store.points_path, media_type="application/octet-stream", headers=headers)


@router.get("/detection/{row_id}", response_model=DetectionRecord)
def detection(row_id: int) -> DetectionRecord:
    _map_meta()
    record = map_store.detection(row_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"no detection with id {row_id}")
    return DetectionRecord(**record)
