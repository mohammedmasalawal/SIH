"""Pydantic request/response contracts for backend/api/routes.py.

No Literal/enum constraints on categorical fields (satellite, confidence, daynight,
landcover_class, nearest_*_facility_type) and no non-negativity constraint on
distances: an unrecognized or out-of-range value must reach
backend/classification/features.py's _other/sentinel handling as a normal 200
response, not bounce as a 422. Only fields the pipeline cannot sensibly proceed
without (latitude/longitude/acq_date/acq_time/satellite) are required.
"""

from __future__ import annotations

from pydantic import BaseModel


class ClassifyRequest(BaseModel):
    latitude: float
    longitude: float
    acq_date: str
    acq_time: str
    satellite: str
    bright_ti4: float | None = None
    bright_ti5: float | None = None
    frp: float | None = None
    confidence: str | None = None
    daynight: str | None = None
    dist_to_industrial_m: float | None = None
    dist_to_flare_capable_m: float | None = None
    nearest_flare_facility_type: str | None = None
    dist_to_heat_industry_m: float | None = None
    nearest_heat_facility_type: str | None = None
    landcover_class: str | None = None
    recurrence_count: int = 1
    is_anomalous: bool = False


class ClassifyResponse(BaseModel):
    label: str
    label_source: str  # "model" | "rule" | "unknown"
    probabilities: dict[str, float] | None = None
    model_version: str | None = None


class HealthResponse(BaseModel):
    status: str = "ok"
    model_loaded: bool
