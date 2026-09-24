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
    osm_industrial_tag: str | None = None
    dist_to_flare_capable_m: float | None = None
    nearest_flare_facility_type: str | None = None
    dist_to_heat_industry_m: float | None = None
    nearest_heat_facility_type: str | None = None
    dist_to_coal_mine_m: float | None = None
    nearest_coal_source: str | None = None
    landcover_class: str | None = None
    recurrence_count: int = 1
    is_anomalous: bool = False


class ClassifyResponse(BaseModel):
    label: str
    # "model" | "rule" | "unknown" -- this is the live single-hotspot /api/classify path
    # (backend/classification/classifier.py), not the batch training/build_labels.py ->
    # rules.apply_rules() contract, so it never returns "conflict": classify_hotspot()
    # falls back to rules.classify() (single label or "unknown"), never apply_rules().
    # "model" appears only when MODEL_ARTIFACT_PATH is loaded and predict() succeeds.
    label_source: str
    probabilities: dict[str, float] | None = None
    model_version: str | None = None


class HealthResponse(BaseModel):
    status: str = "ok"
    model_loaded: bool


class HotspotRecord(BaseModel):
    latitude: float
    longitude: float
    acq_date: str
    label: str
    label_source: str
    frp: float | None = None
    recurrence_count: int | None = None
    nearest_heat_facility_type: str | None = None
    nearest_flare_facility_type: str | None = None


class HotspotsResponse(BaseModel):
    hotspots: list[HotspotRecord]
    source: str  # file name the hotspots were read from, e.g. "demo_hotspots.csv"


class ClassesResponse(BaseModel):
    classes: list[str]
    colors: dict[str, str]  # light-basemap steps
    colors_dark: dict[str, str]  # same hues, dark-basemap steps


class DetectionRecord(BaseModel):
    """One labeled detection from the national parquets, for the map's click popup."""

    id: int
    latitude: float
    longitude: float
    acq_date: str
    acq_time: str | None = None
    satellite: str | None = None
    bright_ti4: float | None = None
    bright_ti5: float | None = None
    frp: float | None = None
    confidence: str | None = None
    daynight: str | None = None
    dist_to_industrial_m: float | None = None
    osm_industrial_tag: str | None = None
    dist_to_flare_capable_m: float | None = None
    nearest_flare_facility_type: str | None = None
    dist_to_heat_industry_m: float | None = None
    nearest_heat_facility_type: str | None = None
    dist_to_coal_mine_m: float | None = None
    nearest_coal_source: str | None = None
    dist_to_brick_kiln_m: float | None = None
    landcover_class: int | None = None
    recurrence_count: int | None = None
    first_seen: str | None = None
    last_seen: str | None = None
    is_anomalous: bool | None = None
    label: str
    label_source: str | None = None
