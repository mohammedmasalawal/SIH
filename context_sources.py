"""Download and cache spatial context used by the hotspot label builder."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import geopandas as gpd
import pandas as pd
import requests
from shapely.geometry import Point, Polygon

OVERPASS_URL = "https://overpass-api.de/api/interpreter"
OVERPASS_HEADERS = {"User-Agent": "agninetra-firms-pipeline/1.0 (SIH 2026 PS 26162)"}


def _bbox_string(bbox: Iterable[float]) -> str:
    west, south, east, north = (float(value) for value in bbox)
    if not west < east or not south < north:
        raise ValueError("bbox must satisfy west < east and south < north")
    return f"{south},{west},{north},{east}"


def _feature_from_element(element: dict[str, Any]) -> dict[str, Any] | None:
    tags = element.get("tags", {})
    geometry = element.get("geometry", [])
    if not geometry:
        return None
    coordinates = [(item["lon"], item["lat"]) for item in geometry]
    if element["type"] == "way" and len(coordinates) >= 4 and coordinates[0] == coordinates[-1]:
        shape = Polygon(coordinates)
    else:
        shape = Point(coordinates[0])
    return {
        "type": "Feature",
        "properties": {"osm_id": element["id"], **tags},
        "geometry": shape.__geo_interface__,
    }


def fetch_osm_industrial(bbox: Iterable[float], cache_path: str | Path) -> gpd.GeoDataFrame:
    """Fetch OSM industrial areas/facilities once and reuse the GeoJSON cache."""
    destination = Path(cache_path)
    if destination.exists():
        return gpd.read_file(destination)

    area_bbox = _bbox_string(bbox)
    query = (
        "[out:json][timeout:90];"
        f"(way[landuse=industrial]({area_bbox});"
        f"node[industrial]({area_bbox});way[industrial]({area_bbox});"
        f"node[man_made=flare]({area_bbox});way[man_made=flare]({area_bbox}););"
        "out body geom;"
    )
    response = requests.post(OVERPASS_URL, data={"data": query}, headers=OVERPASS_HEADERS, timeout=120)
    response.raise_for_status()
    features = [feature for item in response.json().get("elements", []) if (feature := _feature_from_element(item))]
    frame = gpd.GeoDataFrame.from_features(features, crs="EPSG:4326")
    destination.parent.mkdir(parents=True, exist_ok=True)
    frame.to_file(destination, driver="GeoJSON")
    return frame


def load_gem_facilities(source: str | Path, cache_path: str | Path) -> gpd.GeoDataFrame:
    """Load a GEM CSV/URL once, normalising latitude/longitude to GeoJSON."""
    destination = Path(cache_path)
    if destination.exists():
        return gpd.read_file(destination)

    frame = pd.read_csv(source)
    columns = {column.lower().strip(): column for column in frame.columns}
    latitude = columns.get("latitude", columns.get("lat"))
    longitude = columns.get("longitude", columns.get("lon", columns.get("lng")))
    if not latitude or not longitude:
        raise ValueError("GEM source needs latitude/longitude columns")
    frame = frame.dropna(subset=[latitude, longitude]).copy()
    frame["facility_type"] = frame.get("facility_type", frame.get("type", "industrial"))
    facilities = gpd.GeoDataFrame(
        frame,
        geometry=gpd.points_from_xy(frame[longitude], frame[latitude]),
        crs="EPSG:4326",
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    facilities.to_file(destination, driver="GeoJSON")
    return facilities


def worldcover_class(longitude: float, latitude: float, raster_path: str | Path) -> int | None:
    """Return the ESA WorldCover class at a point; kept optional for the join stage."""
    try:
        import rasterio
        from rasterio.sample import sample_gen
    except ImportError as error:
        raise RuntimeError("Install rasterio to use WorldCover lookup") from error

    with rasterio.open(raster_path) as raster:
        value = next(sample_gen(raster, [(longitude, latitude)]))[0]
        return None if raster.nodata is not None and value == raster.nodata else int(value)
