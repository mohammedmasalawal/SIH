"""Offshore zones: which FIRMS detections count as offshore, and how far out they are.

A detection is offshore when it sits inside one of backend.config.OFFSHORE_ZONES, outside
the India boundary polygon, and at least OFFSHORE_MIN_DISTANCE_KM beyond it. The boundary is
Natural Earth's admin-1 polygons, which are generalised: within a few km of the coast they
can put land or shoreline on the wrong side, so the distance floor keeps that slop out.
"""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

from backend.config import (
    INDIA_BOUNDARY_PATH,
    NATIONAL_PROJECTED_CRS,
    OFFSHORE_MIN_DISTANCE_KM,
    OFFSHORE_ZONES,
)


def zone_of(longitude: pd.Series, latitude: pd.Series, zones: dict = OFFSHORE_ZONES) -> pd.Series:
    """Name of the offshore zone each point falls in (bounds inclusive), or None."""
    lon, lat = np.asarray(longitude, dtype="float64"), np.asarray(latitude, dtype="float64")
    zone = pd.Series(None, index=longitude.index, dtype="object")
    for name, (west, south, east, north) in zones.items():
        zone[(lon >= west) & (lon <= east) & (lat >= south) & (lat <= north)] = name
    return zone


def offshore_columns(
    frame: pd.DataFrame,
    boundary_path: str | Path | None = None,
    *,
    min_distance_km: float = OFFSHORE_MIN_DISTANCE_KM,
    zones: dict = OFFSHORE_ZONES,
) -> pd.DataFrame:
    """offshore_zone and dist_offshore_km (distance to the India boundary, km) per row of
    `frame`, both null unless the row is offshore. Only rows inside a zone box are measured,
    so this stays cheap on a national frame. boundary_path defaults to INDIA_BOUNDARY_PATH,
    looked up at call time."""
    boundary_path = boundary_path or INDIA_BOUNDARY_PATH
    out = pd.DataFrame(
        {"offshore_zone": pd.Series(None, index=frame.index, dtype="object"),
         "dist_offshore_km": pd.Series(np.nan, index=frame.index, dtype="float64")}
    )
    zone = zone_of(frame["longitude"], frame["latitude"], zones)
    candidates = frame.index[zone.notna().to_numpy()]
    if len(candidates) == 0:
        return out
    boundary = gpd.read_parquet(boundary_path).to_crs(NATIONAL_PROJECTED_CRS)
    india = boundary.geometry.union_all()
    points = gpd.GeoSeries(
        gpd.points_from_xy(frame.loc[candidates, "longitude"], frame.loc[candidates, "latitude"]),
        index=candidates, crs="EPSG:4326",
    ).to_crs(NATIONAL_PROJECTED_CRS)
    distance_km = points.distance(india) / 1000.0  # 0 for a point inside India
    keep = distance_km >= min_distance_km
    out.loc[candidates[keep.to_numpy()], "offshore_zone"] = zone.loc[candidates[keep.to_numpy()]]
    out.loc[candidates[keep.to_numpy()], "dist_offshore_km"] = distance_km[keep].round(1)
    return out


def is_offshore(frame: pd.DataFrame, boundary_path: str | Path | None = None, **kwargs) -> pd.Series:
    """Boolean mask: the row is an offshore detection (see module docstring)."""
    return offshore_columns(frame, boundary_path, **kwargs)["offshore_zone"].notna()
