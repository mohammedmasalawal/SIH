from __future__ import annotations

import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import Point

from training.spatial_features import (
    add_recurrence_features,
    compute_is_anomalous,
    grid_keys,
    nearest_facility_distance,
)


def test_grid_keys_snaps_nearby_points_together():
    frame = pd.DataFrame({"latitude": [22.300, 22.3001], "longitude": [69.850, 69.8501]})
    grid_lat, grid_lon = grid_keys(frame)
    assert grid_lat.iloc[0] == grid_lat.iloc[1]
    assert grid_lon.iloc[0] == grid_lon.iloc[1]


def test_grid_keys_separates_far_points():
    frame = pd.DataFrame({"latitude": [22.30, 23.30], "longitude": [69.85, 70.85]})
    grid_lat, grid_lon = grid_keys(frame)
    assert grid_lat.iloc[0] != grid_lat.iloc[1]


def test_recurrence_counts_distinct_active_days_per_cell():
    frame = pd.DataFrame({
        "latitude": [22.30, 22.30, 22.30, 25.00],
        "longitude": [69.85, 69.85, 69.85, 68.00],
        "acq_date": ["2026-01-01", "2026-01-02", "2026-01-02", "2026-01-05"],
    })
    grid_lat, grid_lon = grid_keys(frame)
    result = add_recurrence_features(frame, grid_lat, grid_lon)
    assert result["recurrence_count"].iloc[0:3].tolist() == [2, 2, 2]
    assert result["first_seen"].iloc[0] == "2026-01-01"
    assert result["last_seen"].iloc[0] == "2026-01-02"
    assert result["recurrence_count"].iloc[3] == 1


def test_is_anomalous_requires_min_prior_days_then_flags_spike():
    dates = [f"2026-01-{d:02d}" for d in range(1, 8)]
    frps = [5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 100.0]
    frame = pd.DataFrame({
        "latitude": [22.30] * 7,
        "longitude": [69.85] * 7,
        "acq_date": dates,
        "daynight": ["D"] * 7,
        "frp": frps,
    })
    grid_lat, grid_lon = grid_keys(frame)
    flags = compute_is_anomalous(frame, grid_lat, grid_lon, min_prior_active_days=5)
    assert flags.iloc[:6].tolist() == [False] * 6
    assert bool(flags.iloc[6]) is True


def test_nearest_facility_distance_picks_nearer_candidate():
    detections = gpd.GeoDataFrame({"geometry": [Point(0, 0)]}, crs="EPSG:3857")
    facilities = gpd.GeoDataFrame(
        {"facility_type": ["cement_plant", "cement_plant"], "geometry": [Point(100, 0), Point(500, 0)]},
        crs="EPSG:3857",
    )
    dist, facility_type = nearest_facility_distance(detections, facilities, ("cement_plant",))
    assert dist.iloc[0] == pytest.approx(100.0)
    assert facility_type.iloc[0] == "cement_plant"
