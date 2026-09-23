from __future__ import annotations

import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import Point

from training.spatial_features import (
    add_recurrence_features,
    compute_is_anomalous,
    grid_keys,
    nearest_coal_mine_distance,
    nearest_facility_distance,
    nearest_industrial_distance,
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


def test_nearest_industrial_distance_picks_nearer_polygon():
    from shapely.geometry import box

    detections = gpd.GeoDataFrame({"geometry": [Point(0, 0)]}, crs="EPSG:3857")
    industrial = gpd.GeoDataFrame(
        {"geometry": [box(100, -10, 120, 10), box(500, -10, 520, 10)]}, crs="EPSG:3857"
    )
    dist = nearest_industrial_distance(detections, industrial)
    assert dist.iloc[0] == pytest.approx(100.0)  # distance to the near polygon's edge, not the far one


def test_nearest_industrial_distance_is_zero_for_a_point_inside_a_polygon():
    from shapely.geometry import box

    detections = gpd.GeoDataFrame({"geometry": [Point(5, 5)]}, crs="EPSG:3857")
    industrial = gpd.GeoDataFrame({"geometry": [box(0, 0, 10, 10)]}, crs="EPSG:3857")
    dist = nearest_industrial_distance(detections, industrial)
    assert dist.iloc[0] == pytest.approx(0.0)


def test_nearest_industrial_distance_matches_union_all_distance():
    """Regression guard: sjoin_nearest against individual polygons (the current,
    spatial-indexed implementation) must give the same numbers the old
    detections.geometry.distance(industrial.union_all()) approach gave -- only the
    performance should have changed, not the values every rule threshold reads."""
    from shapely.geometry import box

    rng_points = [Point(x, y) for x, y in [(0, 0), (300, 0), (1000, 1000), (-50, 200)]]
    detections = gpd.GeoDataFrame({"geometry": rng_points}, crs="EPSG:3857")
    industrial = gpd.GeoDataFrame(
        {"geometry": [box(100, -10, 120, 10), box(-60, 190, -40, 210), box(2000, 2000, 2010, 2010)]},
        crs="EPSG:3857",
    )

    indexed = nearest_industrial_distance(detections, industrial)
    union_geom = industrial.geometry.union_all()
    unindexed = detections.geometry.distance(union_geom)

    for a, b in zip(indexed.tolist(), unindexed.tolist()):
        assert a == pytest.approx(b)


def test_nearest_industrial_distance_empty_industrial_returns_all_null():
    detections = gpd.GeoDataFrame({"geometry": [Point(0, 0)]}, crs="EPSG:3857")
    industrial = gpd.GeoDataFrame({"geometry": []}, crs="EPSG:3857")
    dist = nearest_industrial_distance(detections, industrial)
    assert dist.isna().all()


def _empty_geoframe() -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame({"geometry": []}, crs="EPSG:3857")


def test_nearest_coal_mine_distance_gem_boundary_wins_over_closer_osm():
    """GEM is checked first and wins whenever it has any data at all, even if an
    OSM industrial=mine feature happens to be closer -- GEM is the stronger source."""
    from shapely.geometry import box

    detections = gpd.GeoDataFrame({"geometry": [Point(0, 0)]}, crs="EPSG:3857")
    gem_boundaries = gpd.GeoDataFrame({"geometry": [box(500, -10, 520, 10)]}, crs="EPSG:3857")
    gem_points = _empty_geoframe()
    osm_mines = gpd.GeoDataFrame({"geometry": [Point(50, 0)]}, crs="EPSG:3857")  # much closer than GEM

    dist, source = nearest_coal_mine_distance(detections, gem_boundaries, gem_points, osm_mines)
    assert dist.iloc[0] == pytest.approx(500.0)  # GEM's distance, not OSM's 50
    assert source.iloc[0] == "gem"


def test_nearest_coal_mine_distance_falls_back_to_osm_when_gem_has_no_data():
    detections = gpd.GeoDataFrame({"geometry": [Point(0, 0)]}, crs="EPSG:3857")
    osm_mines = gpd.GeoDataFrame({"geometry": [Point(200, 0)]}, crs="EPSG:3857")

    dist, source = nearest_coal_mine_distance(detections, _empty_geoframe(), _empty_geoframe(), osm_mines)
    assert dist.iloc[0] == pytest.approx(200.0)
    assert source.iloc[0] == "osm"


def test_nearest_coal_mine_distance_returns_null_when_no_source_has_data():
    detections = gpd.GeoDataFrame({"geometry": [Point(0, 0)]}, crs="EPSG:3857")
    dist, source = nearest_coal_mine_distance(detections, _empty_geoframe(), _empty_geoframe(), _empty_geoframe())
    assert dist.isna().all()
    assert source.isna().all()


def test_nearest_coal_mine_distance_point_scaled_by_area_km2():
    """A GEM point-only mine's effective distance is point_distance minus the
    radius of a circle with the same area_km2 -- an approximation of "distance to
    the boundary" when GEM only has a centroid, not a real boundary polygon."""
    detections = gpd.GeoDataFrame({"geometry": [Point(0, 0)]}, crs="EPSG:3857")
    gem_points = gpd.GeoDataFrame({"geometry": [Point(1000, 0)], "area_km2": [1.0]}, crs="EPSG:3857")

    dist, source = nearest_coal_mine_distance(detections, _empty_geoframe(), gem_points, _empty_geoframe())
    radius_m = (1.0 * 1_000_000 / 3.141592653589793) ** 0.5  # ~564.19 m
    assert dist.iloc[0] == pytest.approx(1000.0 - radius_m, abs=0.1)
    assert source.iloc[0] == "gem"


def test_nearest_coal_mine_distance_point_area_scaling_clips_at_zero():
    """A detection inside a large mine's area-derived circle must read as 0m, not
    negative -- point_distance smaller than the mine's own radius."""
    detections = gpd.GeoDataFrame({"geometry": [Point(0, 0)]}, crs="EPSG:3857")
    gem_points = gpd.GeoDataFrame({"geometry": [Point(100, 0)], "area_km2": [50.0]}, crs="EPSG:3857")

    dist, source = nearest_coal_mine_distance(detections, _empty_geoframe(), gem_points, _empty_geoframe())
    assert dist.iloc[0] == pytest.approx(0.0)
    assert source.iloc[0] == "gem"


def test_nearest_coal_mine_distance_prefers_smaller_of_gem_boundary_and_point_fallback():
    """When a detection is close to a boundary-mapped mine but also has an
    unrelated point-only GEM mine nearby, the smaller (more accurate) of the two
    GEM-sourced distances wins -- still tagged "gem" either way."""
    from shapely.geometry import box

    detections = gpd.GeoDataFrame({"geometry": [Point(0, 0)]}, crs="EPSG:3857")
    gem_boundaries = gpd.GeoDataFrame({"geometry": [box(1000, -10, 1020, 10)]}, crs="EPSG:3857")
    gem_points = gpd.GeoDataFrame({"geometry": [Point(200, 0)], "area_km2": [0.0]}, crs="EPSG:3857")

    dist, source = nearest_coal_mine_distance(detections, gem_boundaries, gem_points, _empty_geoframe())
    assert dist.iloc[0] == pytest.approx(200.0)  # the point (no area) is closer than the boundary
    assert source.iloc[0] == "gem"
