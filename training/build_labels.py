"""Join FIRMS detections to context and create the teammate-2 hand-off CSV."""

from __future__ import annotations

import argparse
from pathlib import Path

import geopandas as gpd
import pandas as pd

from backend.classification.rules import FLARE_CAPABLE_FACILITY_TYPES, HEAT_INDUSTRY_FACILITY_TYPES, apply_rules
from backend.config import CONTRACT_COLUMNS, NATIONAL_PROJECTED_CRS
from training.spatial_features import (
    add_recurrence_features,
    compute_is_anomalous,
    grid_keys,
    landcover_majority,
    nearest_facility_distance,
    nearest_industrial_distance,
    osm_industrial_tag,
)
from training.worldcover_index import WorldCoverTileIndex


def _read_vector(path: str | Path) -> gpd.GeoDataFrame:
    """GeoJSON (regional/testing caches) or GeoParquet (national-scale caches,
    e.g. extract_osm_industrial_context's output), picked by file extension."""
    path = Path(path)
    return gpd.read_parquet(path) if path.suffix == ".parquet" else gpd.read_file(path)


def build_labels(
    firms_csv: str | Path,
    industrial_geojson: str | Path,
    facilities_geojson: str | Path,
    output_csv: str | Path,
    *,
    worldcover_raster: str | Path | None = None,
) -> pd.DataFrame:
    """firms_csv may be .csv or .parquet. industrial_geojson/facilities_geojson may be
    .geojson (regional caches, gpd.read_file) or .parquet (national-scale caches, e.g.
    backend.ingestion.context_sources.extract_osm_industrial_context's output,
    gpd.read_parquet) -- picked by extension either way. worldcover_raster may be a
    single-tile GeoTIFF path (regional) or a directory of tiles (national -- opened
    as a WorldCoverTileIndex, so only the tiles actually covering these detections
    are ever read, never a raster loaded whole for a handful of lookups).

    Distances are computed in NATIONAL_PROJECTED_CRS (EPSG:7755, an India-wide
    Lambert conformal conic) rather than EPSG:3857/Web Mercator -- Web Mercator's
    distance distortion grows with latitude and is severe enough at India's scale
    (roughly 6N-37N) to bias dist_to_*_m by a meaningfully different amount at
    Kashmir than at Kanyakumari; a single equal-ish conformal CRS keeps that
    distortion approximately constant across the whole country.
    """
    firms_csv = Path(firms_csv)
    firms = pd.read_parquet(firms_csv) if firms_csv.suffix == ".parquet" else pd.read_csv(firms_csv)
    industrial = _read_vector(industrial_geojson).to_crs(NATIONAL_PROJECTED_CRS)
    facilities = _read_vector(facilities_geojson).to_crs(NATIONAL_PROJECTED_CRS)
    detections = gpd.GeoDataFrame(
        firms,
        geometry=gpd.points_from_xy(firms["longitude"], firms["latitude"]),
        crs="EPSG:4326",
    ).to_crs(NATIONAL_PROJECTED_CRS)

    result = firms.copy()
    result["dist_to_industrial_m"] = nearest_industrial_distance(detections, industrial).values
    result["osm_industrial_tag"] = osm_industrial_tag(detections, industrial).values
    flare_dist, flare_type = nearest_facility_distance(detections, facilities, FLARE_CAPABLE_FACILITY_TYPES)
    heat_dist, heat_type = nearest_facility_distance(detections, facilities, HEAT_INDUSTRY_FACILITY_TYPES)
    result["dist_to_flare_capable_m"] = flare_dist.values
    result["nearest_flare_facility_type"] = flare_type.values
    result["dist_to_heat_industry_m"] = heat_dist.values
    result["nearest_heat_facility_type"] = heat_type.values

    grid_lat, grid_lon = grid_keys(result)
    result = add_recurrence_features(result, grid_lat, grid_lon)
    result["is_anomalous"] = compute_is_anomalous(result, grid_lat, grid_lon).values

    if worldcover_raster is not None and Path(worldcover_raster).is_dir():
        tile_index = WorldCoverTileIndex(worldcover_raster)
        result["landcover_class"] = tile_index.majority_class_in_buffer_batch(
            firms["longitude"], firms["latitude"]
        ).values
    elif worldcover_raster is not None:
        result["landcover_class"] = landcover_majority(detections.geometry, worldcover_raster).values
    else:
        result["landcover_class"] = result.get("landcover_class", pd.Series(index=result.index, dtype="object"))

    result = apply_rules(result)
    for column in CONTRACT_COLUMNS:
        if column not in result:
            result[column] = None
    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.suffix == ".parquet":
        result[CONTRACT_COLUMNS].to_parquet(output_path, index=False)
    else:
        result[CONTRACT_COLUMNS].to_csv(output_path, index=False)
    return result[CONTRACT_COLUMNS]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--firms", type=Path, required=True)
    parser.add_argument("--industrial", type=Path, required=True)
    parser.add_argument("--facilities", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--worldcover", type=Path, default=None, help="ESA WorldCover GeoTIFF tile, or a directory of tiles")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    build_labels(args.firms, args.industrial, args.facilities, args.output, worldcover_raster=args.worldcover)
