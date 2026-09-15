"""Join FIRMS detections to context and create the teammate-2 hand-off CSV."""

from __future__ import annotations

import argparse
from pathlib import Path

import geopandas as gpd
import pandas as pd

from backend.classification.rules import FLARE_CAPABLE_FACILITY_TYPES, HEAT_INDUSTRY_FACILITY_TYPES, apply_rules
from backend.config import CONTRACT_COLUMNS
from training.spatial_features import (
    add_recurrence_features,
    compute_is_anomalous,
    grid_keys,
    landcover_majority,
    nearest_facility_distance,
)


def build_labels(
    firms_csv: str | Path,
    industrial_geojson: str | Path,
    facilities_geojson: str | Path,
    output_csv: str | Path,
    *,
    worldcover_raster: str | Path | None = None,
) -> pd.DataFrame:
    firms = pd.read_csv(firms_csv)
    industrial = gpd.read_file(industrial_geojson).to_crs("EPSG:3857")
    facilities = gpd.read_file(facilities_geojson).to_crs("EPSG:3857")
    detections = gpd.GeoDataFrame(
        firms,
        geometry=gpd.points_from_xy(firms["longitude"], firms["latitude"]),
        crs="EPSG:4326",
    ).to_crs("EPSG:3857")

    industrial_union = industrial.geometry.union_all() if not industrial.empty else None
    result = firms.copy()
    result["dist_to_industrial_m"] = (
        detections.geometry.distance(industrial_union).values if industrial_union is not None else None
    )
    flare_dist, flare_type = nearest_facility_distance(detections, facilities, FLARE_CAPABLE_FACILITY_TYPES)
    heat_dist, heat_type = nearest_facility_distance(detections, facilities, HEAT_INDUSTRY_FACILITY_TYPES)
    result["dist_to_flare_capable_m"] = flare_dist.values
    result["nearest_flare_facility_type"] = flare_type.values
    result["dist_to_heat_industry_m"] = heat_dist.values
    result["nearest_heat_facility_type"] = heat_type.values

    grid_lat, grid_lon = grid_keys(result)
    result = add_recurrence_features(result, grid_lat, grid_lon)
    result["is_anomalous"] = compute_is_anomalous(result, grid_lat, grid_lon).values

    if worldcover_raster is not None:
        result["landcover_class"] = landcover_majority(detections.geometry, worldcover_raster).values
    else:
        result["landcover_class"] = result.get("landcover_class", pd.Series(index=result.index, dtype="object"))

    result["label"] = "unknown"
    result["label_source"] = "unclassified"
    result = apply_rules(result)
    for column in CONTRACT_COLUMNS:
        if column not in result:
            result[column] = None
    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result[CONTRACT_COLUMNS].to_csv(output_path, index=False)
    return result[CONTRACT_COLUMNS]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--firms", type=Path, required=True)
    parser.add_argument("--industrial", type=Path, required=True)
    parser.add_argument("--facilities", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--worldcover", type=Path, default=None, help="ESA WorldCover GeoTIFF tile")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    build_labels(args.firms, args.industrial, args.facilities, args.output, worldcover_raster=args.worldcover)
