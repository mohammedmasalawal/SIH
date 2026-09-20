"""STEP 1.5 (national scale-up): time the nearest-industrial/nearest-facility spatial
joins on a real dataset, in NATIONAL_PROJECTED_CRS, with spatial indexes built up
front. Every distance computed here (dist_to_industrial_m via
nearest_industrial_distance, dist_to_flare/heat_capable_m via
nearest_facility_distance) now goes through gpd.sjoin_nearest against individual
features -- an earlier version computed dist_to_industrial_m via
detections.geometry.distance(industrial.union_all()), a single unpindexed distance
call against one mega-geometry that measured 35.9s of a 46.5s total join time on a
13,120-detection/37,908-feature national benchmark; that's fixed now.

Not part of build_labels.py's own hot path (that would add timing prints to every
routine run) -- run this directly against the same FIRMS/OSM/GEM inputs you're about
to hand to build_labels.py, to get a timing number before committing to the full run.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import geopandas as gpd
import pandas as pd

from backend.classification.rules import FLARE_CAPABLE_FACILITY_TYPES, HEAT_INDUSTRY_FACILITY_TYPES
from backend.config import NATIONAL_PROJECTED_CRS
from training.build_labels import _read_vector
from training.spatial_features import nearest_facility_distance, nearest_industrial_distance, osm_industrial_tag


def benchmark(firms_path: str | Path, industrial_path: str | Path, facilities_path: str | Path) -> dict:
    firms_path = Path(firms_path)
    firms = pd.read_parquet(firms_path) if firms_path.suffix == ".parquet" else pd.read_csv(firms_path)

    t0 = time.perf_counter()
    industrial = _read_vector(industrial_path).to_crs(NATIONAL_PROJECTED_CRS)
    facilities = _read_vector(facilities_path).to_crs(NATIONAL_PROJECTED_CRS)
    detections = gpd.GeoDataFrame(
        firms, geometry=gpd.points_from_xy(firms["longitude"], firms["latitude"]), crs="EPSG:4326"
    ).to_crs(NATIONAL_PROJECTED_CRS)
    t_reproject = time.perf_counter() - t0

    t0 = time.perf_counter()
    industrial.sindex  # warm the polygon spatial index shared by the calls below
    t_industrial_index = time.perf_counter() - t0

    t0 = time.perf_counter()
    _ = nearest_industrial_distance(detections, industrial)
    t_industrial_distance = time.perf_counter() - t0

    t0 = time.perf_counter()
    _ = osm_industrial_tag(detections, industrial)
    t_industrial_tag = time.perf_counter() - t0

    t0 = time.perf_counter()
    _flare_dist, _flare_type = nearest_facility_distance(detections, facilities, FLARE_CAPABLE_FACILITY_TYPES)
    t_flare_join = time.perf_counter() - t0

    t0 = time.perf_counter()
    _heat_dist, _heat_type = nearest_facility_distance(detections, facilities, HEAT_INDUSTRY_FACILITY_TYPES)
    t_heat_join = time.perf_counter() - t0

    return {
        "n_detections": len(detections),
        "n_industrial_features": len(industrial),
        "n_facilities": len(facilities),
        "crs": NATIONAL_PROJECTED_CRS,
        "reproject_seconds": round(t_reproject, 3),
        "industrial_sindex_build_seconds": round(t_industrial_index, 3),
        "dist_to_industrial_join_seconds": round(t_industrial_distance, 3),
        "osm_industrial_tag_join_seconds": round(t_industrial_tag, 3),
        "flare_facility_sjoin_nearest_seconds": round(t_flare_join, 3),
        "heat_facility_sjoin_nearest_seconds": round(t_heat_join, 3),
        "total_seconds": round(
            t_reproject + t_industrial_index + t_industrial_distance + t_industrial_tag + t_flare_join + t_heat_join, 3
        ),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--firms", type=Path, required=True)
    parser.add_argument("--industrial", type=Path, required=True)
    parser.add_argument("--facilities", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    result = benchmark(args.firms, args.industrial, args.facilities)
    for key, value in result.items():
        print(f"{key}: {value}")
