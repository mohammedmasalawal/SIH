"""One-off: add dist_to_coal_mine_m/nearest_coal_source to an already-labeled
national parquet and re-run apply_rules() -- reuses every other cached column
(landcover_class, osm_industrial_tag, dist_to_heat/flare_*, recurrence_count,
is_anomalous, ...) instead of re-running the whole build_labels pipeline, since
none of those inputs changed. Not part of the routine build_labels.py path --
a rule change (new is_industrial coal-mine path) triggered this one-time patch
of already-built outputs.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import geopandas as gpd
import pandas as pd

from backend.classification.rules import apply_rules
from backend.config import (
    CONTRACT_COLUMNS,
    GEM_COAL_MINE_BOUNDARIES_PATH,
    GEM_COAL_MINE_CSV_PATH,
    GEM_COAL_STATUSES_EXCLUDED,
    NATIONAL_PROJECTED_CRS,
)
from backend.ingestion.context_sources import load_gem_coal_mine_boundaries, load_gem_coal_mine_points, load_osm_mines
from training.spatial_features import nearest_coal_mine_distance


def recompute_and_relabel(labeled_path: str | Path, industrial_context_path: str | Path) -> pd.DataFrame:
    frame = pd.read_parquet(labeled_path)
    detections = gpd.GeoDataFrame(
        frame, geometry=gpd.points_from_xy(frame["longitude"], frame["latitude"]), crs="EPSG:4326"
    ).to_crs(NATIONAL_PROJECTED_CRS)

    industrial = gpd.read_parquet(industrial_context_path).to_crs(NATIONAL_PROJECTED_CRS)
    gem_boundaries = load_gem_coal_mine_boundaries(GEM_COAL_MINE_BOUNDARIES_PATH).to_crs(NATIONAL_PROJECTED_CRS)
    gem_points = load_gem_coal_mine_points(GEM_COAL_MINE_CSV_PATH, excluded_statuses=GEM_COAL_STATUSES_EXCLUDED).to_crs(
        NATIONAL_PROJECTED_CRS
    )
    osm_mines = load_osm_mines(industrial)

    coal_dist, coal_source = nearest_coal_mine_distance(detections, gem_boundaries, gem_points, osm_mines)
    frame["dist_to_coal_mine_m"] = coal_dist.values
    frame["nearest_coal_source"] = coal_source.values

    frame = apply_rules(frame)
    return frame[CONTRACT_COLUMNS]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labeled", type=Path, required=True)
    parser.add_argument("--industrial", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    result = recompute_and_relabel(args.labeled, args.industrial)
    result.to_parquet(args.output, index=False)
    print(f"Wrote {len(result):,} rows to {args.output}")
