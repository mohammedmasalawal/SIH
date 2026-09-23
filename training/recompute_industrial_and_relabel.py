"""One-off: re-run only the OSM industrial-context-derived columns on an
already-labeled national parquet, then re-run apply_rules() -- reuses every
other cached column (dist_to_flare_capable_m, dist_to_heat_industry_m,
dist_to_coal_mine_m, landcover_class, recurrence_count, is_anomalous, ...)
instead of re-running the whole build_labels pipeline, since none of those
inputs read from the industrial-context layer (the coal-mine OSM fallback
matches on industrial=mine, never present on a solar/wind plant or a brick
kiln, so it's unaffected too). Not part of the routine build_labels.py path --
a rule-input fix triggered this one-time patch of already-built outputs, same
reasoning as recompute_coal_and_relabel.py's coal-mine patch.

Recomputes, from the industrial-context parquet:
    dist_to_industrial_m, osm_industrial_tag  -- solar/wind power=plant excluded
    dist_to_brick_kiln_m                      -- industrial=brickyard/man_made=kiln
"""

from __future__ import annotations

import argparse
from pathlib import Path

import geopandas as gpd
import pandas as pd

from backend.classification.rules import apply_rules
from backend.config import CONTRACT_COLUMNS, NATIONAL_PROJECTED_CRS
from backend.ingestion.context_sources import exclude_renewable_power_plants, load_osm_brick_kilns
from training.spatial_features import nearest_brick_kiln_distance, nearest_industrial_distance, osm_industrial_tag


def recompute_and_relabel(labeled_path: str | Path, industrial_context_path: str | Path) -> pd.DataFrame:
    frame = pd.read_parquet(labeled_path)
    detections = gpd.GeoDataFrame(
        frame, geometry=gpd.points_from_xy(frame["longitude"], frame["latitude"]), crs="EPSG:4326"
    ).to_crs(NATIONAL_PROJECTED_CRS)

    industrial = gpd.read_parquet(industrial_context_path).to_crs(NATIONAL_PROJECTED_CRS)
    industrial = exclude_renewable_power_plants(industrial)

    frame["dist_to_industrial_m"] = nearest_industrial_distance(detections, industrial).values
    frame["osm_industrial_tag"] = osm_industrial_tag(detections, industrial).values

    osm_brick_kilns = load_osm_brick_kilns(industrial)
    frame["dist_to_brick_kiln_m"] = nearest_brick_kiln_distance(detections, osm_brick_kilns).values

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
