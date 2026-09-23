"""One-off: re-run only the OSM industrial-context join on an already-labeled
national parquet after excluding solar/wind power=plant features
(backend.ingestion.context_sources.exclude_renewable_power_plants), then re-run
apply_rules() -- reuses every other cached column (dist_to_flare_capable_m,
dist_to_heat_industry_m, dist_to_coal_mine_m, landcover_class, recurrence_count,
is_anomalous, ...) instead of re-running the whole build_labels pipeline, since
none of those inputs read from the industrial-context layer's power=plant rows
(the coal-mine OSM fallback matches on industrial=mine, never present on a
solar/wind plant, so it's unaffected too). Not part of the routine build_labels.py
path -- a rule-input fix (solar/wind excluded from industrial context) triggered
this one-time patch of already-built outputs, same reasoning as
recompute_coal_and_relabel.py's coal-mine patch.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import geopandas as gpd
import pandas as pd

from backend.classification.rules import apply_rules
from backend.config import CONTRACT_COLUMNS, NATIONAL_PROJECTED_CRS
from backend.ingestion.context_sources import exclude_renewable_power_plants
from training.spatial_features import nearest_industrial_distance, osm_industrial_tag


def recompute_and_relabel(labeled_path: str | Path, industrial_context_path: str | Path) -> pd.DataFrame:
    frame = pd.read_parquet(labeled_path)
    detections = gpd.GeoDataFrame(
        frame, geometry=gpd.points_from_xy(frame["longitude"], frame["latitude"]), crs="EPSG:4326"
    ).to_crs(NATIONAL_PROJECTED_CRS)

    industrial = gpd.read_parquet(industrial_context_path).to_crs(NATIONAL_PROJECTED_CRS)
    industrial = exclude_renewable_power_plants(industrial)

    frame["dist_to_industrial_m"] = nearest_industrial_distance(detections, industrial).values
    frame["osm_industrial_tag"] = osm_industrial_tag(detections, industrial).values

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
