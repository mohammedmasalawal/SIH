"""Report label mix broken down by Indian state/UT, for a national labeled_hotspots build.

State assignment is a point-in-polygon join against Natural Earth's admin-1
boundaries (data/external/boundaries/india_states.parquet, filtered to India,
36 states/UTs) -- purely a reporting join. It is NOT written into CONTRACT_COLUMNS
or the training CSV: this phase is ingest/context only, no rule or contract changes.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import geopandas as gpd
import pandas as pd

DEFAULT_STATES_PATH = Path("data/external/boundaries/india_states.parquet")


def assign_state(frame: pd.DataFrame, states_path: str | Path = DEFAULT_STATES_PATH) -> pd.Series:
    """Indian state/UT name for each row (point-in-polygon), or NaN if outside every
    state polygon (e.g. offshore detections, or coordinates outside India)."""
    states = gpd.read_parquet(states_path)[["name", "geometry"]]
    states.sindex  # spatial index for the join, built explicitly rather than lazily
    points = gpd.GeoDataFrame(
        {"_lon": frame["longitude"].to_numpy(), "_lat": frame["latitude"].to_numpy()},
        geometry=gpd.points_from_xy(frame["longitude"], frame["latitude"]),
        index=frame.index,
        crs="EPSG:4326",
    )
    joined = gpd.sjoin(points, states, how="left", predicate="within")
    joined = joined[~joined.index.duplicated(keep="first")]  # a point exactly on a shared border can double-match
    return joined["name"].reindex(frame.index)


def label_mix_by_state(frame: pd.DataFrame, states_path: str | Path = DEFAULT_STATES_PATH) -> pd.DataFrame:
    """Crosstab of label x state, with a `total` column, sorted by total descending.
    Rows with no matching state polygon are grouped under "(outside any state polygon)"."""
    state = assign_state(frame, states_path).fillna("(outside any state polygon)")
    table = pd.crosstab(state, frame["label"])
    table["total"] = table.sum(axis=1)
    return table.sort_values("total", ascending=False)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labeled", type=Path, required=True, help="labeled_hotspots.csv or .parquet")
    parser.add_argument("--states", type=Path, default=DEFAULT_STATES_PATH)
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    frame = pd.read_parquet(args.labeled) if args.labeled.suffix == ".parquet" else pd.read_csv(args.labeled)
    table = label_mix_by_state(frame, args.states)
    pd.set_option("display.width", 200)
    pd.set_option("display.max_rows", 100)
    print(table)
    print(f"\nTotal detections: {len(frame):,}")
