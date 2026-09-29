"""Read-only report: how many currently-`wildfire`-labelled detections would move to
`agricultural burning` if CROPLAND_SHARE_FIX_ENABLED were turned on (backend/config.py) --
i.e. a shrubland (20) or grassland (30) detection whose 1 km surroundings are more than
CROPLAND_SHARE_THRESHOLD cropland (class 40). Tree cover (10) is never affected.

Applies NOTHING: it doesn't write cropland_share_1km back onto any stored parquet, flip
the config switch, or touch a label. It reads every detection currently on the national
map (the same source list training.pack_map_points wrote to MAP_POINTS_META_PATH --
historical parquets plus every live month), computes cropland_share_1km with the
existing WorldCoverTileIndex for the wildfire/shrubland-or-grassland subset only, and
prints the count of would-change detections nationally and by state.

    python -m training.report_cropland_share_impact
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

from backend.config import (
    CROPLAND_LANDCOVER_CODES,
    CROPLAND_SHARE_RADIUS_M,
    CROPLAND_SHARE_SWITCH_CLASSES,
    CROPLAND_SHARE_THRESHOLD,
    MAP_POINTS_META_PATH,
    NATIONAL_WORLDCOVER_DIR,
    ROOT_DIR,
)
from training.report_by_state import assign_state
from training.worldcover_index import WorldCoverTileIndex

_READ_COLUMNS = ["latitude", "longitude", "label", "landcover_class"]


def _source_paths(meta_path=MAP_POINTS_META_PATH) -> list[Path]:
    """Every parquet the national map pack was last built from -- historical files
    plus every live month (training.pack_map_points writes this list to
    MAP_POINTS_META_PATH's "sources" on every repack)."""
    sources = json.loads(Path(meta_path).read_text())["sources"]
    paths = [Path(s["path"]) for s in sources]
    return [p if p.is_absolute() else ROOT_DIR / p for p in paths]


def load_wildfire_shrub_grass(meta_path=MAP_POINTS_META_PATH) -> pd.DataFrame:
    """Every currently `wildfire`-labelled detection on shrubland or grassland, across
    every source the national map pack lists (historical + live). Reads only the 4
    columns needed -- not the full contract -- since this is a read-only report."""
    frames = []
    for path in _source_paths(meta_path):
        present = set(pq.read_schema(path).names)
        columns = [c for c in _READ_COLUMNS if c in present]
        frame = pd.read_parquet(path, columns=columns).reindex(columns=_READ_COLUMNS)
        frames.append(frame[(frame["label"] == "wildfire")])
    rows = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=_READ_COLUMNS)
    code = pd.to_numeric(rows["landcover_class"], errors="coerce")
    return rows[code.isin(CROPLAND_SHARE_SWITCH_CLASSES)].reset_index(drop=True)


def compute_impact(
    rows: pd.DataFrame,
    worldcover_dir=NATIONAL_WORLDCOVER_DIR,
    radius_m: float = CROPLAND_SHARE_RADIUS_M,
    threshold: float = CROPLAND_SHARE_THRESHOLD,
) -> pd.DataFrame:
    """rows (from load_wildfire_shrub_grass) with cropland_share_1km and
    would_reclassify added. Nothing is written back anywhere."""
    if rows.empty:
        return rows.assign(cropland_share_1km=pd.Series(dtype="float64"), would_reclassify=pd.Series(dtype="bool"))
    index = WorldCoverTileIndex(worldcover_dir)
    share = index.class_share_in_buffer_batch(
        rows["longitude"], rows["latitude"], target_class=CROPLAND_LANDCOVER_CODES[0], buffer_m=radius_m
    )
    out = rows.copy()
    out["cropland_share_1km"] = share
    out["would_reclassify"] = share > threshold
    return out


def report(out: pd.DataFrame) -> str:
    state = assign_state(out[["latitude", "longitude"]]).fillna("(offshore/border)")
    changed = out[out["would_reclassify"]]
    by_state = (
        changed.assign(state=state[changed.index])
        .groupby("state").size().sort_values(ascending=False)
    )
    by_class = changed["landcover_class"].astype("Int64").value_counts()
    lines = [
        f"{len(out):,} wildfire detections on shrubland/grassland checked "
        f"(cropland_share_1km > {CROPLAND_SHARE_THRESHOLD:g} within {CROPLAND_SHARE_RADIUS_M:g} m)",
        f"{len(changed):,} of {len(out):,} ({len(changed) / len(out) * 100 if len(out) else 0:.1f}%) "
        f"would move from wildfire to agricultural burning -- CROPLAND_SHARE_FIX_ENABLED is OFF; nothing was changed",
        "",
        "By WorldCover class:",
        *(f"  {code} ({'shrubland' if code == 20 else 'grassland'}): {n:,}" for code, n in by_class.items()),
        "",
        "By state (top 15):",
        *(f"  {name:<32}{n:>7,}" for name, n in by_state.head(15).items()),
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    rows = load_wildfire_shrub_grass()
    out = compute_impact(rows)
    print(report(out))
    out_path = ROOT_DIR / "training" / "data" / "cropland_share_impact_report.csv"
    out.to_csv(out_path, index=False)
    print(f"\nPer-detection detail: {out_path}")
