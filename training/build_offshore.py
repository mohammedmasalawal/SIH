"""Offshore detections: select them from unclipped FIRMS pulls, add context, label them.

    python -m training.build_offshore --raw data/raw/unclipped/india_*.parquet

Bulk mode rebuilds the whole offshore history in one file (OFFSHORE_HISTORY_PATH), so
recurrence_count is the number of distinct active days of a ~375 m cell across that file's
whole span, both directions -- the historical convention (active days within the file), but a
year-long file rather than the 1-3 month period files the onshore data uses. Live mode
(training/ingest_latest.py) counts a cell's active days on or before each detection within
RECURRENCE_LOOKBACK_DAYS instead, like every live row.

Onshore data is never read or written here: the offshore parquets are separate files with
the CONTRACT_COLUMNS plus region / offshore_zone / dist_offshore_km.
"""

from __future__ import annotations

import argparse
import glob
from pathlib import Path

import numpy as np
import pandas as pd

from backend.classification.rules import apply_rules
from backend.config import (
    CONTRACT_COLUMNS,
    NATIONAL_FACILITIES_PATH,
    NATIONAL_INDUSTRIAL_CONTEXT_PATH,
    NATIONAL_WORLDCOVER_DIR,
    OFFSHORE_EXTRA_COLUMNS,
    OFFSHORE_HISTORY_PATH,
    OFFSHORE_REGION,
)
from backend.ingestion.offshore import offshore_columns
from training.build_labels import add_context_features
from training.spatial_features import (
    add_causal_recurrence_features,
    add_recurrence_features,
    anomaly_baselines,
    compute_is_anomalous,
    grid_keys,
)

OFFSHORE_COLUMNS = [*CONTRACT_COLUMNS, *OFFSHORE_EXTRA_COLUMNS]
_KEY = ["latitude", "longitude", "acq_date", "acq_time", "satellite"]


def normalise_raw(raw: pd.DataFrame) -> pd.DataFrame:
    """Same normalisation live ingestion applies to a raw FIRMS frame."""
    frame = raw.copy()
    frame["acq_date"] = frame["acq_date"].astype(str).str[:10]
    frame["acq_time"] = frame["acq_time"].astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(4)
    for column in ("satellite", "confidence", "daynight"):
        if column in frame:
            frame[column] = frame[column].astype(str)
    return frame


def select_offshore(raw: pd.DataFrame, boundary_path: Path | None = None) -> pd.DataFrame:
    """The offshore rows of an unclipped FIRMS frame, with region / offshore_zone /
    dist_offshore_km, de-duplicated on the detection key."""
    frame = normalise_raw(raw)
    frame = frame.drop_duplicates(subset=_KEY).reset_index(drop=True)
    columns = offshore_columns(frame, boundary_path)
    keep = columns["offshore_zone"].notna().to_numpy()
    rows = pd.concat([frame[keep], columns[keep]], axis=1).reset_index(drop=True)
    rows["region"] = OFFSHORE_REGION
    return rows


def add_zone_columns(rows: pd.DataFrame, boundary_path: Path | None = None) -> pd.DataFrame:
    """region / offshore_zone / dist_offshore_km for rows the fetch already tagged offshore."""
    out = rows.copy()
    columns = offshore_columns(out, boundary_path)
    out["offshore_zone"] = columns["offshore_zone"].to_numpy()
    out["dist_offshore_km"] = columns["dist_offshore_km"].to_numpy()
    out["region"] = OFFSHORE_REGION
    return out


def _context(rows: pd.DataFrame, industrial_path, facilities_path, worldcover) -> pd.DataFrame:
    """add_context_features for offshore rows. WorldCover has no data over open sea (tiles are
    absent or all zero there), which the lookup leaves null -- landcover_class is never
    fabricated -- and the offshore rule doesn't read it."""
    return add_context_features(
        rows, industrial_path, facilities_path, worldcover_raster=worldcover
    ).reset_index(drop=True)


def label_offshore_history(
    rows: pd.DataFrame,
    *,
    industrial_path=NATIONAL_INDUSTRIAL_CONTEXT_PATH,
    facilities_path=NATIONAL_FACILITIES_PATH,
    worldcover=NATIONAL_WORLDCOVER_DIR,
) -> pd.DataFrame:
    """Bulk labelling of offshore rows (already selected): recurrence over the frame."""
    labeled = _context(rows, industrial_path, facilities_path, worldcover)
    grid_lat, grid_lon = grid_keys(labeled)
    labeled = add_recurrence_features(labeled, grid_lat, grid_lon)
    labeled["is_anomalous"] = compute_is_anomalous(labeled, grid_lat, grid_lon).values
    labeled = apply_rules(labeled)
    return _finish(labeled)


def label_offshore_live(
    new_rows: pd.DataFrame,
    history: pd.DataFrame,
    *,
    lookback_days: int | None,
    industrial_path=NATIONAL_INDUSTRIAL_CONTEXT_PATH,
    facilities_path=NATIONAL_FACILITIES_PATH,
    worldcover=NATIONAL_WORLDCOVER_DIR,
) -> pd.DataFrame:
    """Live labelling: features come only from earlier detections (recurrence counts a
    cell's active days on or before the row's date; anomaly baselines use strictly earlier
    days), history = every stored offshore detection (latitude, longitude, acq_date, ...)."""
    labeled = _context(new_rows, industrial_path, facilities_path, worldcover)
    cols = ["latitude", "longitude", "acq_date", "acq_time", "satellite", "daynight", "frp"]
    combined = pd.concat([history[cols], labeled[cols]], ignore_index=True)
    targets = pd.Series(np.r_[np.zeros(len(history), bool), np.ones(len(labeled), bool)], index=combined.index)
    grid_lat, grid_lon = grid_keys(combined)
    recurrence = add_causal_recurrence_features(
        combined[["acq_date"]], grid_lat, grid_lon, targets, lookback_days=lookback_days
    )
    baselines = anomaly_baselines(combined, grid_lat, grid_lon, targets=targets)
    batch = targets.to_numpy()
    labeled["recurrence_count"] = recurrence.loc[batch, "recurrence_count"].astype(int).to_numpy()
    labeled["first_seen"] = recurrence.loc[batch, "first_seen"].to_numpy()
    labeled["last_seen"] = recurrence.loc[batch, "last_seen"].to_numpy()
    labeled["is_anomalous"] = baselines.loc[batch, "is_anomalous"].to_numpy()
    labeled["baseline_frp"] = baselines.loc[batch, "baseline_frp"].to_numpy()
    labeled["prior_active_days"] = baselines.loc[batch, "prior_active_days"].to_numpy()
    return _finish(apply_rules(labeled), keep_baselines=True)


def _finish(labeled: pd.DataFrame, keep_baselines: bool = False) -> pd.DataFrame:
    for column in OFFSHORE_COLUMNS:
        if column not in labeled:
            labeled[column] = None
    # landcover is null everywhere at sea; a typed column keeps parquet schemas mergeable
    labeled["landcover_class"] = labeled["landcover_class"].astype("string")
    extra = [c for c in ("baseline_frp", "prior_active_days") if keep_baselines and c in labeled]
    return labeled[[*OFFSHORE_COLUMNS, *extra]]


def build_history(raw_paths: list[Path], out_path: Path = OFFSHORE_HISTORY_PATH) -> pd.DataFrame:
    raw = pd.concat([pd.read_parquet(p) for p in raw_paths], ignore_index=True)
    rows = select_offshore(raw)
    labeled = label_offshore_history(rows)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    labeled[OFFSHORE_COLUMNS].to_parquet(out_path, index=False)
    return labeled


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--raw", nargs="+", required=True, help="unclipped FIRMS parquet files (globs ok)")
    parser.add_argument("--out", type=Path, default=OFFSHORE_HISTORY_PATH)
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    paths = sorted(Path(p) for pattern in args.raw for p in glob.glob(pattern))
    result = build_history(paths, args.out)
    print(f"{len(result):,} offshore detections from {len(paths)} files -> {args.out}")
    print(result["label"].value_counts().to_string())
