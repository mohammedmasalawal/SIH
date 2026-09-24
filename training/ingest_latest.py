"""Live ingestion: pull recent FIRMS NRT detections for India, label them with the
current rules, append them to the live store, and refresh the map.

    python -m training.ingest_latest                     # last 2 days (UTC), the scheduled run
    python -m training.ingest_latest --days 5
    python -m training.ingest_latest --start 2026-09-01 --end 2026-09-23   # backfill

Each run:
  1. pulls VIIRS SNPP / NOAA-20 / NOAA-21 NRT over INDIA_BBOX and clips to India's
     real boundary (backend.ingestion.firms_client);
  2. drops duplicates on (latitude, longitude, acq_date, acq_time, satellite) --
     within the pull and against everything already stored -- so re-running the
     same window adds nothing;
  3. computes every context column with build_labels.add_context_features (the
     same cached OSM / GEM / coal-mine / WorldCover inputs as the historical build);
  4. computes recurrence_count and is_anomalous against the historical + live store,
     from earlier detections only (never later ones);
  5. applies rules.apply_rules unchanged;
  6. writes alerts_latest.csv (this run's new alerts of every type in
     training/alerts.py -- empty when nothing new) and appends them to alerts_history.csv,
     then appends to one live parquet per month under LIVE_DIR, then repacks the map
     (historical + live).

The historical NATIONAL_LABELED_PARQUETS are read, never written.
"""

from __future__ import annotations

import argparse
import os
import time
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from backend.classification.rules import apply_rules
from backend.config import (
    ALERTS_LATEST_PATH,
    CONTRACT_COLUMNS,
    INDIA_BOUNDARY_PATH,
    LIVE_DIR,
    LIVE_PARTITION_PREFIX,
    MAP_POINTS_META_PATH,
    MAP_POINTS_PATH,
    NATIONAL_FACILITIES_PATH,
    NATIONAL_INDUSTRIAL_CONTEXT_PATH,
    NATIONAL_LABELED_PARQUETS,
    NATIONAL_WORLDCOVER_DIR,
    RECURRENCE_LOOKBACK_DAYS,
)
from backend.ingestion.firms_client import INDIA_BBOX, NRT_SATELLITE_SOURCES, fetch_firms_multi
from training.alerts import (
    ALERT_TYPES,
    build_alerts,
    complete_days,
    detection_keys,
    drop_known,
    load_infrastructure_sites,
    read_alert_history,
)
from training.build_labels import add_context_features
from training.pack_map_points import map_source_parquets, pack
from training.spatial_features import add_causal_recurrence_features, anomaly_baselines, grid_keys

KEY_COLUMNS = ["latitude", "longitude", "acq_date", "acq_time", "satellite"]
_HISTORY_COLUMNS = ["latitude", "longitude", "acq_date", "acq_time", "satellite", "daynight", "frp"]
_STORE_COLUMNS = [*_HISTORY_COLUMNS, "label"]
LOCK_STALE_SECONDS = 6 * 3600

Fetcher = Callable[[date, date, Path], pd.DataFrame]


def fetch_nrt_india(start: date, end: date, raw_path: Path) -> pd.DataFrame:
    """All three VIIRS satellites, NRT sources only, clipped to India's boundary."""
    return fetch_firms_multi(
        INDIA_BBOX, start, end, raw_path,
        satellite_sources=NRT_SATELLITE_SOURCES, clip_boundary=INDIA_BOUNDARY_PATH,
    )


def live_partition_path(live_dir: Path, month: str) -> Path:
    return Path(live_dir) / f"{LIVE_PARTITION_PREFIX}{month}.parquet"


def _live_partitions(live_dir: Path) -> list[Path]:
    return sorted(Path(live_dir).glob(f"{LIVE_PARTITION_PREFIX}*.parquet"))


def load_history(history_paths: list[Path], live_dir: Path) -> pd.DataFrame:
    frames = [pd.read_parquet(path, columns=_STORE_COLUMNS) for path in [*history_paths, *_live_partitions(live_dir)]]
    if not frames:
        return pd.DataFrame(columns=_STORE_COLUMNS)
    history = pd.concat(frames, ignore_index=True)
    history["acq_date"] = history["acq_date"].astype(str).str[:10]
    return history


def _normalise_raw(raw: pd.DataFrame) -> pd.DataFrame:
    frame = raw.copy()
    frame["acq_date"] = frame["acq_date"].astype(str).str[:10]
    frame["acq_time"] = frame["acq_time"].astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(4)
    for column in ("satellite", "confidence", "daynight"):
        if column in frame:
            frame[column] = frame[column].astype(str)
    return frame


def publish_alerts(
    alerts: pd.DataFrame, latest_path: Path, history_path: Path, *, only_new: bool = True
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Write alerts_latest.csv and append unseen alerts to the append-only history.
    Returns (latest, newly added). only_new=False (re-evaluation) writes every alert
    evaluated into latest, not just the ones history hasn't seen."""
    history = read_alert_history(history_path)
    new = drop_known(alerts, history)
    latest = new if only_new else alerts
    _write_atomic(latest, Path(latest_path))
    if not new.empty:
        _write_atomic(pd.concat([history, new], ignore_index=True) if not history.empty else new, Path(history_path))
    return latest, new


def _write_atomic(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    if path.suffix == ".parquet":
        frame.to_parquet(tmp, index=False)
    else:
        frame.to_csv(tmp, index=False)
    os.replace(tmp, path)


def append_to_partitions(labeled: pd.DataFrame, live_dir: Path) -> dict[str, int]:
    """Append rows to their month's live parquet, existing rows first and in their
    original order (so a stored row's position -- and its map row id -- never moves).
    Returns {month: rows added}."""
    added = {}
    for month, rows in labeled.groupby(labeled["acq_date"].str[:7]):
        path = live_partition_path(live_dir, month)
        new_rows = rows[CONTRACT_COLUMNS]
        if path.exists():
            existing = pd.read_parquet(path)
            new_rows = new_rows[~detection_keys(new_rows).isin(set(detection_keys(existing)))]
            combined = pd.concat([existing, new_rows], ignore_index=True)
        else:
            combined = new_rows.reset_index(drop=True)
        if new_rows.empty:
            continue
        _write_atomic(combined, path)
        added[month] = len(new_rows)
    return added


@dataclass
class RunSummary:
    start: date
    end: date
    fetched: int = 0
    duplicates_skipped: int = 0
    new_detections: int = 0
    label_mix: dict[str, int] = field(default_factory=dict)
    alerts: int = 0
    alerts_by_type: dict[str, int] = field(default_factory=dict)
    partitions: dict[str, int] = field(default_factory=dict)
    map_repacked: bool = False
    map_points: int | None = None
    seconds: float = 0.0

    def report(self) -> str:
        lines = [
            f"FIRMS NRT {self.start}..{self.end}: {self.fetched:,} detections fetched (India, de-duplicated)",
            f"  already stored (skipped): {self.duplicates_skipped:,}",
            f"  new detections:           {self.new_detections:,}",
        ]
        for label, n in sorted(self.label_mix.items(), key=lambda kv: -kv[1]):
            lines.append(f"    {label:<22}{n:>8,}")
        lines.append(f"  new alerts: {self.alerts:,}")
        for alert_type in ALERT_TYPES:
            lines.append(f"    {alert_type:<26}{self.alerts_by_type.get(alert_type, 0):>6,}")
        if self.partitions:
            lines.append("  appended: " + ", ".join(f"{m} +{n:,}" for m, n in sorted(self.partitions.items())))
        if self.map_repacked:
            lines.append(f"  map repacked: {self.map_points:,} points")
        lines.append(f"  took {self.seconds:.0f} s")
        return "\n".join(lines)


def run_ingest(
    start: date,
    end: date,
    *,
    fetch: Fetcher = fetch_nrt_india,
    history_paths: list[Path] | None = None,
    live_dir: Path = LIVE_DIR,
    industrial_path: Path = NATIONAL_INDUSTRIAL_CONTEXT_PATH,
    facilities_path: Path = NATIONAL_FACILITIES_PATH,
    worldcover: Path | None = NATIONAL_WORLDCOVER_DIR,
    alerts_path: Path = ALERTS_LATEST_PATH,
    alerts_history_path: Path | None = None,
    lookback_days: int | None = RECURRENCE_LOOKBACK_DAYS,
    repack_map: bool = True,
    map_points_path: Path = MAP_POINTS_PATH,
    map_meta_path: Path = MAP_POINTS_META_PATH,
    today: date | None = None,
) -> tuple[RunSummary, pd.DataFrame]:
    """Returns the run summary and the newly labeled rows (with baseline columns)."""
    started = time.perf_counter()
    today = today or datetime.now(timezone.utc).date()
    history_paths = list(NATIONAL_LABELED_PARQUETS if history_paths is None else history_paths)
    live_dir = Path(live_dir)
    summary = RunSummary(start=start, end=end)

    raw = _normalise_raw(fetch(start, end, live_dir / "raw" / "firms_nrt_latest.parquet"))
    raw = raw[(raw["acq_date"] >= start.isoformat()) & (raw["acq_date"] <= end.isoformat())]
    raw = raw.loc[~detection_keys(raw).duplicated()].reset_index(drop=True)
    summary.fetched = len(raw)

    history = load_history(history_paths, live_dir)
    stored = detection_keys(history[history["acq_date"].between(start.isoformat(), end.isoformat())])
    is_new = ~detection_keys(raw).isin(set(stored))
    summary.duplicates_skipped = int((~is_new).sum())
    new = raw[is_new].reset_index(drop=True)
    summary.new_detections = len(new)

    labeled = pd.DataFrame()
    if not new.empty:
        labeled = add_context_features(
            new, industrial_path, facilities_path, worldcover_raster=worldcover
        ).reset_index(drop=True)

        # History first, this batch last: features for the batch come only from
        # detections dated on/before each row (recurrence) or strictly before (anomaly).
        combined = pd.concat([history, labeled[_HISTORY_COLUMNS]], ignore_index=True)
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

        labeled = apply_rules(labeled)
        for column in CONTRACT_COLUMNS:
            if column not in labeled:
                labeled[column] = None
        summary.label_mix = labeled["label"].value_counts().to_dict()

    # Alerts before the store: if a crash lands between the two, the re-run still
    # sees these rows as new and rewrites the alerts. Large-fire events are judged on
    # every complete day in the window, even when this run brought nothing new.
    store = pd.concat([history, labeled[_STORE_COLUMNS]], ignore_index=True) if not labeled.empty else history
    alerts = build_alerts(
        labeled, store, load_infrastructure_sites(industrial_path, facilities_path), complete_days(start, end, today)
    )
    history_path = Path(alerts_history_path) if alerts_history_path else Path(alerts_path).with_name("alerts_history.csv")
    _, new_alerts = publish_alerts(alerts, Path(alerts_path), history_path)
    summary.alerts = len(new_alerts)
    summary.alerts_by_type = new_alerts["alert_type"].value_counts().to_dict()

    if not labeled.empty:
        summary.partitions = append_to_partitions(labeled, live_dir)

    if repack_map and _map_is_stale(live_dir, Path(map_meta_path)):
        meta = pack(map_source_parquets(live_dir, history_paths), Path(map_points_path), Path(map_meta_path))
        summary.map_repacked, summary.map_points = True, meta["count"]

    summary.seconds = time.perf_counter() - started
    return summary, labeled


def reevaluate_alerts(
    start: date,
    end: date,
    *,
    history_paths: list[Path] | None = None,
    live_dir: Path = LIVE_DIR,
    industrial_path: Path = NATIONAL_INDUSTRIAL_CONTEXT_PATH,
    facilities_path: Path = NATIONAL_FACILITIES_PATH,
    alerts_path: Path = ALERTS_LATEST_PATH,
    today: date | None = None,
) -> pd.DataFrame:
    """Evaluate every alert type over live detections already stored for [start, end]
    -- after a threshold or alert-type change, without re-fetching anything. Anomaly
    baselines are recomputed from strictly earlier days, so they match what ingestion
    computed. alerts_latest.csv gets every alert evaluated; the history gets the new ones."""
    today = today or datetime.now(timezone.utc).date()
    history_paths = list(NATIONAL_LABELED_PARQUETS if history_paths is None else history_paths)
    partitions = _live_partitions(live_dir)
    if not partitions:
        raise SystemExit(f"no live partitions in {live_dir}")
    live = pd.concat([pd.read_parquet(p) for p in partitions], ignore_index=True)
    live["acq_date"] = live["acq_date"].astype(str).str[:10]
    historical = pd.concat([pd.read_parquet(p, columns=_STORE_COLUMNS) for p in history_paths], ignore_index=True)
    historical["acq_date"] = historical["acq_date"].astype(str).str[:10]

    in_window = live["acq_date"].between(start.isoformat(), end.isoformat()).to_numpy()
    combined = pd.concat([historical, live[_STORE_COLUMNS]], ignore_index=True)
    targets = pd.Series(np.r_[np.zeros(len(historical), bool), in_window], index=combined.index)
    grid_lat, grid_lon = grid_keys(combined)
    baselines = anomaly_baselines(combined, grid_lat, grid_lon, targets=targets)
    batch = live[in_window].reset_index(drop=True)
    batch["baseline_frp"] = baselines.loc[targets.to_numpy(), "baseline_frp"].to_numpy()
    batch["prior_active_days"] = baselines.loc[targets.to_numpy(), "prior_active_days"].to_numpy()

    alerts = build_alerts(
        batch, combined, load_infrastructure_sites(industrial_path, facilities_path), complete_days(start, end, today)
    )
    publish_alerts(alerts, Path(alerts_path), Path(alerts_path).with_name("alerts_history.csv"), only_new=False)
    return alerts


def _map_is_stale(live_dir: Path, meta_path: Path) -> bool:
    partitions = _live_partitions(live_dir)
    if not partitions:
        return False
    if not meta_path.exists():
        return True
    return max(p.stat().st_mtime for p in partitions) > meta_path.stat().st_mtime


class RunLock:
    """Stops two scheduled runs overlapping; a lock older than LOCK_STALE_SECONDS is
    assumed to be left over from a crashed run and taken over."""

    def __init__(self, path: Path):
        self.path = Path(path)

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists() and time.time() - self.path.stat().st_mtime > LOCK_STALE_SECONDS:
            self.path.unlink()
        try:
            fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            raise SystemExit(f"another ingest run holds {self.path}; exiting") from None
        os.write(fd, str(os.getpid()).encode())
        os.close(fd)
        return self

    def __exit__(self, *exc):
        self.path.unlink(missing_ok=True)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--days", type=int, default=2, help="pull the last N days, ending today (UTC); default 2")
    parser.add_argument("--start", type=date.fromisoformat, help="backfill start (overrides --days)")
    parser.add_argument("--end", type=date.fromisoformat, help="backfill end, default today (UTC)")
    parser.add_argument("--lookback-days", type=int, default=RECURRENCE_LOOKBACK_DAYS,
                        help="recurrence look-back window; 0 = all history")
    parser.add_argument("--no-map", action="store_true", help="skip repacking the map points")
    parser.add_argument("--alerts-only", action="store_true",
                        help="re-evaluate alerts over live detections already stored for the window (no fetch)")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    today = datetime.now(timezone.utc).date()
    end = args.end or today
    start = args.start or end - timedelta(days=args.days - 1)
    with RunLock(LIVE_DIR / "ingest.lock"):
        if args.alerts_only:
            alerts = reevaluate_alerts(start, end)
            print(f"Alerts for stored detections {start}..{end}: {len(alerts):,}")
            for alert_type in ALERT_TYPES:
                print(f"  {alert_type:<26}{int((alerts['alert_type'] == alert_type).sum()):>6,}")
        else:
            summary, _ = run_ingest(
                start, end, lookback_days=args.lookback_days or None, repack_map=not args.no_map
            )
            print(summary.report())
