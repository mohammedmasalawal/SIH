from __future__ import annotations

import hashlib
from datetime import date

import pandas as pd
import pytest

from backend.config import ROOT_DIR
from backend.map_store import MapStore
from training.ingest_latest import RunLock, append_to_partitions, detection_keys, live_partition_path, run_ingest
from training.spatial_features import add_causal_recurrence_features, grid_keys

SAMPLE = ROOT_DIR / "data" / "sample"
SITE = (22.3000, 69.8500)  # 163 m from the sample cement_plant -> industrial when recurring
FAR = (22.9000, 70.9000)


def _raw(lat, lon, day, time="0130", sat="N", frp=5.0, daynight="N"):
    return {"latitude": lat, "longitude": lon, "acq_date": day, "acq_time": time, "satellite": sat,
            "bright_ti4": 330.0, "bright_ti5": 290.0, "frp": frp, "confidence": "n", "daynight": daynight}


@pytest.fixture
def store(tmp_path):
    """One historical file with 6 quiet nights at SITE (FRP 5) -- enough prior days
    for an anomaly baseline -- plus the paths run_ingest needs."""
    history = pd.DataFrame([
        _raw(*SITE, f"2026-08-{d:02d}", frp=5.0) | {"label": "industrial", "is_anomalous": False}
        for d in (20, 22, 24, 26, 28, 30)
    ])
    hist_path = tmp_path / "historical.parquet"
    history.to_parquet(hist_path, index=False)
    return {
        "history_paths": [hist_path],
        "live_dir": tmp_path / "live",
        "industrial_path": SAMPLE / "industrial_sample.geojson",
        "facilities_path": SAMPLE / "facilities_sample.geojson",
        "worldcover": None,
        "alerts_path": tmp_path / "live" / "alerts_latest.csv",
        "map_points_path": tmp_path / "map" / "points.bin",
        "map_meta_path": tmp_path / "map" / "points.json",
    }


def _fetcher(rows):
    def fetch(start, end, raw_path):
        return pd.DataFrame(rows)
    return fetch


PULL = [
    _raw(*SITE, "2026-09-01", frp=40.0),            # 8x the site's normal -> alert
    _raw(*SITE, "2026-09-01", frp=40.0),            # exact duplicate in the same pull
    _raw(*SITE, "2026-09-02", time=745, sat="1"),   # int acq_time, as FIRMS CSVs parse it
    _raw(*FAR, "2026-09-02"),
]


def _digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_first_run_labels_appends_and_alerts(store):
    before = _digest(store["history_paths"][0])
    summary, labeled = run_ingest(date(2026, 9, 1), date(2026, 9, 2), fetch=_fetcher(PULL), **store)

    assert summary.fetched == 3 and summary.new_detections == 3 and summary.duplicates_skipped == 0
    assert _digest(store["history_paths"][0]) == before  # historical store untouched

    site_first = labeled[(labeled["acq_date"] == "2026-09-01")].iloc[0]
    assert site_first["recurrence_count"] == 7  # 6 August nights + its own day
    assert site_first["is_anomalous"] is True or site_first["is_anomalous"] == True  # noqa: E712
    assert site_first["baseline_frp"] == pytest.approx(5.0)
    assert site_first["label"] == "industrial"

    alerts = pd.read_csv(store["alerts_path"])
    assert len(alerts) == summary.alerts == 1
    alert = alerts.iloc[0]
    assert alert["site_normal_frp"] == pytest.approx(5.0) and alert["frp_vs_normal"] == pytest.approx(8.0)
    assert alert["facility_type"] == "cement_plant"
    assert alert["acq_date"] == "2026-09-01"

    partition = pd.read_parquet(live_partition_path(store["live_dir"], "2026-09"))
    assert len(partition) == 3
    assert set(partition["acq_time"]) == {"0130", "0745"}
    assert summary.map_repacked and summary.map_points == 6 + 3


def test_rerun_is_idempotent(store):
    run_ingest(date(2026, 9, 1), date(2026, 9, 2), fetch=_fetcher(PULL), **store)
    partition_path = live_partition_path(store["live_dir"], "2026-09")
    first = _digest(partition_path)

    summary, labeled = run_ingest(date(2026, 9, 1), date(2026, 9, 2), fetch=_fetcher(PULL), **store)
    assert summary.new_detections == 0 and summary.duplicates_skipped == 3
    assert labeled.empty and summary.alerts == 0
    assert _digest(partition_path) == first
    assert pd.read_csv(store["alerts_path"]).empty  # latest = this run's alerts only
    assert len(pd.read_csv(store["alerts_path"].with_name("alerts_history.csv"))) == 1  # but none lost
    assert not summary.map_repacked  # nothing changed, pack already current


def test_later_run_appends_without_reordering(store):
    run_ingest(date(2026, 9, 1), date(2026, 9, 2), fetch=_fetcher(PULL), **store)
    partition_path = live_partition_path(store["live_dir"], "2026-09")
    before = pd.read_parquet(partition_path)

    late = [*PULL, _raw(*FAR, "2026-09-03", time="1200")]
    summary, _ = run_ingest(date(2026, 9, 1), date(2026, 9, 3), fetch=_fetcher(late), **store)
    after = pd.read_parquet(partition_path)
    assert summary.new_detections == 1
    pd.testing.assert_frame_equal(after.iloc[:3].reset_index(drop=True), before)  # stored rows keep their positions
    assert after.iloc[3]["acq_date"] == "2026-09-03"


def test_live_rows_resolve_through_the_map(store):
    run_ingest(date(2026, 9, 1), date(2026, 9, 2), fetch=_fetcher(PULL), **store)
    map_store = MapStore(store["map_points_path"], store["map_meta_path"])
    assert map_store.meta()["max_date"] == "2026-09-02"
    live_first_row = map_store.detection(6)  # rows 0-5 are the historical file
    assert live_first_row["acq_date"] == "2026-09-01" and live_first_row["label"] == "industrial"


def test_causal_recurrence_ignores_later_days_and_respects_lookback():
    frame = pd.DataFrame({"latitude": [22.3] * 4, "longitude": [69.85] * 4,
                          "acq_date": ["2026-01-01", "2026-06-01", "2026-06-10", "2026-06-20"]})
    grid_lat, grid_lon = grid_keys(frame)
    targets = pd.Series([False, False, True, True])
    all_history = add_causal_recurrence_features(frame, grid_lat, grid_lon, targets)
    assert all_history.loc[2, "recurrence_count"] == 3  # Jan, Jun 1, own day -- not Jun 20
    assert all_history.loc[3, "recurrence_count"] == 4
    assert all_history.loc[2, "first_seen"] == "2026-01-01" and all_history.loc[2, "last_seen"] == "2026-06-10"

    windowed = add_causal_recurrence_features(frame, grid_lat, grid_lon, targets, lookback_days=90)
    assert windowed.loc[2, "recurrence_count"] == 2  # January is outside the 90-day window
    assert windowed.loc[2, "first_seen"] == "2026-06-01"


def test_detection_keys_match_across_stored_dtypes():
    a = pd.DataFrame([_raw(22.1, 69.1, "2026-09-01", time=626)])
    b = pd.DataFrame([_raw(22.100001, 69.1, "2026-09-01", time="0626")])
    assert detection_keys(a).iloc[0] == detection_keys(b).iloc[0]


def test_append_skips_rows_already_in_partition(tmp_path):
    from backend.config import CONTRACT_COLUMNS

    row = {column: None for column in CONTRACT_COLUMNS} | _raw(22.1, 69.1, "2026-09-01") | {"label": "unknown"}
    frame = pd.DataFrame([row])
    assert append_to_partitions(frame, tmp_path) == {"2026-09": 1}
    assert append_to_partitions(frame, tmp_path) == {}
    assert len(pd.read_parquet(live_partition_path(tmp_path, "2026-09"))) == 1


def test_run_lock_blocks_overlap(tmp_path):
    with RunLock(tmp_path / "ingest.lock"):
        with pytest.raises(SystemExit):
            with RunLock(tmp_path / "ingest.lock"):
                pass
    with RunLock(tmp_path / "ingest.lock"):  # released after the first run
        pass
