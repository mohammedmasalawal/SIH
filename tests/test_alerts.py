from __future__ import annotations

from datetime import date

import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import Point

from backend.config import NATIONAL_PROJECTED_CRS
from training.alerts import (
    ALERT_COLUMNS,
    complete_days,
    drop_known,
    fire_near_infrastructure_alerts,
    large_fire_event_alerts,
    load_infrastructure_sites,
    new_unmapped_source_alerts,
    read_alert_history,
)
from training.ingest_latest import publish_alerts

KM_LAT = 1 / 111.32  # degrees of latitude per km


def _det(lat, lon, day, label="agricultural burning", time="0130", **extra):
    return {"latitude": lat, "longitude": lon, "acq_date": day, "acq_time": time, "satellite": "N",
            "daynight": "N", "frp": 3.0, "label": label, "is_anomalous": False, "recurrence_count": 1,
            "dist_to_industrial_m": 50_000.0, "dist_to_heat_industry_m": 50_000.0,
            "dist_to_flare_capable_m": 50_000.0, "dist_to_coal_mine_m": 50_000.0,
            "dist_to_brick_kiln_m": 50_000.0} | extra


@pytest.fixture
def sites():
    plant = gpd.GeoDataFrame(
        {"facility_name": ["Test CCGT"], "facility_type": ["power plant (oil/gas)"], "source": ["GEM"]},
        geometry=[Point(75.0, 25.0)], crs="EPSG:4326",
    )
    return plant.to_crs(NATIONAL_PROJECTED_CRS)


def test_fire_near_infrastructure_one_off_detections_any_label(sites):
    batch = pd.DataFrame([
        _det(25.0 + 1.0 * KM_LAT, 75.0, "2026-09-05", label="industrial"),         # one-off, 1 km -> alert
        _det(25.0 + 0.5 * KM_LAT, 75.0, "2026-09-05", recurrence_count=4),         # recurring site -> no
        _det(25.0 + 3.0 * KM_LAT, 75.0, "2026-09-05", label="wildfire"),           # 3 km -> too far
    ])
    alerts = fire_near_infrastructure_alerts(batch, sites)
    assert len(alerts) == 1
    alert = alerts.iloc[0]
    assert alert["alert_type"] == "fire_near_infrastructure" and alert["label"] == "industrial"
    assert alert["facility_name"] == "Test CCGT" and alert["facility_type"] == "power plant (oil/gas) · GEM"
    assert alert["facility_distance_m"] == pytest.approx(1000, rel=0.03)  # EPSG:7755 scale error at 25N ~2%
    assert list(alerts.columns) == ALERT_COLUMNS


def test_infrastructure_sites_exclude_renewables(tmp_path):
    gem = gpd.GeoDataFrame(
        {"facility_type": ["cement_plant", "oil_gas_power_plant", "coal_mine"], "facility_name": ["C", "P", "M"]},
        geometry=[Point(75, 25), Point(76, 25), Point(77, 25)], crs="EPSG:4326",
    )
    osm = gpd.GeoDataFrame(
        {"power": ["plant", "plant", "plant", "plant", "plant", None, None],
         "plant:source": ["solar", "wind", "hydro", "coal", "coal;gas", None, None],
         "industrial": [None, None, None, None, None, "refinery", "oil"],
         "name": ["S", "W", "H", "Coal", "Mixed", "Ref", "Oil depot"]},
        geometry=[Point(78, 25 + i * 0.1) for i in range(7)], crs="EPSG:4326",
    )
    gem_path, osm_path = tmp_path / "gem.geojson", tmp_path / "osm.geojson"
    gem.to_file(gem_path)
    osm.to_file(osm_path)
    names = set(load_infrastructure_sites(osm_path, gem_path)["facility_name"])
    assert names == {"C", "P", "Coal", "Mixed", "Ref"}  # no solar/wind/hydro, no coal_mine, no industrial=oil


def _cell_days(lat, lon, days, **extra):
    return [_det(lat, lon, d, **extra) for d in days]


def test_new_unmapped_source_fires_once_when_a_quiet_cell_starts_recurring():
    sept = [f"2026-09-{d:02d}" for d in (1, 4, 8, 12, 15)]
    store = pd.DataFrame(
        _cell_days(21.0, 80.0, sept)                                        # new, unmapped -> alert
        + _cell_days(22.0, 80.0, ["2026-06-01", "2026-06-20", "2026-07-10", *sept])  # recurring before
        + _cell_days(23.0, 80.0, sept, dist_to_industrial_m=800.0)         # facility within 2 km
    )
    alerts = new_unmapped_source_alerts(store, store)
    assert len(alerts) == 1
    alert = alerts.iloc[0]
    assert alert["latitude"] == pytest.approx(21.0)
    assert alert["acq_date"] == "2026-09-15"  # the day it reached 5 active days
    assert alert["active_days_30d"] == 5 and alert["recurrence_count"] == 5
    assert alert["first_seen"] == "2026-09-01"


def test_new_unmapped_source_ignores_days_outside_the_window():
    spread = pd.DataFrame(_cell_days(21.0, 80.0, ["2026-07-01", "2026-07-20", "2026-08-10", "2026-08-30", "2026-09-20"]))
    assert new_unmapped_source_alerts(spread, spread).empty  # 5 days, but only 2 within any 30


def test_large_fire_event_clusters_crop_and_natural_fires_only():
    fires = [_det(20.0 + i * 0.2 * KM_LAT, 78.0, "2026-09-10", label="wildfire" if i < 8 else "agricultural burning")
             for i in range(12)]
    industrial = [_det(20.0 + i * 0.2 * KM_LAT, 78.01, "2026-09-10", label="industrial") for i in range(6)]
    scattered = [_det(10.0 + i, 75.0, "2026-09-10") for i in range(5)]
    other_day = [_det(20.0, 78.0 + i * 0.001, "2026-09-11") for i in range(9)]  # 9 -> below threshold
    store = pd.DataFrame(fires + industrial + scattered + other_day)
    alerts = large_fire_event_alerts(store, ["2026-09-10", "2026-09-11"])
    assert len(alerts) == 1
    event = alerts.iloc[0]
    assert event["event_count"] == 12 and event["dominant_class"] == "wildfire"  # industrial ones not counted
    assert event["acq_date"] == "2026-09-10"

    only_industrial = pd.DataFrame([_det(20.0 + i * 0.2 * KM_LAT, 78.0, "2026-09-10", label="industrial") for i in range(15)])
    assert large_fire_event_alerts(only_industrial, ["2026-09-10"]).empty


def test_large_fire_event_skips_persistently_active_cells():
    """A coal-seam fire: the same cells burned on 6 of the previous 30 days."""
    cells = [(20.0 + i * 0.2 * KM_LAT, 78.0) for i in range(12)]
    history = [_det(lat, lon, f"2026-09-{d:02d}", label="unknown") for lat, lon in cells for d in (1, 3, 5, 7, 9, 11)]
    today = [_det(lat, lon, "2026-09-20", label="wildfire") for lat, lon in cells]
    assert large_fire_event_alerts(pd.DataFrame(history + today), ["2026-09-20"]).empty
    five_days = [row for row in history if row["acq_date"] != "2026-09-11"]  # 5 prior days: not excluded
    assert len(large_fire_event_alerts(pd.DataFrame(five_days + today), ["2026-09-20"])) == 1


def test_rescore_replaces_only_the_window(tmp_path):
    history_path, latest_path = tmp_path / "alerts_history.csv", tmp_path / "alerts_latest.csv"
    def alert(alert_id, day):
        return {"alert_id": alert_id, "alert_type": "fire_near_infrastructure", "latitude": 20.0, "longitude": 78.0, "acq_date": day}
    pd.DataFrame([alert("old-in-window", "2026-09-05"), alert("outside", "2026-08-05")]).reindex(columns=ALERT_COLUMNS).to_csv(history_path, index=False)
    rescored = pd.DataFrame([alert("new-in-window", "2026-09-06")]).reindex(columns=ALERT_COLUMNS)
    publish_alerts(rescored, latest_path, history_path, rescore_window=(date(2026, 9, 1), date(2026, 9, 23)))
    assert set(pd.read_csv(history_path)["alert_id"]) == {"outside", "new-in-window"}
    assert list(pd.read_csv(latest_path)["alert_id"]) == ["new-in-window"]


def test_complete_days_excludes_today():
    assert complete_days(date(2026, 9, 22), date(2026, 9, 24), today=date(2026, 9, 24)) == ["2026-09-22", "2026-09-23"]
    assert complete_days(date(2026, 9, 24), date(2026, 9, 24), today=date(2026, 9, 24)) == []


def test_old_history_upgrades_and_known_alerts_drop(tmp_path):
    path = tmp_path / "alerts_history.csv"
    pd.DataFrame([{"latitude": 22.1, "longitude": 69.1, "acq_date": "2026-09-01", "acq_time": "0130",
                   "satellite": "N", "label": "industrial", "frp": 9.0}]).to_csv(path, index=False)
    history = read_alert_history(path)
    assert history.iloc[0]["alert_type"] == "industrial_anomaly"
    assert history.iloc[0]["alert_id"] == "industrial_anomaly|22.10000|69.10000|2026-09-01|0130|N"

    event = {"alert_id": "large_fire_event|2026-09-10|1|1", "alert_type": "large_fire_event",
             "latitude": 20.0, "longitude": 78.0, "acq_date": "2026-09-10"}
    history = pd.concat([history, pd.DataFrame([event])], ignore_index=True)
    moved = event | {"alert_id": "large_fire_event|2026-09-10|2|2", "latitude": 20.0 + 2 * KM_LAT}  # centre drifted 2 km
    elsewhere = event | {"alert_id": "large_fire_event|2026-09-10|9|9", "latitude": 21.0}
    candidates = pd.DataFrame([history.iloc[0].to_dict(), moved, elsewhere])
    fresh = drop_known(candidates, history)
    assert fresh["alert_id"].tolist() == ["large_fire_event|2026-09-10|9|9"]
