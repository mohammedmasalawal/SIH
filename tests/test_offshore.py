from __future__ import annotations

import hashlib
import io
from datetime import date

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import box

import backend.ingestion.offshore as offshore_module
from backend.classification.rules import _matched_labels, apply_offshore_rules, apply_rules
from backend.config import (
    GAS_FLARE_MIN_RECURRENCE,
    OFFSHORE_LABEL_SOURCE,
    OFFSHORE_MIN_DISTANCE_KM,
    OFFSHORE_MIN_RECURRENCE,
    OFFSHORE_ZONES,
    ROOT_DIR,
)
from backend.ingestion.firms_client import _clip_to_boundary
from backend.ingestion.offshore import offshore_columns, zone_of
from training.build_offshore import OFFSHORE_COLUMNS, add_zone_columns, label_offshore_history
from training.ingest_latest import live_partition_path, offshore_partition_path, run_ingest
from training.worldcover_index import WorldCoverTileIndex

SAMPLE = ROOT_DIR / "data" / "sample"
MUMBAI_HIGH = (71.3, 19.4)  # lon, lat -- inside the Mumbai High zone, ~130 km off a coast at 72.7E
KG_BASIN = (82.6, 16.55)


@pytest.fixture
def coast(tmp_path):
    """A synthetic 'India': land east of 72.5E between 18N and 21N (a coast 120+ km east of Mumbai
    High's centre), and land west of 82.0E / south of 17.5N for the KG side (its coast at 82.0E)."""
    path = tmp_path / "boundary.parquet"
    gpd.GeoDataFrame(
        {"name": ["west-coast", "east-coast"]},
        geometry=[box(72.5, 18.0, 74.0, 21.0), box(80.0, 14.0, 82.0, 18.0)],
        crs="EPSG:4326",
    ).to_parquet(path)
    return path


def _names(series):
    return [None if pd.isna(v) else v for v in series]


def _frame(points):
    return pd.DataFrame(points, columns=["longitude", "latitude"])


# --- the pinned constants -------------------------------------------------------------------

def test_zones_threshold_and_distance_floor_are_the_values_fixed_in_advance():
    assert OFFSHORE_ZONES == {"Mumbai High": (70.5, 18.5, 72.3, 20.5), "KG basin": (81.0, 14.5, 83.5, 17.5)}
    assert OFFSHORE_MIN_DISTANCE_KM == 10
    assert OFFSHORE_MIN_RECURRENCE == GAS_FLARE_MIN_RECURRENCE == 5  # the existing flare bar, not tuned
    assert OFFSHORE_LABEL_SOURCE == "offshore_persistent"


# --- which detections are offshore -----------------------------------------------------------

def test_offshore_needs_a_zone_and_10_km_beyond_the_boundary(coast):
    points = _frame([
        MUMBAI_HIGH,          # in the zone, ~140 km from the coast            -> offshore
        (72.35, 19.4),        # outside the zone box (east of 72.3), 15 km out  -> not (no zone)
        (72.2, 19.4),         # in the zone, 31 km from the coast               -> offshore
        (82.15, 16.5),        # in the KG zone, 16 km east of its coast (82.0E) -> offshore
        (82.6, 16.55),        # in the KG zone, 60 km from the coast            -> offshore
        (69.0, 19.4),         # far west, outside both zones                    -> not
        (73.0, 19.4),         # inside the boundary                             -> not
    ])
    out = offshore_columns(points, coast)
    assert _names(out["offshore_zone"])[:3] == ["Mumbai High", None, "Mumbai High"]
    assert out["offshore_zone"].iloc[3] == out["offshore_zone"].iloc[4] == "KG basin"
    assert out["dist_offshore_km"].iloc[3] == pytest.approx(16, abs=3)
    assert out["offshore_zone"].iloc[5:].isna().all()
    assert out["dist_offshore_km"].iloc[0] > 100 and out["dist_offshore_km"].iloc[2] == pytest.approx(31, abs=3)
    assert out["dist_offshore_km"].isna().tolist()[1] and out["dist_offshore_km"].isna().tolist()[6]


def test_points_inside_the_10_km_floor_are_not_offshore(coast):
    point = _frame([(72.29, 19.4)])  # inside the Mumbai High zone, next to its eastern edge
    # a coast at 72.38E is about 9.4 km away (0.09 deg of longitude at 19N): too close to count
    assert offshore_columns(point, _boundary(coast.parent, 72.38))["offshore_zone"].isna().all()
    # a coast at 72.5E is about 22 km away: offshore
    assert offshore_columns(point, _boundary(coast.parent, 72.5))["offshore_zone"].tolist() == ["Mumbai High"]


def _boundary(directory, west_edge):
    path = directory / f"coast_{west_edge}.parquet"
    gpd.GeoDataFrame({"name": ["x"]}, geometry=[box(west_edge, 18.0, 74.0, 21.0)], crs="EPSG:4326").to_parquet(path)
    return path


def test_zone_of_is_inclusive_on_the_edges():
    lon = pd.Series([70.5, 72.3, 70.49, 81.0, 83.5])
    lat = pd.Series([18.5, 20.5, 19.0, 14.5, 17.5])
    assert _names(zone_of(lon, lat)) == ["Mumbai High", "Mumbai High", None, "KG basin", "KG basin"]


def _raw_pull():
    rows = [
        (73.0, 19.4, "onshore-inside"), (MUMBAI_HIGH[0], MUMBAI_HIGH[1], "mumbai"), (72.35, 19.4, "dropped"),
        (81.0, 16.0, "kg-coast"), (KG_BASIN[0], KG_BASIN[1], "kg"), (75.0, 25.0, "far-land"),
    ]
    return pd.DataFrame({
        "longitude": [r[0] for r in rows], "latitude": [r[1] for r in rows], "tag": [r[2] for r in rows],
        "acq_date": "2026-09-01", "acq_time": "0130", "satellite": "N",
    })


def test_clip_keep_offshore_adds_only_zone_detections_and_leaves_onshore_rows_alone(coast):
    frame = _raw_pull()
    plain = _clip_to_boundary(frame, coast)
    kept = _clip_to_boundary(frame, coast, keep_offshore=True)

    assert "region" not in plain.columns  # the default path is exactly as before
    assert set(kept.loc[kept["region"] == "offshore", "tag"]) == {"mumbai", "kg"}
    onshore = kept[kept["region"] == "onshore"].drop(columns="region").reset_index(drop=True)
    pd.testing.assert_frame_equal(onshore, plain)  # same rows, same order, same values
    assert "dropped" not in set(kept["tag"])  # in no zone and outside the boundary: still clipped


# --- the rule ---------------------------------------------------------------------------------

def _offshore_rows(recurrences):
    return pd.DataFrame({
        "region": "offshore", "recurrence_count": recurrences,
        # land context that would trigger onshore rules if they applied
        "dist_to_flare_capable_m": 100.0, "dist_to_industrial_m": 0.0, "osm_industrial_tag": "yes",
        "landcover_class": "40", "daynight": "D", "frp": 1.0,
    })


def test_offshore_rule_flare_at_five_days_unknown_below():
    labeled = apply_rules(_offshore_rows([1, 4, 5, 6, 122]))
    assert labeled["label"].tolist() == ["unknown", "unknown", "gas flare", "gas flare", "gas flare"]
    assert labeled["label_source"].tolist() == ["unknown", "unknown", *[OFFSHORE_LABEL_SOURCE] * 3]


def test_offshore_rows_never_see_onshore_rules():
    # recurrence 3 inside an OSM industrial polygon on cropland: onshore this would be labelled;
    # offshore it is just a 3-day detection
    labeled = apply_rules(_offshore_rows([3]))
    assert labeled["label"].iloc[0] == "unknown" and labeled["label_source"].iloc[0] == "unknown"
    assert apply_offshore_rules(_offshore_rows([9]), adopted=False)["label"].tolist() == ["unknown"]


# --- no onshore label may change ------------------------------------------------------------------

def _onshore_rows():
    cases = [  # (dist heat, dist flare, dist industrial, osm tag, landcover, recurrence)
        (150.0, 20_000.0, 50.0, "yes", "50", 3), (None, 100.0, None, None, None, 6), (None, None, None, None, "40", 1),
        (None, None, 9_000.0, None, "10", 1), (None, None, None, None, "20", 2), (1_900.0, None, None, None, None, 2),
        (None, None, 5.0, "refinery", "50", 2), (None, None, None, None, None, 1), (None, None, 0.0, "yes", "40", 1),
    ]
    return pd.DataFrame({
        "dist_to_heat_industry_m": [c[0] for c in cases], "dist_to_flare_capable_m": [c[1] for c in cases],
        "dist_to_industrial_m": [c[2] for c in cases], "osm_industrial_tag": [c[3] for c in cases],
        "landcover_class": [c[4] for c in cases], "recurrence_count": [c[5] for c in cases],
        "dist_to_coal_mine_m": None, "nearest_coal_source": None, "frp": 5.0, "daynight": "N",
    })


def _legacy_apply_rules(frame):
    """apply_rules exactly as it was before offshore existed."""
    result = frame.copy()
    matched = result.apply(_matched_labels, axis=1)
    result["label"] = matched.map(lambda labels: labels[0] if len(labels) == 1 else "unknown")
    result["label_source"] = matched.map(
        lambda labels: "rule" if len(labels) == 1 else ("conflict" if len(labels) > 1 else "unknown")
    )
    return result


def _label_bytes(frame) -> bytes:
    buffer = io.BytesIO()
    frame[["label", "label_source"]].reset_index(drop=True).to_parquet(buffer, index=False)
    return buffer.getvalue()


def test_onshore_label_columns_are_byte_identical_before_and_after():
    onshore = _onshore_rows()
    before = _label_bytes(_legacy_apply_rules(onshore))

    assert _label_bytes(apply_rules(onshore)) == before  # no region column at all

    mixed = pd.concat([onshore.assign(region="onshore"), _offshore_rows([2, 8, 30])], ignore_index=True)
    mixed = mixed.sample(frac=1, random_state=3)  # offshore rows interleaved through the onshore ones
    labeled = apply_rules(mixed)
    after = labeled[labeled["region"] == "onshore"].sort_index()
    assert _label_bytes(after) == before  # same bytes with offshore rows in the frame
    assert set(labeled.loc[labeled["region"] == "offshore", "label"]) == {"unknown", "gas flare"}


def _digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _raw(lat, lon, day, time="0130", sat="N", frp=5.0, daynight="N", region=None):
    row = {"latitude": lat, "longitude": lon, "acq_date": day, "acq_time": time, "satellite": sat,
           "bright_ti4": 330.0, "bright_ti5": 290.0, "frp": frp, "confidence": "n", "daynight": daynight}
    return row | ({"region": region} if region else {})


SITE = (22.3000, 69.8500)
FAR = (22.9000, 70.9000)


def _store(tmp_path, monkeypatch, coast):
    monkeypatch.setattr(offshore_module, "INDIA_BOUNDARY_PATH", coast)
    history = pd.DataFrame([
        _raw(*SITE, f"2026-08-{d:02d}") | {"label": "industrial", "is_anomalous": False} for d in (20, 22, 24, 26, 28, 30)
    ])
    hist_path = tmp_path / "historical.parquet"
    history.to_parquet(hist_path, index=False)
    return dict(
        history_paths=[hist_path], industrial_path=SAMPLE / "industrial_sample.geojson",
        facilities_path=SAMPLE / "facilities_sample.geojson", worldcover=None, repack_map=False,
    )


def test_live_ingest_leaves_onshore_partition_and_alerts_byte_identical(tmp_path, monkeypatch, coast):
    onshore = [_raw(*SITE, "2026-09-01", frp=40.0), _raw(*SITE, "2026-09-02"), _raw(*FAR, "2026-09-02")]
    # 8 offshore nights at one Mumbai High cell: a persistent flare, and one lone detection elsewhere
    sea = [_raw(MUMBAI_HIGH[1], MUMBAI_HIGH[0], f"2026-09-{d:02d}", region="offshore") for d in range(1, 9)]
    sea.append(_raw(19.9, 70.9, "2026-09-03", region="offshore"))

    def run(rows, name):
        (tmp_path / name).mkdir()
        cfg = _store(tmp_path / name, monkeypatch, coast)
        live = tmp_path / name / "live"
        summary, labeled = run_ingest(
            date(2026, 9, 1), date(2026, 9, 8), fetch=lambda s, e, p: pd.DataFrame(rows), live_dir=live,
            alerts_path=live / "alerts_latest.csv", **cfg,
        )
        return summary, labeled, live

    _, _, plain_live = run([{**r, "region": "onshore"} for r in onshore], "plain")
    summary, labeled, mixed_live = run([{**r, "region": "onshore"} for r in onshore] + sea, "mixed")

    onshore_partition = live_partition_path(mixed_live, "2026-09")
    assert _digest(onshore_partition) == _digest(live_partition_path(plain_live, "2026-09"))
    assert _label_bytes(pd.read_parquet(onshore_partition)) == _label_bytes(pd.read_parquet(live_partition_path(plain_live, "2026-09")))
    assert "region" not in labeled.columns and len(labeled) == 3  # the onshore return value is unchanged

    partition = pd.read_parquet(offshore_partition_path(mixed_live, "2026-09"))
    assert list(partition.columns) == OFFSHORE_COLUMNS
    assert len(partition) == 9 and (partition["region"] == "offshore").all()
    by_day = partition.sort_values("acq_date")
    flare = by_day[by_day["latitude"] == MUMBAI_HIGH[1]]
    # causal: active-day count on or before each day, so the label switches on at the 5th day
    assert flare["recurrence_count"].tolist() == [1, 2, 3, 4, 5, 6, 7, 8]
    assert flare["label"].tolist() == ["unknown"] * 4 + ["gas flare"] * 4
    assert set(flare["label_source"]) == {"unknown", OFFSHORE_LABEL_SOURCE}
    assert partition[partition["latitude"] == 19.9]["label"].tolist() == ["unknown"]
    assert set(partition["offshore_zone"]) == {"Mumbai High"} and (partition["dist_offshore_km"] > 10).all()
    assert summary.offshore_new == 9 and summary.offshore_label_mix == {"unknown": 5, "gas flare": 4}

    # alerts: the offshore cell raises its own, and every onshore alert is exactly what it was
    alerts = pd.read_csv(mixed_live / "alerts_history.csv")
    plain_alerts = pd.read_csv(plain_live / "alerts_history.csv")
    at_sea = zone_of(alerts["longitude"], alerts["latitude"]).notna()
    assert at_sea.any()
    pd.testing.assert_frame_equal(alerts[~at_sea].reset_index(drop=True), plain_alerts.reset_index(drop=True), check_dtype=False)


def test_rerunning_a_window_adds_no_offshore_rows(tmp_path, monkeypatch, coast):
    cfg = _store(tmp_path, monkeypatch, coast)
    sea = [_raw(MUMBAI_HIGH[1], MUMBAI_HIGH[0], f"2026-09-{d:02d}", region="offshore") for d in range(1, 4)]
    kwargs = dict(fetch=lambda s, e, p: pd.DataFrame(sea), live_dir=tmp_path / "live", alerts_path=tmp_path / "live" / "a.csv", **cfg)
    run_ingest(date(2026, 9, 1), date(2026, 9, 3), **kwargs)
    path = offshore_partition_path(tmp_path / "live", "2026-09")
    first = _digest(path)
    summary, _ = run_ingest(date(2026, 9, 1), date(2026, 9, 3), **kwargs)
    assert summary.offshore_new == 0 and _digest(path) == first


# --- the offshore build and the landcover lookup over open sea --------------------------------------

def test_bulk_history_counts_active_days_over_the_whole_file(tmp_path, monkeypatch, coast):
    monkeypatch.setattr(offshore_module, "INDIA_BOUNDARY_PATH", coast)
    rows = [_raw(MUMBAI_HIGH[1], MUMBAI_HIGH[0], f"2026-0{m}-15", region="offshore") for m in (1, 2, 3, 4, 5, 6)]
    rows.append(_raw(19.9, 70.9, "2026-03-03", region="offshore"))
    labeled = label_offshore_history(
        add_zone_columns(pd.DataFrame(rows)), industrial_path=SAMPLE / "industrial_sample.geojson",
        facilities_path=SAMPLE / "facilities_sample.geojson", worldcover=None,
    )
    flare = labeled[labeled["latitude"] == MUMBAI_HIGH[1]]
    assert (flare["recurrence_count"] == 6).all() and (flare["label"] == "gas flare").all()
    assert labeled[labeled["latitude"] == 19.9]["label"].tolist() == ["unknown"]
    assert labeled["landcover_class"].isna().all()  # never fabricated over open sea
    assert list(labeled.columns) == OFFSHORE_COLUMNS


def _tile(path, lon, lat, value, size=3000):
    import rasterio
    from rasterio.transform import from_origin

    with rasterio.open(path, "w", driver="GTiff", height=size, width=size, count=1, dtype="uint8",
                       crs="EPSG:4326", transform=from_origin(lon, lat + 3, 3.0 / size, 3.0 / size), nodata=0) as dst:
        dst.write(np.full((size, size), value, dtype="uint8"), 1)


def test_landcover_lookup_does_not_fail_over_open_sea(tmp_path):
    tiles = tmp_path / "tiles"
    tiles.mkdir()
    _tile(tiles / "ESA_WorldCover_10m_2021_v200_N18E069_Map.tif", 69, 18, 0)   # an all-ocean tile: every pixel nodata
    _tile(tiles / "ESA_WorldCover_10m_2021_v200_N18E072_Map.tif", 72, 18, 40)  # land
    index = WorldCoverTileIndex(tiles)
    lon = pd.Series([70.5, 80.0, 73.0, 71.9999], dtype=float)   # ocean tile, no tile at all, land, tile edge over sea
    lat = pd.Series([19.0, 10.0, 19.0, 19.0], dtype=float)
    result = index.majority_class_in_buffer_batch(lon, lat)
    assert result.isna().tolist() == [True, True, False, True]
    assert result.iloc[2] == 40
    assert len(index.majority_class_in_buffer_batch(pd.Series([], dtype=float), pd.Series([], dtype=float))) == 0


# --- the pre-deploy check: nothing outside India except the offshore zones ------------------------------

def test_preflight_recount_allows_only_zone_detections_outside_india(tmp_path, coast):
    import json

    from training.site_preflight import OFFSHORE_STATE, source_counts

    def rows(points, region=None):
        frame = pd.DataFrame({
            "latitude": [p[1] for p in points], "longitude": [p[0] for p in points], "label": "unknown",
            "acq_date": "2026-09-01", "acq_time": "0130", "is_anomalous": False,
        })
        return frame.assign(region=region) if region else frame

    onshore = tmp_path / "onshore.parquet"
    rows([(73.0, 19.4)]).to_parquet(onshore, index=False)
    good = tmp_path / "offshore_good.parquet"
    rows([MUMBAI_HIGH, KG_BASIN], "offshore").to_parquet(good, index=False)
    bad = tmp_path / "offshore_bad.parquet"  # tagged offshore but in no zone (69E), and one inside a state (73E)
    rows([(69.0, 19.4), (73.0, 19.4)], "offshore").to_parquet(bad, index=False)
    stray = tmp_path / "stray.parquet"  # untagged, outside India: not allowed
    rows([MUMBAI_HIGH]).to_parquet(stray, index=False)

    def counts(*files):
        meta = tmp_path / "meta.json"
        meta.write_text(json.dumps({"sources": [{"path": str(f), "rows": 1} for f in files]}))
        # the onshore file's state assignment reads the synthetic coast, so give its polygon a name
        return source_counts(meta, states_path=coast)

    clean = counts(onshore, good)
    assert clean["offshore_bad"] == 0 and OFFSHORE_STATE not in clean["by_state"]
    assert clean["by_state"] == {"west-coast": 1, "Offshore": 2} and clean["offshore_by_class"] == {"unknown": 2}

    assert counts(onshore, bad)["offshore_bad"] == 2  # outside every zone; inside a state polygon
    assert counts(onshore, stray)["by_state"][OFFSHORE_STATE] == 1  # untagged sea detection: still fails the check
