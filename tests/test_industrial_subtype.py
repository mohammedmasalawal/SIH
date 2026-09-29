from __future__ import annotations

import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import Point, box

from backend.classification import rules
from backend.config import CONTRACT_COLUMNS
from training.industrial_subtype import (
    ALL_SUBTYPES,
    NOT_IDENTIFIED,
    OUTPUT_COLUMNS,
    SUBTYPE_CODE,
    compute_industrial_subtypes,
    label_fingerprint,
    load_typed_facilities,
    split_table,
)

KM_LAT = 1 / 111.32
LAT, LON = 23.7, 86.4  # Jharkhand coal belt


@pytest.fixture
def gem_csv(tmp_path):
    path = tmp_path / "gem.csv"
    pd.DataFrame([
        {"facility_name": "Test Steel Works", "facility_type": "steel_plant", "latitude": LAT, "longitude": LON, "status": "operating"},
        {"facility_name": "Test Cement", "facility_type": "cement_plant", "latitude": LAT + 0.5, "longitude": LON, "status": "operating"},
        {"facility_name": "Old Mine", "facility_type": "coal_mine", "latitude": LAT + 1.0, "longitude": LON, "status": "closed"},
        {"facility_name": "Never Built", "facility_type": "coal_mine", "latitude": LAT + 1.5, "longitude": LON, "status": "cancelled"},
        {"facility_name": "Gas Plant", "facility_type": "oil_gas_power_plant", "latitude": LAT + 2.0, "longitude": LON, "status": "operating"},
    ]).to_csv(path, index=False)
    return path


@pytest.fixture
def osm_parquet(tmp_path):
    path = tmp_path / "osm.parquet"
    gpd.GeoDataFrame(
        {
            "industrial": ["refinery", "yes", None, None, "steel"],
            "power": [None, None, "plant", "plant", None],
            "plant:source": [None, None, "coal", "solar", None],
            "name": ["OSM Refinery", None, "OSM Coal Plant", "OSM Solar Farm", None],
        },
        geometry=[box(LON + 1.0, LAT + 3.0, LON + 1.01, LAT + 3.01),   # refinery polygon
                  Point(LON + 1.0, LAT + 4.0),                          # generic industrial=yes: no type
                  Point(LON + 1.0, LAT + 5.0),                          # thermal plant
                  Point(LON + 1.0, LAT + 6.0),                          # solar farm: excluded upstream
                  Point(LON + 1.0, LAT + 7.0)],                         # unnamed OSM steel
        crs="EPSG:4326",
    ).to_parquet(path)
    return path


def _detections(*offsets_km, label="industrial", base=(LAT, LON)):
    return pd.DataFrame({
        "latitude": [base[0] + km * KM_LAT for km in offsets_km],
        "longitude": [base[1]] * len(offsets_km),
        "label": [label] * len(offsets_km),
    })


def test_label_column_is_byte_identical_and_never_returned(gem_csv, osm_parquet):
    frame = _detections(0.3, 5.0, 0.2, label="industrial")
    frame.loc[2, "label"] = "wildfire"
    before, snapshot = label_fingerprint(frame["label"]), frame.copy(deep=True)
    typed = load_typed_facilities(gem_csv, osm_parquet)
    result = compute_industrial_subtypes(frame, typed)
    assert label_fingerprint(frame["label"]) == before
    pd.testing.assert_frame_equal(frame, snapshot)          # the input isn't modified at all
    assert "label" not in result.columns and list(result.columns) == OUTPUT_COLUMNS
    assert frame["label"].astype(str).str.cat(sep="|").encode() == snapshot["label"].astype(str).str.cat(sep="|").encode()


def test_rules_and_contract_never_reference_subtypes():
    import inspect

    source = inspect.getsource(rules)
    assert "industrial_subtype" not in source and "training.industrial_subtype" not in source
    assert "subtype_distance_m" not in source and "GEM_MULTISECTOR" not in source
    assert not any("subtype" in column for column in CONTRACT_COLUMNS)


def test_nearest_typed_facility_within_1km_else_not_identified(gem_csv, osm_parquet):
    frame = _detections(0.3, 0.99, 1.2, 60.0)
    result = compute_industrial_subtypes(frame, load_typed_facilities(gem_csv, osm_parquet))
    assert list(result["industrial_subtype"]) == ["Steel", "Steel", NOT_IDENTIFIED, NOT_IDENTIFIED]
    assert result.loc[0, "subtype_facility"] == "Test Steel Works" and result.loc[0, "subtype_source"] == "GEM"
    assert result.loc[0, "subtype_distance_m"] == pytest.approx(300, rel=0.03)  # EPSG:7755 scale error ~2% here
    assert pd.isna(result.loc[2, "subtype_facility"]) and pd.isna(result.loc[2, "subtype_distance_m"])


def test_only_industrial_rows_get_a_subtype(gem_csv, osm_parquet):
    frame = pd.concat([_detections(0.1, label="industrial"), _detections(0.1, label="wildfire"),
                       _detections(0.1, label="unknown")], ignore_index=True)
    result = compute_industrial_subtypes(frame, load_typed_facilities(gem_csv, osm_parquet))
    assert result.loc[0, "industrial_subtype"] == "Steel"
    assert result.loc[[1, 2], OUTPUT_COLUMNS].isna().all().all()


def test_gem_excluded_statuses_and_closed_mines(gem_csv, osm_parquet):
    typed = load_typed_facilities(gem_csv, osm_parquet)
    assert "Never Built" not in set(typed["name"])            # cancelled: dropped
    frame = _detections(0.2, base=(LAT + 1.0, LON))            # at the closed mine
    assert compute_industrial_subtypes(frame, typed).loc[0, "industrial_subtype"] == "Mine / coal-seam fire"
    at_cancelled = _detections(0.2, base=(LAT + 1.5, LON))
    assert compute_industrial_subtypes(at_cancelled, typed).loc[0, "industrial_subtype"] == NOT_IDENTIFIED


def test_osm_tags_map_to_subtypes_and_generic_industrial_stays_untyped(gem_csv, osm_parquet):
    typed = load_typed_facilities(gem_csv, osm_parquet)
    cases = {3.0: ("Refinery / oil & gas", "OSM Refinery"), 4.0: (NOT_IDENTIFIED, None),
             5.0: ("Thermal power plant", "OSM Coal Plant"), 6.0: (NOT_IDENTIFIED, None), 7.0: ("Steel", None)}
    for offset_deg, (subtype, name) in cases.items():
        frame = pd.DataFrame({"latitude": [LAT + offset_deg + 0.002], "longitude": [LON + 1.0], "label": ["industrial"]})
        result = compute_industrial_subtypes(frame, typed).loc[0]
        assert result["industrial_subtype"] == subtype, offset_deg
        if name:
            assert result["subtype_facility"] == name and result["subtype_source"] == "OSM"
    assert "OSM Solar Farm" not in set(typed["name"].dropna())  # solar plants never count


def test_nearest_wins_and_gem_wins_an_exact_tie(gem_csv, tmp_path):
    osm = tmp_path / "osm2.parquet"
    gpd.GeoDataFrame({"industrial": ["cement"], "name": ["OSM Cement"]},
                     geometry=[Point(LON, LAT)], crs="EPSG:4326").to_parquet(osm)   # same point as the GEM steel plant
    typed = load_typed_facilities(gem_csv, osm)
    at_site = _detections(0.0)
    assert compute_industrial_subtypes(at_site, typed).loc[0, "industrial_subtype"] == "Steel"  # GEM listed first
    nearer_osm = pd.DataFrame({"latitude": [LAT - 0.4 * KM_LAT], "longitude": [LON], "label": ["industrial"]})
    assert compute_industrial_subtypes(nearer_osm, typed).loc[0, "subtype_distance_m"] == pytest.approx(400, rel=0.03)


def test_missing_input_files_leave_everything_not_identified(tmp_path):
    typed = load_typed_facilities(tmp_path / "none.csv", tmp_path / "none.parquet")
    result = compute_industrial_subtypes(_detections(0.1, 0.2), typed)
    assert set(result["industrial_subtype"]) == {NOT_IDENTIFIED}


def test_codes_and_split_table():
    assert SUBTYPE_CODE[NOT_IDENTIFIED] == len(ALL_SUBTYPES) and min(SUBTYPE_CODE.values()) == 1
    table = split_table(pd.DataFrame({"industrial_subtype": ["Steel", "Steel", NOT_IDENTIFIED, None]}))
    assert table.loc["Steel", "detections"] == 2 and table.loc[NOT_IDENTIFIED, "share_pct"] == pytest.approx(33.3, abs=0.1)
    assert list(table.index) == list(ALL_SUBTYPES)
