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


# ---- colours for "Colour by: Industrial type" ---------------------------------------------------------

# Pinned: these are the class colours users see today. "Colour by: Class" must keep using exactly these.
PINNED_CLASS_COLORS = {"industrial": "#2a78d6", "gas flare": "#e87ba4", "agricultural burning": "#eda100",
                       "wildfire": "#008300", "unknown": "#8f8e89"}
PINNED_CLASS_COLORS_DARK = {"industrial": "#3987e5", "gas flare": "#d55181", "agricultural burning": "#c98500",
                            "wildfire": "#008300", "unknown": "#6b6a65"}


def test_class_colours_are_unchanged():
    from backend.config import CLASS_COLORS, CLASS_COLORS_DARK

    assert CLASS_COLORS == PINNED_CLASS_COLORS and CLASS_COLORS_DARK == PINNED_CLASS_COLORS_DARK


def test_export_writes_class_colours_unchanged_and_the_subtype_palette(tmp_path):
    import gzip
    import json

    from training.export_site import export
    from training.industrial_subtype import DIM_ALPHA, DIM_NEUTRAL, SUBTYPE_COLORS

    meta_dir = tmp_path / "m"
    meta_dir.mkdir()
    rows = pd.DataFrame([{"latitude": 22.1, "longitude": 69.1, "acq_date": "2026-01-01", "acq_time": "0130", "satellite": "N",
                          "frp": 3.0, "daynight": "N", "recurrence_count": 2, "landcover_class": 40, "label_source": "rule",
                          "dist_to_industrial_m": 5000.0, "label": "industrial", "is_anomalous": False}])
    parquet = meta_dir / "a.parquet"
    rows.to_parquet(parquet, index=False)
    from training.pack_map_points import pack

    pack([parquet], meta_dir / "p.bin", meta_dir / "p.json")
    alerts = tmp_path / "alerts.csv"
    alerts.write_text("alert_id,alert_type,latitude,longitude,acq_date\n")
    dist = tmp_path / "dist"
    export(dist, meta_path=meta_dir / "p.json", points_path=meta_dir / "p.bin", alerts_history=alerts,
           facilities_path=tmp_path / "n.csv", industrial_context_path=tmp_path / "n.parquet", subtype_gem_path=tmp_path / "n.csv")
    classes = json.loads((dist / "data" / "classes.json").read_text(encoding="utf-8"))
    assert classes["colors"] == PINNED_CLASS_COLORS and classes["colors_dark"] == PINNED_CLASS_COLORS_DARK
    assert {t["name"]: t["color"] for t in classes["subtypes"]} == SUBTYPE_COLORS
    assert classes["subtype_dim"] == {"color": DIM_NEUTRAL, "alpha": DIM_ALPHA}


def test_subtype_palette_stays_out_of_the_class_palette():
    from training.colour_check import delta_e, oklch
    from training.industrial_subtype import SUBTYPE_COLORS

    chromatic = [c for name, c in SUBTYPE_COLORS.items() if name != NOT_IDENTIFIED]
    assert len(set(SUBTYPE_COLORS.values())) == 7 and len(chromatic) == 6
    for colour in chromatic:  # blue / cyan / violet only: OKLCH hue 185-315 (yellow ~90, green ~145, magenta/red ~330-30)
        assert 185 <= oklch(colour)[2] <= 315, colour
    grey_l, grey_c, grey_h = oklch(SUBTYPE_COLORS[NOT_IDENTIFIED])
    assert 0.02 < grey_c < 0.08 and 215 <= grey_h <= 280  # muted, and bluer than the grey "unclassified" class colour
    for colour in SUBTYPE_COLORS.values():
        for klass in PINNED_CLASS_COLORS_DARK.values():
            assert delta_e(colour, klass) >= 8, (colour, klass)  # no sub-type reads as a class colour
    assert delta_e(SUBTYPE_COLORS[NOT_IDENTIFIED], PINNED_CLASS_COLORS_DARK["unknown"]) >= 12


def test_subtype_palette_passes_the_validator_checks_on_adjacent_pairs_and_the_weak_pairs_are_the_documented_ones():
    from training.colour_check import check, subtype_palette, weak_pairs

    palette = subtype_palette()
    report = check(palette, chromatic=6)
    assert report["off_band"] == [] and report["low_chroma"] == [] and report["low_contrast"] == []
    assert report["adjacent_worst_cvd"] >= 8 and report["adjacent_worst_normal"] >= 15  # legend / panel order
    # Six blue/cyan/violet hues + a blue-grey can't all be pairwise separated in the dark lightness band. These
    # are the pairs that can't be; if the palette changes, this list must be reviewed (README documents it).
    weak = {(ALL_SUBTYPES[p["i"]], ALL_SUBTYPES[p["j"]]) for p in weak_pairs(palette)}
    assert weak == {
        ("Refinery / oil & gas", "Mine / coal-seam fire"), ("Refinery / oil & gas", NOT_IDENTIFIED),
        ("Thermal power plant", "Chemical / petrochemical"), ("Mine / coal-seam fire", NOT_IDENTIFIED),
        ("Steel", "Chemical / petrochemical"), ("Cement", NOT_IDENTIFIED),
    }
