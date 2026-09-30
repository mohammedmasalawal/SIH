from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path

import numpy as np
import osmium
import osmium.osm.mutable as mutable
import pandas as pd
import pytest
from pyproj import Geod

from backend.config import PLACE_VILLAGE_FALLBACK_KM
from backend.places import COMPASS, Places, compass_index, format_near
from training.extract_places import extract_places, is_latin, parse_population, place_name
from training.place_display import RAW_OSM_TITLE, alert_title, build_alert_places, near_text

GEOD = Geod(ellps="WGS84")
ANGUL = (85.0, 20.85)  # lon, lat


def _places(rows) -> Places:
    return Places.from_frame(pd.DataFrame(
        [{"osm_id": i, "place": p, "name": n, "state": s, "lon": lon, "lat": lat, "population": None}
         for i, (p, n, s, (lon, lat)) in enumerate(rows)]
    ))


def _offset(origin, azimuth, km):
    lon, lat, _ = GEOD.fwd(origin[0], origin[1], azimuth, km * 1000.0)
    return lat, lon


# --- lookup -------------------------------------------------------------------------------------

def test_point_at_a_town_returns_that_town_at_about_zero_km():
    places = _places([("town", "Angul", "Odisha", ANGUL), ("town", "Talcher", "Odisha", (85.2, 21.05))])
    out = places.nearest([ANGUL[1]], [ANGUL[0]])
    assert out["name"].iat[0] == "Angul" and out["distance_km"].iat[0] < 0.01
    assert out["text"].iat[0] == "Near Angul, Odisha (0 km)"


def test_ten_km_due_east_reads_10_km_E():
    places = _places([("town", "Angul", "Odisha", ANGUL)])
    lat, lon = _offset(ANGUL, 90, 10)
    out = places.nearest([lat], [lon])
    assert out["text"].iat[0] == "Near Angul, Odisha (10 km E)"
    assert out["distance_km"].iat[0] == pytest.approx(10, rel=0.02)  # EPSG:7755 distances differ from geodesic by up to ~2%


@pytest.mark.parametrize("azimuth, expected", list(zip(range(0, 360, 45), COMPASS)))
def test_all_eight_compass_directions(azimuth, expected):
    places = _places([("town", "Angul", "Odisha", ANGUL)])
    lat, lon = _offset(ANGUL, azimuth, 10)
    out = places.nearest([lat], [lon])
    assert COMPASS[out["compass"].iat[0]] == expected
    assert out["text"].iat[0] == f"Near Angul, Odisha (10 km {expected})"


def test_compass_sectors_are_45_degrees_centred_on_each_point():
    assert compass_index([0, 22.4, 22.6, 67.4, 67.6, 337.4, 337.6, 359.9]).tolist() == [0, 0, 1, 1, 2, 7, 0, 0]


def test_city_and_town_are_pooled_and_the_nearest_wins():
    places = _places([("city", "Big City", "Odisha", _swap(_offset(ANGUL, 0, 12))), ("town", "Angul", "Odisha", ANGUL)])
    lat, lon = _offset(ANGUL, 0, 3)
    assert places.nearest([lat], [lon])["name"].iat[0] == "Angul"  # 3 km from the town, 9 from the city
    lat, lon = _offset(ANGUL, 0, 9)
    assert places.nearest([lat], [lon])["name"].iat[0] == "Big City"  # 3 km from the city, 9 from the town


def _swap(latlon):
    return (latlon[1], latlon[0])


def test_villages_are_used_only_when_no_town_is_within_15_km():
    village = _swap(_offset(ANGUL, 90, 2))
    places = _places([("town", "Angul", "Odisha", ANGUL), ("village", "Tiny", "Odisha", village)])
    # 1 km from the village, 3 km from the town: the town wins, the nearer village does not
    lat, lon = _offset(ANGUL, 90, 3)
    near = places.nearest([lat], [lon])
    assert near["name"].iat[0] == "Angul" and near["level"].iat[0] == "town"
    # 30 km from the town, 28 km from the village: no town within 15 km, so the village
    lat, lon = _offset(ANGUL, 90, 30)
    far = places.nearest([lat], [lon])
    assert far["name"].iat[0] == "Tiny" and far["level"].iat[0] == "village"
    assert PLACE_VILLAGE_FALLBACK_KM == 15.0  # fixed in advance, not tuned
    # the boundary: a town at 14 km still beats a village at 1 km; at 16 km it does not
    v_here = _swap(_offset(ANGUL, 90, 15))
    places = _places([("town", "Angul", "Odisha", ANGUL), ("village", "Near-village", "Odisha", v_here)])
    assert places.nearest(*_lonlat(_offset(ANGUL, 90, 14)))["name"].iat[0] == "Angul"
    assert places.nearest(*_lonlat(_offset(ANGUL, 90, 16)))["name"].iat[0] == "Near-village"


def _lonlat(latlon):
    return [latlon[0]], [latlon[1]]


def test_with_no_villages_the_nearest_town_is_used_at_any_distance():
    places = _places([("town", "Angul", "Odisha", ANGUL)])
    out = places.nearest(*_lonlat(_offset(ANGUL, 180, 80)))
    assert out["distance_km"].iat[0] == pytest.approx(80, rel=0.02) and out["level"].iat[0] == "town"
    assert out["text"].iat[0].endswith("km S)")  # the point is south of the town: the bearing is place -> point


def test_state_is_left_out_when_the_place_has_none():
    assert format_near("Kathmandu", None, 12.4, 2) == "Near Kathmandu (12 km E)"
    assert format_near("Angul", "Odisha", 0.3, 4) == "Near Angul, Odisha (0 km)"
    assert format_near("Angul", "Odisha", 0.5, 4) == "Near Angul, Odisha (1 km S)"


def test_many_points_at_once_match_one_by_one():
    rng = np.random.default_rng(1)
    rows = [("town" if i % 3 else "village", f"P{i}", "S", (75 + rng.random() * 10, 15 + rng.random() * 10)) for i in range(200)]
    places = _places(rows)
    lats, lons = 15 + rng.random(50) * 10, 75 + rng.random(50) * 10
    batch = places.nearest(lats, lons)
    for i in (0, 7, 33, 49):
        single = places.nearest(lats[i:i + 1], lons[i:i + 1])
        assert single["text"].iat[0] == batch["text"].iat[i]


# --- extraction ----------------------------------------------------------------------------------

def test_name_rules():
    assert is_latin("Angul") and is_latin("Nuapada-2") and not is_latin("अंगुल") and not is_latin("Angul अंगुल") and not is_latin("123")
    assert place_name({"name:en": "Angul", "name": "अंगुल"}) == ("Angul", "name:en")
    assert place_name({"name": "Bhubaneswar"}) == ("Bhubaneswar", "name")
    assert place_name({"name": "अंगुल"}) is None and place_name({}) is None
    assert parse_population("12,345") == 12345 and parse_population("about 5") is None and parse_population("1000;2000") == 1000


def test_extract_places_keeps_only_named_city_town_village_nodes(tmp_path):
    pbf = tmp_path / "x.osm.pbf"
    writer = osmium.SimpleWriter(str(pbf))
    nodes = [
        (1, {"place": "city", "name": "Bhubaneswar", "population": "837737"}),
        (2, {"place": "town", "name": "अंगुल", "name:en": "Angul"}),
        (3, {"place": "village", "name": "Kaniha"}),
        (4, {"place": "village", "name": "गाँव"}),                       # non-Latin, no name:en -> skipped
        (5, {"place": "hamlet", "name": "Tiny"}),                         # not a wanted type
        (6, {"place": "town"}),                                           # no name at all
        (7, {"shop": "bakery", "name": "Bread"}),
    ]
    for i, (osm_id, tags) in enumerate(nodes):
        writer.add_node(mutable.Node(id=osm_id, location=(85.0 + i * 0.1, 20.0 + i * 0.1), tags=tags, version=1))
    writer.close()
    out = tmp_path / "places_india.parquet"
    report = extract_places(pbf, out, boundary_path=tmp_path / "no-states.parquet")
    frame = pd.read_parquet(out)
    assert sorted(frame["name"]) == ["Angul", "Bhubaneswar", "Kaniha"]
    assert report["kept"] == {"city": 1, "town": 1, "village": 1}
    assert report["tagged"] == {"city": 1, "town": 2, "village": 2} and report["skipped_no_latin_name"] == {"city": 0, "town": 1, "village": 1}
    by_name = frame.set_index("name")
    assert by_name.loc["Angul", "name_source"] == "name:en" and by_name.loc["Kaniha", "name_source"] == "name"
    assert by_name.loc["Bhubaneswar", "population"] == 837737 and pd.isna(by_name.loc["Kaniha", "population"])
    assert list(frame.columns) == ["osm_id", "place", "name", "name_source", "population", "lon", "lat", "state"]


def test_extract_places_will_not_overwrite_the_industrial_context_cache(tmp_path):
    cache = tmp_path / "india_industrial_context.parquet"
    cache.write_bytes(b"precious")
    with pytest.raises(FileExistsError):
        extract_places(tmp_path / "missing.osm.pbf", cache)
    assert cache.read_bytes() == b"precious"


# --- titles ---------------------------------------------------------------------------------------

def test_generic_industrial_titles_become_industrial_site_near_place():
    assert alert_title("industrial_anomaly", "Industrial site · Odisha", None, "Angul") == "Industrial site near Angul"
    assert alert_title("industrial_anomaly", "OSM industrial=yes", None, "Angul") == "Industrial site near Angul"
    assert alert_title("industrial_anomaly", "cement plant", None, "Angul") is None  # a real facility keeps its title
    assert alert_title("new_activity_at_critical_site", "power plant (coal) · OSM", "unnamed", "Kudgi") == "Power plant (coal) near Kudgi"
    assert alert_title("new_activity_at_critical_site", "power plant (coal) · OSM", "Kudgi Super Thermal", "Kudgi") is None
    assert alert_title("large_fire_event", "", None, "Angul") is None
    assert RAW_OSM_TITLE.search("OSM industrial=yes") and RAW_OSM_TITLE.search("OSM industrial area") and not RAW_OSM_TITLE.search("Industrial site near Angul")


def test_build_alert_places_encodes_near_text_and_titles():
    places = _places([("town", "Angul", "Odisha", ANGUL)])
    lat, lon = _offset(ANGUL, 45, 12)
    alerts = pd.DataFrame([{"alert_id": "a1", "alert_type": "industrial_anomaly", "latitude": lat, "longitude": lon,
                            "facility_type": "Industrial site · Odisha", "facility_name": None}])
    encoded = build_alert_places(alerts, places, {"a1": "OSM industrial=yes"})
    place, km, direction, level, title, raw = encoded["alerts"]["a1"]
    assert near_text(encoded["places"], place, km, direction) == "Near Angul, Odisha (12 km NE)"
    assert (level, title, raw) == (0, "Industrial site near Angul", "OSM industrial=yes")


# --- display-only: nothing but the place fields may change -----------------------------------------

def _sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def test_export_with_place_names_changes_nothing_but_the_place_fields(tmp_path):
    from training.export_site import export
    from training.pack_map_points import pack

    rows = pd.DataFrame([
        {"latitude": 22.10 + i * 0.05, "longitude": 69.10, "acq_date": f"2026-01-0{1 + i % 3}", "acq_time": "0130", "satellite": "N",
         "frp": 3.0 + i, "daynight": "N", "recurrence_count": 2 + i, "landcover_class": 40, "label_source": "rule",
         "dist_to_industrial_m": 5000.0, "label": label, "is_anomalous": bool(i % 2)}
        for i, label in enumerate(["industrial", "gas flare", "agricultural burning", "wildfire", "unknown", "industrial"])
    ])
    parquet = tmp_path / "a.parquet"
    rows.to_parquet(parquet, index=False)
    pack([parquet], tmp_path / "p.bin", tmp_path / "p.json")
    source_before = _sha(parquet)

    alerts = tmp_path / "alerts.csv"
    alerts.write_text(
        "alert_id,alert_type,latitude,longitude,acq_date,acq_time,satellite,daynight,label,frp,facility_type,facility_name\n"
        "industrial_anomaly|1,industrial_anomaly,22.1,69.1,2026-01-01,0130,N,N,industrial,9.0,Industrial site · Gujarat,\n"
        "new_activity_at_critical_site|2,new_activity_at_critical_site,22.15,69.1,2026-01-02,0130,N,N,unknown,2.0,power plant (coal) · OSM,unnamed\n"
        "large_fire_event|3,large_fire_event,22.2,69.1,2026-01-03,,,,wildfire,20.0,,\n", encoding="utf-8")
    facilities = tmp_path / "gem.csv"
    facilities.write_text("facility_name,facility_type,latitude,longitude\nPlant A,cement_plant,22.12,69.11\nPlant B,steel_plant,22.3,69.2\n")
    places_path = tmp_path / "places_india.parquet"
    pd.DataFrame([
        {"osm_id": 1, "place": "town", "name": "Jamnagar", "name_source": "name", "population": 600000, "lon": 70.07, "lat": 22.47, "state": "Gujarat"},
        {"osm_id": 2, "place": "village", "name": "Sikka", "name_source": "name", "population": None, "lon": 69.83, "lat": 22.43, "state": "Gujarat"},
    ]).to_parquet(places_path, index=False)

    def run(name, places):
        dist = tmp_path / name
        export(dist, meta_path=tmp_path / "p.json", points_path=tmp_path / "p.bin", alerts_history=alerts,
               facilities_path=facilities, industrial_context_path=tmp_path / "none.parquet",
               subtype_gem_path=tmp_path / "none.csv", places_path=places)
        return dist / "data"

    without, with_names = run("plain", None), run("named", places_path)

    # everything that carries labels, counts, alerts or points is byte-identical
    for name in ("points.bin.gz", "stats.json.gz", "point_state.bin.gz", "point_subtype.bin.gz", "classes.json",
                 "alerts_history.csv", "states.json.gz", "details/0000.json.gz", "details/index.json"):
        assert (with_names / name).read_bytes() == (without / name).read_bytes(), name
    assert (with_names / "alerts_history.csv").read_bytes() == alerts.read_bytes()  # schema and rows untouched
    assert _sha(parquet) == source_before  # the labelled source file was never touched
    stats = json.loads(gzip.decompress((with_names / "stats.json.gz").read_bytes()))
    assert sum(stats["n"]) == 6

    # the only additions: alert_places.json.gz, and near_* keys inside facilities.json.gz
    assert not (without / "alert_places.json.gz").exists() and (with_names / "alert_places.json.gz").exists()
    plain_fac = json.loads(gzip.decompress((without / "facilities.json.gz").read_bytes()))
    named_fac = json.loads(gzip.decompress((with_names / "facilities.json.gz").read_bytes()))
    extra = set(named_fac) - set(plain_fac)
    assert extra == {"near_places", "near_place", "near_km", "near_dir", "near_level"}
    assert {k: v for k, v in named_fac.items() if k not in extra} == plain_fac

    # and the names themselves
    encoded = json.loads(gzip.decompress((with_names / "alert_places.json.gz").read_bytes()))
    assert set(encoded["alerts"]) == {"industrial_anomaly|1", "new_activity_at_critical_site|2", "large_fire_event|3"}
    assert encoded["alerts"]["industrial_anomaly|1"][4].startswith("Industrial site near ")
    assert encoded["alerts"]["new_activity_at_critical_site|2"][4].startswith("Power plant (coal) near ")
    assert encoded["alerts"]["large_fire_event|3"][4] is None
    assert len(named_fac["near_place"]) == len(named_fac["lat"]) == 2
