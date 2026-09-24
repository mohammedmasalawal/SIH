from __future__ import annotations

import gzip
import json

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from backend import map_store as map_store_module
from backend.api import routes
from backend.classification import classifier
from backend.classification.model_classifier import ModelClassifier
from backend.classification.rules import CLASSES
from backend.config import CLASS_COLORS, CLASS_COLORS_DARK
from backend.map_store import MapStore
from main import app
from training.pack_map_points import pack


def _labeled(rows: list[dict]) -> pd.DataFrame:
    base = {"acq_time": "0130", "satellite": "N", "frp": 3.0, "daynight": "N", "recurrence_count": 2,
            "landcover_class": 40, "label_source": "rule", "dist_to_industrial_m": 5000.0}
    return pd.DataFrame([base | row for row in rows])


@pytest.fixture
def packed(tmp_path):
    first = _labeled([
        {"latitude": 22.1, "longitude": 69.1, "acq_date": "2026-01-01", "label": "industrial", "is_anomalous": True},
        {"latitude": 22.2, "longitude": 69.2, "acq_date": "2026-01-03", "label": "wildfire", "is_anomalous": False},
    ])
    second = _labeled([
        {"latitude": 30.5, "longitude": 75.5, "acq_date": "2026-02-10", "label": "agricultural burning",
         "is_anomalous": None, "acq_time": 626},  # int acq_time, as in some real period files
    ])
    paths = [tmp_path / "a.parquet", tmp_path / "b.parquet"]
    first.to_parquet(paths[0], index=False)
    second.to_parquet(paths[1], index=False)
    out_bin, out_meta = tmp_path / "points.bin", tmp_path / "points.json"
    meta = pack(paths, out_bin, out_meta)
    return meta, out_bin, out_meta


def test_pack_layout_round_trips(packed):
    meta, out_bin, _ = packed
    raw = out_bin.read_bytes()
    assert len(raw) == meta["bytes"] == 16 * 3
    assert gzip.decompress(out_bin.with_name(out_bin.name + ".gz").read_bytes()) == raw

    def view(name):
        spec = meta["layout"][name]
        return np.frombuffer(raw, dtype=np.dtype(spec["dtype"]).newbyteorder("<"), count=spec["length"], offset=spec["offset"])

    # stored in draw order (agri, wildfire, then the rarer industrial on top); row_id
    # still points at each detection's original row
    assert [CLASSES[i] for i in view("class_id")] == ["agricultural burning", "wildfire", "industrial"]
    assert view("row_id").tolist() == [2, 1, 0]
    np.testing.assert_allclose(view("positions"), [75.5, 30.5, 69.2, 22.2, 69.1, 22.1], rtol=1e-6)
    assert meta["base_date"] == "2026-01-01"
    assert view("day").tolist() == [40, 2, 0]
    assert view("is_anomalous").tolist() == [0, 0, 1]
    assert [s["rows"] for s in meta["sources"]] == [2, 1]
    assert meta["class_counts"]["industrial"] == 1


def test_pack_rejects_unknown_label(tmp_path):
    path = tmp_path / "bad.parquet"
    _labeled([{"latitude": 1.0, "longitude": 1.0, "acq_date": "2026-01-01", "label": "industrial fire",
               "is_anomalous": False}]).to_parquet(path, index=False)
    with pytest.raises(ValueError, match="industrial fire"):
        pack([path], tmp_path / "p.bin", tmp_path / "p.json")


@pytest.fixture
def client(monkeypatch, tmp_path, packed):
    _, out_bin, out_meta = packed
    monkeypatch.setattr(routes, "map_store", MapStore(out_bin, out_meta))
    monkeypatch.setattr(classifier, "_model_classifier", ModelClassifier(tmp_path / "no_model.pkl"))
    return TestClient(app)


def test_detection_resolves_row_ids_across_files(client):
    first = client.get("/api/detection/0").json()
    assert first["id"] == 0 and first["label"] == "industrial" and first["is_anomalous"] is True
    third = client.get("/api/detection/2").json()
    assert third["label"] == "agricultural burning"
    assert third["latitude"] == pytest.approx(30.5)
    assert third["acq_time"] == "0626"


def test_detection_out_of_range_is_404(client):
    assert client.get("/api/detection/3").status_code == 404


def test_map_meta_hides_server_paths(client):
    meta = client.get("/api/map/meta").json()
    assert meta["count"] == 3 and "sources" not in meta


def test_points_served_gzipped_when_accepted(client, packed):
    _, out_bin, _ = packed
    response = client.get("/api/map/points.bin", headers={"Accept-Encoding": "gzip"})
    assert response.status_code == 200
    assert response.headers["content-encoding"] == "gzip"
    assert response.content == out_bin.read_bytes()  # the client transparently decompresses


def test_map_endpoints_503_when_not_built(monkeypatch, tmp_path):
    monkeypatch.setattr(routes, "map_store", MapStore(tmp_path / "missing.bin", tmp_path / "missing.json"))
    client = TestClient(app)
    assert client.get("/api/map/meta").status_code == 503
    assert client.get("/api/detection/0").status_code == 503


def test_classify_passes_coal_fields_to_rules(client):
    payload = {"latitude": 23.7, "longitude": 86.4, "acq_date": "2026-01-10", "acq_time": "0130", "satellite": "N",
               "dist_to_industrial_m": 5000, "dist_to_heat_industry_m": 40000, "dist_to_flare_capable_m": 40000,
               "dist_to_coal_mine_m": 400, "nearest_coal_source": "osm", "recurrence_count": 5, "landcover_class": "60"}
    assert client.post("/api/classify", json=payload).json()["label"] == "industrial"


def test_classes_expose_both_color_modes(client):
    body = client.get("/api/classes").json()
    assert body["colors"] == CLASS_COLORS and body["colors_dark"] == CLASS_COLORS_DARK
    assert set(body["colors_dark"]) == set(CLASSES)


def test_hotspots_source_names_the_file(client):
    assert client.get("/api/hotspots").json()["source"] == "demo_hotspots.csv"


def test_module_level_store_points_at_config_paths():
    assert map_store_module.map_store.points_path.name == "national_points.bin"
