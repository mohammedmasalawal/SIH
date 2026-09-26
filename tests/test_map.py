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


def test_export_site_round_trips_details_and_stats(packed, tmp_path):
    from training.export_site import export

    meta, out_bin, out_meta = packed
    alerts = tmp_path / "alerts_history.csv"
    alerts.write_text("alert_id,alert_type,latitude,longitude,acq_date\nx,industrial_anomaly,22.1,69.1,2026-01-01\n")
    dist = tmp_path / "dist"
    gem = tmp_path / "gem.csv"
    gem.write_text("facility_name,facility_type,latitude,longitude\nJamnagar Refinery,refinery,22.35,69.85\n")
    summary = export(dist, meta_path=out_meta, points_path=out_bin, alerts_history=alerts,
                     facilities_path=gem, industrial_context_path=tmp_path / "no_osm.parquet")
    data = dist / "data"
    assert summary["rows"] == 3 and summary["shards"] == 1
    for name in ("index.html", "static/js/dashboard.js", "vercel.json", "data/meta.json", "data/classes.json",
                 "data/points.bin.gz", "data/stats.json.gz", "data/alerts_history.csv", "data/details/index.json"):
        assert (dist / name).exists(), name
    assert gzip.decompress((data / "points.bin.gz").read_bytes()) == out_bin.read_bytes()

    index = json.loads((data / "details" / "index.json").read_text())
    shard = json.loads(gzip.decompress((data / "details" / "0000.json.gz").read_bytes()))
    row = 2  # third source row: the int-acq_time agricultural-burning detection
    assert shard["lat"][row] / 1e5 == pytest.approx(30.5) and shard["acq_time"][row] == 626
    assert index["dicts"]["label"][shard["label"][row]] == "agricultural burning"
    assert shard["day"][row] == 40 and shard["anom"][0] == 1

    stats = json.loads(gzip.decompress((data / "stats.json.gz").read_bytes()))
    assert sum(stats["n"]) == 3 and sum(stats["anom"]) == 1
    # per-row state index lines up with stats.json's state names
    row_states = [stats["states"][i] for i in gzip.decompress((data / "point_state.bin.gz").read_bytes())]
    assert row_states[0] == row_states[1] == "Gujarat" and row_states[2] == "Punjab"
    outlines = json.loads(gzip.decompress((data / "states.json.gz").read_bytes()))
    assert {"Gujarat", "Punjab"} <= set(outlines["bbox"])
    facilities = json.loads(gzip.decompress((data / "facilities.json.gz").read_bytes()))
    assert facilities["name"] == ["Jamnagar Refinery"] and facilities["group"] == [0]
    assert facilities["lat"] == [2235000] and facilities["kinds"][facilities["kind"][0]] == "refinery"
    classes = json.loads((data / "classes.json").read_text(encoding="utf-8"))
    assert classes["display"]["unknown"] == "Unclassified — needs review"
    assert json.loads((data / "meta.json").read_text())["detail_shard_rows"] == index["shard_rows"]


def test_export_site_is_stale_tracks_pack_and_alerts(tmp_path):
    import os

    from training.export_site import is_stale

    meta, alerts, frontend = tmp_path / "points.json", tmp_path / "alerts.csv", tmp_path / "frontend"
    meta.write_text("{}")
    alerts.write_text("alert_id\n")
    frontend.mkdir()
    page = frontend / "index.html"
    page.write_text("<html></html>")
    reference = tmp_path / "dist" / ".deployed"
    assert is_stale(reference, meta, alerts, frontend)  # never deployed
    reference.parent.mkdir()
    reference.write_text("x")
    for path in (meta, alerts, page):
        os.utime(path, (1_000, 1_000))
    assert not is_stale(reference, meta, alerts, frontend)
    os.utime(alerts, None)  # a new alert after the deploy
    assert is_stale(reference, meta, alerts, frontend)
    os.utime(alerts, (1_000, 1_000))
    os.utime(page, None)  # a page change also needs a redeploy
    assert is_stale(reference, meta, alerts, frontend)


def test_site_preflight_data_checks(packed, tmp_path):
    from training.export_site import export
    from training.site_preflight import check_alerts, check_points

    _, out_bin, out_meta = packed
    alerts = tmp_path / "alerts_history.csv"
    alerts.write_text("alert_id,alert_type,latitude,longitude,acq_date\nx,industrial_anomaly,22.1,69.1,2026-01-01\n")
    dist = tmp_path / "dist"
    export(dist, meta_path=out_meta, points_path=out_bin, alerts_history=alerts,
           facilities_path=tmp_path / "none.csv", industrial_context_path=tmp_path / "none.parquet")
    assert check_points(dist) == [] and check_alerts(dist) == []

    (dist / "data" / "points.bin.gz").write_bytes(b"")
    assert check_points(dist) == ["data/points.bin.gz is empty"]
    (dist / "data" / "alerts_history.csv").write_text("alert_id,alert_type,latitude,longitude,acq_date\nx,bogus,abc,69.1,2026-01-01\n")
    problems = check_alerts(dist)
    assert any("numeric coordinate" in p for p in problems) and any("bogus" in p for p in problems)
    (dist / "data" / "alerts_history.csv").unlink()
    assert check_alerts(dist) == ["data/alerts_history.csv is missing"]


class _FakeVercel:
    """Stands in for subprocess.run: fails the first `failures` deploy calls."""

    def __init__(self, failures: int, secret: str = "tok_SECRET_123"):
        self.failures, self.secret, self.calls = failures, secret, []

    def __call__(self, args, env, **kwargs):
        import subprocess

        self.calls.append((args, env))
        if len(self.calls) <= self.failures:
            return subprocess.CompletedProcess(args, 1, stdout=f"uploading... token={self.secret}\n",
                                               stderr=f"Error: network hiccup {len(self.calls)} ({self.secret})\n")
        return subprocess.CompletedProcess(args, 0, stdout='{"url": "https://agninetra-abc.vercel.app"}', stderr="")


@pytest.fixture
def linked_dist(tmp_path, monkeypatch):
    from training import export_site

    monkeypatch.setattr(export_site.shutil, "which", lambda name: "vercel")
    (tmp_path / ".vercel").mkdir()
    (tmp_path / ".vercel" / "project.json").write_text("{}")
    return tmp_path


def test_deploy_retries_then_succeeds_with_token_in_env_only(linked_dist):
    from training.export_site import DEPLOY_MARKER, deploy

    fake, waits = _FakeVercel(failures=2), []
    url = deploy(linked_dist, token=fake.secret, run=fake, sleep=waits.append)
    assert url == "https://agninetra-abc.vercel.app"
    assert waits == [30, 90] and len(fake.calls) == 3
    for args, env in fake.calls:
        assert env["VERCEL_TOKEN"] == fake.secret
        assert not any(fake.secret in a for a in args)  # never on the command line
    assert (linked_dist / DEPLOY_MARKER).exists()


def test_deploy_failure_logs_every_attempt_with_token_redacted(linked_dist):
    from training.export_site import DEPLOY_MARKER, DeployError, deploy

    fake = _FakeVercel(failures=3)
    with pytest.raises(DeployError) as caught:
        deploy(linked_dist, token=fake.secret, run=fake, sleep=lambda s: None)
    log = caught.value.log
    assert "failed 3 times" in str(caught.value)
    for n in (1, 2, 3):
        assert f"=== attempt {n}: exit status 1 ===" in log and f"network hiccup {n}" in log
    assert fake.secret not in log and "token=***" in log
    assert not (linked_dist / DEPLOY_MARKER).exists()


def test_deploy_require_token_refuses_interactive_login(linked_dist, monkeypatch):
    from training import export_site

    monkeypatch.setattr(export_site, "_env", lambda name: "")
    fake = _FakeVercel(failures=0)
    with pytest.raises(export_site.DeployError, match="VERCEL_TOKEN is not set"):
        export_site.deploy(linked_dist, require_token=True, run=fake, sleep=lambda s: None)
    assert fake.calls == []
