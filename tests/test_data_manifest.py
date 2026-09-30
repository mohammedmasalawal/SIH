from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

import training.data_manifest as dm


@pytest.fixture
def data_dir(tmp_path, monkeypatch):
    root = tmp_path / "data"
    (root / "osm").mkdir(parents=True)
    (root / "raw" / "worldcover_india").mkdir(parents=True)
    (root / "osm" / "india-latest.osm.pbf").write_bytes(b"p" * 100)
    (root / "osm" / "india_industrial_context.parquet").write_bytes(b"c" * 10)
    (root / "raw" / "worldcover_india" / "T1.tif").write_bytes(b"t" * 7)
    monkeypatch.setattr(dm, "DATA_DIR", root)
    monkeypatch.setattr(dm, "OSM_PBF", root / "osm" / "india-latest.osm.pbf")
    monkeypatch.setattr(dm, "NATIONAL_INDUSTRIAL_CONTEXT_PATH", root / "osm" / "india_industrial_context.parquet")
    monkeypatch.setattr(dm, "NATIONAL_WORLDCOVER_DIR", root / "raw" / "worldcover_india")
    return root


def test_check_passes_on_a_matching_folder_and_fails_clearly_otherwise(data_dir, tmp_path):
    manifest = tmp_path / "MANIFEST.json"
    dm.build(manifest_path=manifest)
    assert dm.check(manifest) == []
    assert dm.check(manifest, checksums=True) == []

    (data_dir / "raw" / "worldcover_india" / "T1.tif").write_bytes(b"t" * 8)          # wrong size
    assert dm.check(manifest) == ["size differs: raw/worldcover_india/T1.tif is 8 bytes, manifest says 7"]
    (data_dir / "raw" / "worldcover_india" / "T1.tif").write_bytes(b"x" * 7)          # right size, wrong content
    assert dm.check(manifest) == [] and dm.check(manifest, checksums=True) == ["checksum differs: raw/worldcover_india/T1.tif"]
    (data_dir / "osm" / "india_industrial_context.parquet").unlink()                  # missing
    assert "missing: osm/india_industrial_context.parquet" in dm.check(manifest)
    (data_dir / "raw" / "worldcover_india" / "T2.tif").write_bytes(b"?")              # a tile the manifest doesn't know
    assert "not in the manifest: raw/worldcover_india/T2.tif" in dm.check(manifest)
    assert dm.check(tmp_path / "nope.json") == [f"{tmp_path / 'nope.json'} is missing"]


def test_a_failed_check_pings_only_the_fail_url(data_dir, tmp_path, monkeypatch, capsys):
    requests = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            requests.append(("POST", self.path, self.rfile.read(int(self.headers.get("Content-Length", 0))).decode()))
            self.send_response(200); self.end_headers(); self.wfile.write(b"OK")

        do_GET = do_POST

        def log_message(self, *args):
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        manifest = tmp_path / "MANIFEST.json"
        dm.build(manifest_path=manifest)
        monkeypatch.setattr(dm, "MANIFEST_PATH", manifest)
        monkeypatch.setenv("HEARTBEAT_URL", f"http://127.0.0.1:{server.server_address[1]}/ping/abc")
        monkeypatch.setattr("sys.argv", ["data_manifest", "check"])

        assert dm.main() == 0 and requests == []                      # healthy: no ping of any kind from this check
        (data_dir / "osm" / "india-latest.osm.pbf").write_bytes(b"short")
        assert dm.main() == 2
        err = capsys.readouterr().err
        assert "INGEST NOT STARTED" in err and "size differs: osm/india-latest.osm.pbf" in err and "heartbeat failure ping sent" in err
        assert [(m, p) for m, p, _ in requests] == [("POST", "/ping/abc/fail")]  # the failure URL only, never success
        assert "size differs" in requests[0][2]
    finally:
        server.shutdown()


def test_no_heartbeat_url_means_no_ping_and_a_clear_message(data_dir, tmp_path, monkeypatch, capsys):
    manifest = tmp_path / "MANIFEST.json"
    dm.build(manifest_path=manifest)
    monkeypatch.setattr(dm, "MANIFEST_PATH", manifest)
    monkeypatch.setattr(dm, "_env", lambda name: "")
    monkeypatch.setattr("sys.argv", ["data_manifest", "check"])
    (data_dir / "osm" / "india-latest.osm.pbf").unlink()
    assert dm.main() == 2
    err = capsys.readouterr().err
    assert "missing: osm/india-latest.osm.pbf" in err and "no HEARTBEAT_URL set" in err


def test_manifest_records_sizes_and_checksums(data_dir, tmp_path):
    manifest = json.loads(json.dumps(dm.build(manifest_path=tmp_path / "m.json")))
    assert set(manifest["files"]) == {"osm/india-latest.osm.pbf", "osm/india_industrial_context.parquet", "raw/worldcover_india/T1.tif"}
    tile = manifest["files"]["raw/worldcover_india/T1.tif"]
    assert tile["size"] == 7 and len(tile["sha256"]) == 64 and tile["source"].endswith("/T1.tif")
