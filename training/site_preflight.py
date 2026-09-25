"""Pre-deploy checks for the exported dashboard in dist/ (training/export_site.py).

    python -m training.site_preflight        # check dist/ without deploying

preflight() returns a list of failure reasons (empty = safe to deploy):

1. data/points.bin.gz exists, is non-empty, and decompresses to the byte count and
   point count data/meta.json promises;
2. data/alerts_history.csv parses, has the columns the alerts panel reads, and every
   row has a coordinate, a date and a known alert type;
3. the page loads in headless Chrome, served from dist/ over local HTTP exactly as
   a static host would, and renders the points, the alerts list and one click-detail
   record with no JavaScript exception or console error.
"""

from __future__ import annotations

import functools
import gzip
import json
import os
import shutil
import socket
import subprocess
import tempfile
import threading
import time
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pandas as pd

from backend.config import SITE_DIST_DIR

ALERT_COLUMNS = ("alert_id", "alert_type", "latitude", "longitude", "acq_date")
ALERT_TYPES = ("industrial_anomaly", "new_activity_at_critical_site", "new_unmapped_source", "large_fire_event")
PAGE_TIMEOUT_S = 90
_CHROME_CANDIDATES = (
    os.environ.get("AGNINETRA_CHROME", ""),
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    "google-chrome", "chromium", "chrome",
)


def check_points(dist: Path) -> list[str]:
    data = Path(dist) / "data"
    points, meta_path = data / "points.bin.gz", data / "meta.json"
    if not meta_path.exists():
        return ["data/meta.json is missing"]
    if not points.exists():
        return ["data/points.bin.gz is missing"]
    if points.stat().st_size == 0:
        return ["data/points.bin.gz is empty"]
    meta = json.loads(meta_path.read_text())
    try:
        raw = gzip.decompress(points.read_bytes())
    except (OSError, EOFError) as error:
        return [f"data/points.bin.gz is not valid gzip: {error}"]
    failures = []
    if len(raw) != meta.get("bytes"):
        failures.append(f"data/points.bin.gz holds {len(raw):,} bytes, meta.json says {meta.get('bytes')}")
    if not meta.get("count"):
        failures.append("meta.json reports 0 points")
    return failures


def check_alerts(dist: Path) -> list[str]:
    path = Path(dist) / "data" / "alerts_history.csv"
    if not path.exists():
        return ["data/alerts_history.csv is missing"]
    try:
        alerts = pd.read_csv(path, dtype=str, keep_default_na=False)
    except Exception as error:  # any parse failure blocks the deploy
        return [f"data/alerts_history.csv does not parse: {error}"]
    missing = [c for c in ALERT_COLUMNS if c not in alerts.columns]
    if missing:
        return [f"data/alerts_history.csv lacks columns {missing}"]
    failures = []
    coords = alerts[["latitude", "longitude"]].apply(pd.to_numeric, errors="coerce")
    if coords.isna().any(axis=None):
        failures.append(f"data/alerts_history.csv: {int(coords.isna().any(axis=1).sum())} rows without a numeric coordinate")
    if pd.to_datetime(alerts["acq_date"], errors="coerce", format="%Y-%m-%d").isna().any():
        failures.append("data/alerts_history.csv: rows with an unparseable acq_date")
    unknown = sorted(set(alerts["alert_type"]) - set(ALERT_TYPES))
    if unknown:
        failures.append(f"data/alerts_history.csv: unknown alert_type {unknown}")
    if alerts["alert_id"].duplicated().any():
        failures.append("data/alerts_history.csv: duplicate alert_id")
    return failures


def _chrome() -> str | None:
    for candidate in _CHROME_CANDIDATES:
        if candidate and (Path(candidate).exists() or shutil.which(candidate)):
            return candidate if Path(candidate).exists() else shutil.which(candidate)
    return None


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


class _QuietHandler(SimpleHTTPRequestHandler):
    def log_message(self, *args) -> None:
        pass


# Runs in the page: wait for the first render (or the page's own error state), then
# exercise one detail-shard lookup and read the alerts count.
_PAGE_PROBE = """(async () => {
  const t0 = performance.now();
  while (!(window.__dashboard && window.__dashboard.timings.firstRender)) {
    const status = document.getElementById('status');
    if (status && status.classList.contains('error')) return {error: status.textContent};
    if (performance.now() - t0 > %d) return {error: 'no first render: ' + (status ? status.textContent : '?')};
    await new Promise(r => setTimeout(r, 200));
  }
  const d = window.__dashboard;
  const t1 = performance.now();
  while (!/^[0-9,]+$/.test(document.getElementById('alerts-count').textContent) && performance.now() - t1 < 15000)
    await new Promise(r => setTimeout(r, 200));
  let detail = null;
  try { detail = await d.detailRecord(0); } catch (e) { return {error: 'detail shard: ' + e.message}; }
  return {count: d.meta.count, alerts: document.getElementById('alerts-count').textContent,
          detailLabel: detail && detail.label, status: document.getElementById('status').textContent};
})()"""


def check_page(dist: Path, timeout_s: int = PAGE_TIMEOUT_S) -> list[str]:
    from websockets.sync.client import connect

    chrome = _chrome()
    if chrome is None:
        return ["headless Chrome check: no Chrome/Edge found (set AGNINETRA_CHROME)"]
    server = ThreadingHTTPServer(("127.0.0.1", 0), functools.partial(_QuietHandler, directory=str(dist)))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{server.server_address[1]}/"
    port = _free_port()
    profile = tempfile.mkdtemp(prefix="agninetra-preflight-")
    browser = subprocess.Popen(
        [chrome, "--headless=new", f"--remote-debugging-port={port}", f"--user-data-dir={profile}",
         "--window-size=1440,900", "--no-first-run", "--no-default-browser-check", "--enable-unsafe-swiftshader",
         "about:blank"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    try:
        import urllib.request

        targets = None
        for _ in range(100):
            try:
                targets = json.load(urllib.request.urlopen(f"http://127.0.0.1:{port}/json", timeout=2))
                break
            except OSError:
                time.sleep(0.2)
        if not targets:
            return ["headless Chrome check: browser did not start"]
        ws_url = next(t for t in targets if t["type"] == "page")["webSocketDebuggerUrl"]
        errors: list[str] = []
        with connect(ws_url, max_size=2**26, open_timeout=20) as ws:
            next_id = 0

            def send(method: str, **params):
                nonlocal next_id
                next_id += 1
                ws.send(json.dumps({"id": next_id, "method": method, "params": params}))
                while True:
                    message = json.loads(ws.recv(timeout=timeout_s + 30))
                    if message.get("method") == "Runtime.exceptionThrown":
                        details = message["params"]["exceptionDetails"]
                        errors.append(details.get("exception", {}).get("description", details.get("text", ""))[:300])
                    elif message.get("method") == "Log.entryAdded" and message["params"]["entry"]["level"] == "error":
                        entry = message["params"]["entry"]
                        errors.append(f"{entry['text'][:250]} ({entry.get('url', '')})")
                    elif message.get("method") == "Runtime.consoleAPICalled" and message["params"]["type"] == "error":
                        errors.append("console.error: " + " ".join(
                            str(a.get("value", a.get("description", ""))) for a in message["params"]["args"])[:300])
                    if message.get("id") == next_id:
                        return message.get("result", {})

            send("Page.enable")
            send("Runtime.enable")
            send("Log.enable")
            send("Page.navigate", url=url)
            result = send("Runtime.evaluate", expression=_PAGE_PROBE % (timeout_s * 1000),
                          awaitPromise=True, returnByValue=True)
            time.sleep(1.5)
            send("Runtime.evaluate", expression="1")  # drain errors logged after the probe
        failures = [f"page JavaScript error: {e}" for e in errors]
        if "exceptionDetails" in result:
            failures.append(f"page probe threw: {json.dumps(result['exceptionDetails'])[:300]}")
            return failures
        value = result.get("result", {}).get("value") or {}
        if value.get("error"):
            failures.append(f"page did not load: {value['error']}")
        elif not value.get("count"):
            failures.append("page loaded 0 points")
        elif not value.get("detailLabel"):
            failures.append("page could not decode a detail record")
        elif not str(value.get("alerts", "")).replace(",", "").isdigit():
            failures.append(f"alerts panel did not load (count shows {value.get('alerts')!r})")
        return failures
    finally:
        browser.kill()
        browser.wait(timeout=10)
        server.shutdown()
        shutil.rmtree(profile, ignore_errors=True)


def preflight(dist: Path = SITE_DIST_DIR) -> list[str]:
    failures = check_points(dist) + check_alerts(dist)
    if failures:  # don't bother loading a page whose data is already known bad
        return failures
    try:
        return check_page(dist)
    except Exception as error:  # the check itself broke: still a reason not to deploy
        return [f"headless Chrome check failed to run: {type(error).__name__}: {error}"]


if __name__ == "__main__":
    problems = preflight()
    print("Pre-deploy checks passed" if not problems else "Pre-deploy checks FAILED:\n  - " + "\n  - ".join(problems))
    raise SystemExit(1 if problems else 0)
