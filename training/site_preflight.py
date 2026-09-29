"""Pre-deploy checks for the exported dashboard in dist/ (training/export_site.py).

    python -m training.site_preflight        # check dist/ without deploying

preflight() returns a list of failure reasons (empty = safe to deploy):

1. data/points.bin.gz exists, is non-empty, and decompresses to the byte count and
   point count data/meta.json promises;
2. data/alerts_history.csv parses, has the columns the alerts panel reads, and every
   row has a coordinate, a date and a known alert type;
3. consistency: an independent recount from the source parquets (the files the map
   pack lists) and training/data/live/alerts_history.csv must match the exported data
   exactly -- total, per class, per state x class, anomalous, the latest detection time,
   every point inside an Indian state (no offshore/border bucket), and the industrial
   sub-types (display-only): they cover exactly the industrial detections -- no other
   class carries one -- and their per-sub-type table sums to the industrial class total;
4. the page loads in headless Chrome, served from dist/ over local HTTP exactly as
   a static host would, with no JavaScript exception, console error or failed request,
   and what it *displays* matches the recount: the detections KPI, class rings, top
   states, alert counts per type and the "Data through" time, for All India / all
   year, the latest month, Gujarat, Gujarat with a class switched off, and Gujarat with
   one industrial sub-type switched off -- and in each view the number of points the GPU
   filter draws equals the KPI.
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

import duckdb
import numpy as np
import pandas as pd

from backend.classification.rules import CLASSES
from backend.config import (
    ALERTS_HISTORY_PATH,
    CLASS_COLORS,
    CLASS_COLORS_DARK,
    CLASS_DISPLAY_NAMES,
    MAP_POINTS_META_PATH,
    ROOT_DIR,
    SITE_DIST_DIR,
)
from training.industrial_subtype import SUBTYPE_COLORS
from training.report_by_state import assign_state

ALERT_COLUMNS = ("alert_id", "alert_type", "latitude", "longitude", "acq_date")
ALERT_TYPES = ("industrial_anomaly", "new_activity_at_critical_site", "new_unmapped_source", "large_fire_event")
# as the alerts panel names them (frontend/static/js/alerts-panel.js TYPES)
ALERT_TYPE_NAMES = {
    "industrial_anomaly": "Industrial anomaly", "new_activity_at_critical_site": "New activity at critical site",
    "new_unmapped_source": "New unmapped source", "large_fire_event": "Large fire event",
}
AUDIT_STATE = "Gujarat"  # the state view the page audit checks
AUDIT_CLASS_OFF = "agricultural burning"  # switched off in one audit view
AUDIT_SUBTYPE_OFF = 1  # industrial sub-type code (Refinery / oil & gas) switched off in the last audit view
OFFSHORE_STATE = "(offshore/border)"
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


def _fmt_count(n: int) -> str:
    return f"{int(n):,}"


def ist_label(utc_iso: str) -> str:
    """'2026-09-28T08:52Z' -> '28 Sep 2026, 14:22 IST' (the dashboard's format)."""
    t = pd.Timestamp(utc_iso.rstrip("Z")) + pd.Timedelta(minutes=330)
    return f"{t.day} {t.strftime('%b %Y')}, {t.strftime('%H:%M')} IST"


def source_counts(meta_path: Path = MAP_POINTS_META_PATH, states_path: Path | None = None) -> dict:
    """Recount everything the dashboard shows, straight from the source parquets."""
    sources = json.loads(Path(meta_path).read_text())["sources"]
    paths = [Path(s["path"]) if Path(s["path"]).is_absolute() else ROOT_DIR / s["path"] for s in sources]
    listing = "[" + ",".join("'" + p.as_posix().replace("'", "''") + "'" for p in paths) + "]"
    rows = duckdb.sql(
        f"select latitude, longitude, label, cast(acq_date as varchar) as acq_date, "
        f"cast(acq_time as varchar) as acq_time, coalesce(cast(is_anomalous as boolean), false) as anom "
        f"from read_parquet({listing}, union_by_name=true)"
    ).df()
    rows["acq_date"] = rows["acq_date"].str[:10]
    state = (assign_state(rows[["latitude", "longitude"]], states_path) if states_path
             else assign_state(rows[["latitude", "longitude"]])).fillna(OFFSHORE_STATE)
    hhmm = pd.to_numeric(rows["acq_time"], errors="coerce").fillna(0).astype(int)
    stamps = pd.to_datetime(rows["acq_date"]) + pd.to_timedelta(hhmm // 100, unit="h") + pd.to_timedelta(hhmm % 100, unit="m")
    by_state_class = rows.assign(state=state).groupby(["state", "label"]).size()
    latest_month = rows["acq_date"].max()[:7]
    in_state = state == AUDIT_STATE
    return {
        "total": len(rows),
        "anomalous": int(rows["anom"].sum()),
        "by_class": {c: int(v) for c, v in rows["label"].value_counts().items()},
        "by_state_class": {f"{s}|{c}": int(v) for (s, c), v in by_state_class.items()},
        "by_state": {s: int(v) for s, v in state.value_counts().items()},
        "max_date": rows["acq_date"].max(),
        "data_through_utc": stamps.max().strftime("%Y-%m-%dT%H:%MZ"),
        "latest_month": latest_month,
        "latest_month_total": int((rows["acq_date"].str[:7] == latest_month).sum()),
        "audit_state_by_class": {c: int(v) for c, v in rows.loc[in_state, "label"].value_counts().items()},
    }


def check_consistency(dist: Path, expected: dict, alerts_source: Path = ALERTS_HISTORY_PATH) -> list[str]:
    """The exported files against the source recount (source_counts)."""
    data = Path(dist) / "data"
    failures = []
    meta = json.loads((data / "meta.json").read_text())
    stats = json.loads(gzip.decompress((data / "stats.json.gz").read_bytes()))
    n = np.array(stats["n"])
    exported_class = {CLASSES[c]: int(n[np.array(stats["cls"]) == c].sum()) for c in set(stats["cls"])}
    exported_state_class = {}
    for s_i, c_i, count in zip(stats["state"], stats["cls"], stats["n"]):
        key = f"{stats['states'][s_i]}|{CLASSES[c_i]}"
        exported_state_class[key] = exported_state_class.get(key, 0) + count
    raw = gzip.decompress((data / "points.bin.gz").read_bytes())
    layout = meta["layout"]["class_id"]
    drawn = np.bincount(np.frombuffer(raw, np.uint8, count=layout["length"], offset=layout["offset"]), minlength=len(CLASSES))
    drawn_class = {CLASSES[i]: int(v) for i, v in enumerate(drawn) if v}
    point_state = np.frombuffer(gzip.decompress((data / "point_state.bin.gz").read_bytes()), np.uint8)

    def same(what, got, want):
        if got != want:
            failures.append(f"consistency: {what}: exported {got!r} != source {want!r}")

    same("total detections (meta.json)", meta["count"], expected["total"])
    same("total detections (stats.json)", int(n.sum()), expected["total"])
    same("points per row id (point_state.bin)", len(point_state), expected["total"])
    same("detections per class (stats.json)", exported_class, expected["by_class"])
    same("points drawn per class (points.bin)", drawn_class, expected["by_class"])
    diff = {k for k in set(exported_state_class) | set(expected["by_state_class"])
            if exported_state_class.get(k) != expected["by_state_class"].get(k)}
    if diff:
        sample = sorted(diff)[:5]
        failures.append(f"consistency: per state x class differs for {len(diff)} cells, e.g. "
                        + ", ".join(f"{k}: {exported_state_class.get(k)} vs {expected['by_state_class'].get(k)}" for k in sample))
    same("anomalous detections", int(sum(stats["anom"])), expected["anomalous"])
    same("data through (date)", meta["max_date"], expected["max_date"])
    same("data through (latest detection, UTC)", meta.get("data_through_utc"), expected["data_through_utc"])
    # colours: class colours are the configured palette, sub-type colours the module's
    classes_json = json.loads((data / "classes.json").read_text(encoding="utf-8"))
    same("class colours (classes.json)", classes_json["colors_dark"], CLASS_COLORS_DARK)
    same("light class colours (classes.json)", classes_json["colors"], CLASS_COLORS)
    same("sub-type colours (classes.json)", {t["name"]: t["color"] for t in classes_json["subtypes"]}, SUBTYPE_COLORS)
    # industrial sub-types: display-only, so they must line up with the class exactly
    sub = stats.get("sub")
    if sub is None:
        failures.append("consistency: stats.json has no industrial sub-type table")
    else:
        same("industrial sub-type total (stats.json sub)", int(sum(sub["n"])), expected["by_class"].get("industrial", 0))
        point_sub = np.frombuffer(gzip.decompress((data / "point_subtype.bin.gz").read_bytes()), np.uint8)
        same("sub-type codes per row id (point_subtype.bin)", len(point_sub), expected["total"])
        row_ids = np.frombuffer(raw, np.uint32, count=meta["layout"]["row_id"]["length"], offset=meta["layout"]["row_id"]["offset"])
        class_ids = np.frombuffer(raw, np.uint8, count=layout["length"], offset=layout["offset"])
        has_sub = point_sub[row_ids] > 0
        is_industrial = class_ids == CLASSES.index("industrial")
        if len(point_sub) == expected["total"] and (has_sub != is_industrial).any():
            failures.append(f"consistency: {int((has_sub != is_industrial).sum()):,} points carry a sub-type without being "
                            "industrial, or are industrial without one")
    outside = expected["by_state"].get(OFFSHORE_STATE, 0)
    if outside or OFFSHORE_STATE in stats["states"]:
        failures.append(f"consistency: {outside:,} detections fall outside every Indian state polygon")
    exported_alerts, source_alerts = data / "alerts_history.csv", Path(alerts_source)
    if source_alerts.exists() and exported_alerts.read_bytes() != source_alerts.read_bytes():
        failures.append("consistency: data/alerts_history.csv differs from training/data/live/alerts_history.csv")
    return failures


def subtype_count(dist: Path, state_name: str, code: int) -> int:
    """Industrial detections of one sub-type in one state, from the exported stats.json."""
    stats = json.loads(gzip.decompress((Path(dist) / "data" / "stats.json.gz").read_bytes()))
    index = stats["states"].index(state_name)
    sub = stats["sub"]
    return int(sum(n for n, c, st in zip(sub["n"], sub["code"], sub["state"]) if c == code and st == index))


def expected_display(expected: dict, alerts_csv: Path, sub_off_count: int = 0) -> dict:
    """What the page must show, formatted the way the dashboard formats it."""
    name = lambda c: CLASS_DISPLAY_NAMES.get(c, c)
    states = sorted(((s, v) for s, v in expected["by_state"].items() if s != OFFSHORE_STATE), key=lambda sv: -sv[1])
    alerts = pd.read_csv(alerts_csv, dtype=str, keep_default_na=False)
    by_type = alerts["alert_type"].value_counts()
    audit = expected["audit_state_by_class"]
    audit_total = sum(audit.values())
    return {
        "all": {
            "kpi": _fmt_count(expected["total"]), "visible": expected["total"],
            "rings": {name(c): _fmt_count(expected["by_class"].get(c, 0)) for c in CLASSES},
            "states": {s: _fmt_count(v) for s, v in states},
            "top_state": states[0][0] if states else None,
            "chips": [f"All {len(alerts)}"] + [f"{ALERT_TYPE_NAMES[t]} {by_type[t]}" for t in ALERT_TYPES if by_type.get(t)],
            "data_through": f"Data through {ist_label(expected['data_through_utc'])}",
        },
        "month": {"kpi": _fmt_count(expected["latest_month_total"]), "visible": expected["latest_month_total"],
                  "key": expected["latest_month"]},
        "state": {"kpi": _fmt_count(audit_total), "visible": audit_total,
                  "rings": {name(c): _fmt_count(audit.get(c, 0)) for c in CLASSES}},
        "state_minus": {"kpi": _fmt_count(audit_total - audit.get(AUDIT_CLASS_OFF, 0)),
                        "visible": audit_total - audit.get(AUDIT_CLASS_OFF, 0)},
        "sub_minus": {"kpi": _fmt_count(audit_total - sub_off_count), "visible": audit_total - sub_off_count},
    }


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
  const out = {count: d.meta.count, alerts: document.getElementById('alerts-count').textContent,
               detailLabel: detail && detail.label, status: document.getElementById('status').textContent};
  // display audit: read what each view shows once the GPU filter count has settled
  const pause = (ms) => new Promise(r => setTimeout(r, ms));
  const settle = async () => {
    let last = null, stable = 0;
    for (let i = 0; i < 80 && stable < 3; i++) {
      await new Promise(r => requestAnimationFrame(() => r())); await pause(60);
      stable = d.state.visible === last ? stable + 1 : 0; last = d.state.visible;
    }
  };
  const read = () => ({
    kpi: document.querySelector('.kpi .kpi-value').textContent,
    visible: d.state.visible,
    rings: Object.fromEntries([...document.querySelectorAll('.ring-btn')].map(b =>
      [b.querySelector('.ring-name').textContent, b.querySelector('.ring-count').textContent])),
    states: [...document.querySelectorAll('.state-row')].map(r =>
      [r.querySelector('.state-name').textContent, r.querySelector('.state-count').textContent]),
  });
  await settle();
  out.all = {...read(), status: document.getElementById('status').textContent,
             chips: [...document.querySelectorAll('#alerts-filters button')].map(b => b.textContent.trim())};
  d.state.allYear = false; d.state.monthIndex = d.months.length - 1; d.update();
  await settle(); out.month = {...read(), key: d.months[d.state.monthIndex].key};
  d.state.allYear = true; d.selectRegion(d.states.indexOf('%s'));
  await settle(); out.state = read();
  d.state.active.delete(d.meta.classes.indexOf('%s')); d.update({classesChanged: true});
  await settle(); out.state_minus = read();
  d.state.active.add(d.meta.classes.indexOf('%s')); d.setSubtypeActive(%d, false);
  await settle(); out.sub_minus = read();
  d.setSubtypeActive(%d, true);
  // colour modes: class mode is exactly the class palette in classes.json; industrial-type mode colours only
  // industrial points (their sub-type colour) and dims everything else
  const info = await (await fetch('data/classes.json')).json();
  const hex = (h) => [1, 3, 5].map((k) => parseInt(h.slice(k, k + 2), 16));
  const colourErrors = [], ind = d.meta.classes.indexOf('industrial');
  const same = (a, b) => a.length === b.length && a.every((v, k) => v === b[k]);
  for (let i = 0; i < d.N; i += Math.max(1, Math.floor(d.N / 600))) {
    const {cls, sub} = d.pointInfo(i);
    const classWant = [...hex(info.colors_dark[d.meta.classes[cls]]), 255];
    const notIdentified = sub === info.subtype_not_identified, ni = info.subtype_style.not_identified;
    const typeWant = cls === ind && sub > 0 ? [...hex(info.subtypes[sub - 1].color), notIdentified ? ni.alpha : 255] : [...hex(info.subtype_dim.color), info.subtype_dim.alpha];
    const radiusWant = cls === ind && notIdentified ? Math.fround(ni.radius_scale) : 1;
    if (d.radiusFor(i, 'class') !== 1) colourErrors.push('class mode point ' + i + ': radius ' + d.radiusFor(i, 'class') + ' != 1');
    if (d.radiusFor(i, 'type') !== radiusWant) colourErrors.push('type mode point ' + i + ': radius ' + d.radiusFor(i, 'type') + ' != ' + radiusWant);
    if (!same(d.fillColor(i, 'class'), classWant)) colourErrors.push('class mode point ' + i + ': ' + d.fillColor(i, 'class') + ' != ' + classWant);
    if (!same(d.fillColor(i, 'type'), typeWant)) colourErrors.push('type mode point ' + i + ': ' + d.fillColor(i, 'type') + ' != ' + typeWant);
    if (colourErrors.length > 5) break;
  }
  out.colour_errors = colourErrors; out.colour_mode_default = d.state.colorMode;
  return out;
})()"""


def _compare_display(shown: dict, want: dict) -> list[str]:
    failures = []
    for view, label in (("all", "All India, all year"), ("month", "latest month"),
                        ("state", AUDIT_STATE), ("state_minus", f"{AUDIT_STATE} without {AUDIT_CLASS_OFF}"),
                        ("sub_minus", f"{AUDIT_STATE} without industrial sub-type {AUDIT_SUBTYPE_OFF}")):
        got, exp = shown.get(view) or {}, want[view]
        if got.get("kpi") != exp["kpi"]:
            failures.append(f"page ({label}): Detections KPI shows {got.get('kpi')!r}, source says {exp['kpi']!r}")
        if got.get("visible") != exp["visible"]:
            failures.append(f"page ({label}): map draws {got.get('visible')!r} points, source says {exp['visible']!r}")
        if "rings" in exp and got.get("rings") != exp["rings"]:
            failures.append(f"page ({label}): class counts {got.get('rings')} != source {exp['rings']}")
    if shown.get("colour_mode_default") != "class":
        failures.append(f"page: default colour mode is {shown.get('colour_mode_default')!r}, not 'class'")
    for message in shown.get("colour_errors") or []:
        failures.append(f"page: {message}")
    got, exp = shown.get("all") or {}, want["all"]
    for name, count in got.get("states", []):
        if exp["states"].get(name) != count:
            failures.append(f"page: top states shows {name} {count}, source says {exp['states'].get(name)}")
    if got.get("states") and got["states"][0][0] != exp["top_state"]:
        failures.append(f"page: top state is {got['states'][0][0]!r}, source says {exp['top_state']!r}")
    if got.get("chips") != exp["chips"]:
        failures.append(f"page: alert counts {got.get('chips')} != alerts_history.csv {exp['chips']}")
    if exp["data_through"] not in (got.get("status") or ""):
        failures.append(f"page: status {got.get('status')!r} lacks {exp['data_through']!r}")
    if (shown.get("month") or {}).get("key") != want["month"]["key"]:
        failures.append(f"page: latest month is {(shown.get('month') or {}).get('key')!r}, source says {want['month']['key']!r}")
    return failures


def check_page(dist: Path, timeout_s: int = PAGE_TIMEOUT_S, want: dict | None = None) -> list[str]:
    from websockets.sync.client import connect

    chrome = _chrome()
    if chrome is None:
        return ["headless Chrome check: no Chrome/Edge found (set AGNINETRA_CHROME)"]
    server = ThreadingHTTPServer(("127.0.0.1", 0), functools.partial(_QuietHandler, directory=str(dist)))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{server.server_address[1]}/?all=1"
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
            result = send("Runtime.evaluate", expression=_PAGE_PROBE % (timeout_s * 1000, AUDIT_STATE, AUDIT_CLASS_OFF, AUDIT_CLASS_OFF, AUDIT_SUBTYPE_OFF, AUDIT_SUBTYPE_OFF),
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
        elif want is not None:
            failures.extend(_compare_display(value, want))
        return failures
    finally:
        browser.kill()
        browser.wait(timeout=10)
        server.shutdown()
        shutil.rmtree(profile, ignore_errors=True)


def preflight(dist: Path = SITE_DIST_DIR, meta_path: Path = MAP_POINTS_META_PATH,
              alerts_source: Path = ALERTS_HISTORY_PATH) -> list[str]:
    failures = check_points(dist) + check_alerts(dist)
    if failures:  # don't bother with the rest when the data is already known bad
        return failures
    expected = source_counts(meta_path)
    failures = check_consistency(dist, expected, alerts_source)
    if failures:
        return failures
    try:
        sub_off = subtype_count(dist, AUDIT_STATE, AUDIT_SUBTYPE_OFF)
        return check_page(dist, want=expected_display(expected, Path(dist) / "data" / "alerts_history.csv", sub_off))
    except Exception as error:  # the check itself broke: still a reason not to deploy
        return [f"headless Chrome check failed to run: {type(error).__name__}: {error}"]


if __name__ == "__main__":
    problems = preflight()
    print("Pre-deploy checks passed" if not problems else "Pre-deploy checks FAILED:\n  - " + "\n  - ".join(problems))
    raise SystemExit(1 if problems else 0)
