"""Build the static dashboard site in dist/ -- everything the browser needs as plain
files, so it can be served by any static host (Vercel) with no backend:

    dist/index.html, dist/static/...     the dashboard (copied from frontend/)
    dist/data/points.bin.gz              packed points (training/pack_map_points.py)
    dist/data/meta.json                  pack layout + class order + export time
    dist/data/classes.json               class names and colours
    dist/data/stats.json.gz              detections per day x class x state (+ anomalous)
    dist/data/alerts_history.csv         every alert raised by live ingestion
    dist/data/details/NNNN.json.gz       full records, DETAIL_SHARD_ROWS rows per shard
    dist/data/point_state.bin.gz         uint8 state index per row id (stats.json's state order)
    dist/data/states.json.gz             state outlines (simplified) + bounding boxes
    dist/data/facilities.json.gz         known facilities: GEM points + OSM industrial context

A click resolves a point's row id to shard row_id // DETAIL_SHARD_ROWS. Row ids follow
the pack's source order (historical files, then live months), so appending live data
only changes the last shards -- a redeploy re-uploads just those.

    python -m training.export_site                          # rebuild dist/
    python -m training.export_site --deploy                 # ...and deploy it to Vercel (production)
    python -m training.export_site --deploy --if-changed    # only if the data changed (scheduled runs)

--deploy runs training/site_preflight.py first (points file, alerts CSV, and the page
in headless Chrome); if any check fails the deploy is skipped and the reason printed.
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import re
import shutil
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from backend.classification.rules import CLASSES
from backend.config import (
    ALERTS_HISTORY_PATH,
    CLASS_COLORS,
    CLASS_COLORS_DARK,
    CLASS_DISPLAY_NAMES,
    CRITICAL_SITE_OSM_INDUSTRIAL_TAGS,
    CRITICAL_SITE_OSM_POWER_SOURCES,
    FRONTEND_DIR,
    GEM_FACILITIES_PATH,
    INDIA_BOUNDARY_PATH,
    MAP_POINTS_META_PATH,
    MAP_POINTS_PATH,
    NATIONAL_INDUSTRIAL_CONTEXT_PATH,
    ROOT_DIR,
    SITE_DIST_DIR,
)
from backend.ingestion.context_sources import exclude_renewable_power_plants
from training.report_by_state import assign_state

DIST_DIR = SITE_DIST_DIR
DETAIL_SHARD_ROWS = 10_000
VERCEL_PROJECT = "agninetra"

# Detail fields, encoded compactly: ints (distances in whole metres, coordinates x1e5),
# dictionary codes for repeated strings (decoded with details/index.json).
_DICT_FIELDS = (
    "label", "satellite", "daynight", "nearest_heat_facility_type", "nearest_flare_facility_type",
    "nearest_coal_source", "osm_industrial_tag", "label_source",
)
_INT_FIELDS = (
    "acq_time", "recurrence_count", "landcover_class",
    "dist_to_heat_industry_m", "dist_to_flare_capable_m", "dist_to_coal_mine_m", "dist_to_industrial_m",
)
_DETAIL_COLUMNS = [
    "latitude", "longitude", "acq_date", "frp", "is_anomalous", *_INT_FIELDS, *_DICT_FIELDS,
]
DEPLOY_MARKER = ".deployed"  # dist/.deployed: written after each successful deploy
OFFSHORE_STATE = "(offshore/border)"
FACILITY_GROUPS = [
    "GEM facility (flare-capable / heat industry)",
    "OSM refinery or fossil / nuclear power plant",
    "Other OSM industrial site",
]
_SITE_FILES = ("index.html", "static/css/dashboard.css", "static/js/dashboard.js", "static/js/alerts-panel.js")
_VERCEL_JSON = {
    "cleanUrls": True,
    "headers": [
        # always revalidate: data changes every ingest run; unchanged files answer 304
        {"source": "/(.*)", "headers": [{"key": "Cache-Control", "value": "public, max-age=0, must-revalidate"}]},
    ],
}


def _write_gz_json(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(obj, separators=(",", ":")).encode()
    # mtime=0: identical content -> identical bytes, so unchanged shards aren't re-uploaded
    with open(path, "wb") as raw, gzip.GzipFile(fileobj=raw, mode="wb", compresslevel=9, mtime=0) as gz:
        gz.write(data)


def _nullable_ints(values: pd.Series, scale: float = 1.0) -> list:
    numbers = pd.to_numeric(values, errors="coerce") * scale
    return [None if pd.isna(v) else int(round(v)) for v in numbers]


def _write_gz_bytes(data: bytes, path: Path) -> None:
    with open(path, "wb") as raw, gzip.GzipFile(fileobj=raw, mode="wb", compresslevel=9, mtime=0) as gz:
        gz.write(data)


def _state_outlines(states_path: Path, names: list[str]) -> dict:
    """Simplified outline + bbox per state, keyed by the names stats.json uses."""
    states = gpd.read_parquet(states_path)[["name", "geometry"]]
    states = states[states["name"].isin(names)]
    outlines = gpd.GeoDataFrame({"name": states["name"].to_numpy()},
                                geometry=states.geometry.simplify(0.01, preserve_topology=True).to_numpy(), crs=states.crs)
    bounds = {n: [round(v, 4) for v in g.bounds] for n, g in zip(states["name"], states.geometry)}
    return {"bbox": bounds, "outlines": json.loads(outlines.to_json(drop_id=True))}


def _facility_frame(facilities_path: Path, industrial_context_path: Path) -> pd.DataFrame:
    """Known facilities as points (polygons -> a point inside them), with a display group:
    the same GEM and OSM inputs the labelling rules measure distances to, with solar/wind
    power plants excluded exactly as the pipeline excludes them."""
    parts = []
    if Path(facilities_path).exists():
        gem = pd.read_csv(facilities_path)
        parts.append(pd.DataFrame({
            "lat": gem["latitude"], "lon": gem["longitude"], "group": 0,
            "kind": gem["facility_type"].astype(str).str.replace("_", " "),
            "name": gem["facility_name"], "source": "GEM",
        }))
    if Path(industrial_context_path).exists():
        osm = exclude_renewable_power_plants(gpd.read_parquet(industrial_context_path))
        col = lambda name: osm[name] if name in osm else pd.Series(None, index=osm.index, dtype="object")
        tag = osm["matched_tag"].astype(str)
        value = pd.Series([osm.at[i, t] if t in osm else None for i, t in zip(osm.index, tag)], index=osm.index)
        plant_source = col("plant:source").fillna("").astype(str).str.lower()
        fossil = plant_source.str.split(";").map(lambda ps: any(p.strip() in CRITICAL_SITE_OSM_POWER_SOURCES for p in ps))
        is_plant = col("power") == "plant"
        critical = col("industrial").isin(CRITICAL_SITE_OSM_INDUSTRIAL_TAGS) | (is_plant & fossil)
        kind = tag + "=" + value.astype(str)
        kind = kind.where(~(is_plant & (plant_source != "")), kind + " (" + plant_source + ")")
        points = osm.geometry.representative_point()
        parts.append(pd.DataFrame({
            "lat": points.y.to_numpy(), "lon": points.x.to_numpy(), "group": np.where(critical, 1, 2),
            "kind": kind.to_numpy(), "name": col("name").to_numpy(), "source": "OSM",
        }))
    if not parts:
        return pd.DataFrame(columns=["lat", "lon", "group", "kind", "name", "source"])
    return pd.concat(parts, ignore_index=True).dropna(subset=["lat", "lon"])


def _encode_facilities(frame: pd.DataFrame) -> dict:
    kinds = sorted(frame["kind"].astype(str).unique())
    kind_code = {k: i for i, k in enumerate(kinds)}
    return {
        "groups": FACILITY_GROUPS, "kinds": kinds, "sources": ["GEM", "OSM"],
        "lat": _nullable_ints(frame["lat"], 1e5), "lon": _nullable_ints(frame["lon"], 1e5),
        "group": frame["group"].astype(int).tolist(),
        "kind": frame["kind"].astype(str).map(kind_code).tolist(),
        "source": (frame["source"] == "OSM").astype(int).tolist(),
        "name": [None if pd.isna(n) or str(n).strip() == "" else str(n) for n in frame["name"]],
    }


def export(
    dist: Path = DIST_DIR,
    *,
    meta_path: Path = MAP_POINTS_META_PATH,
    points_path: Path = MAP_POINTS_PATH,
    alerts_history: Path = ALERTS_HISTORY_PATH,
    frontend_dir: Path = FRONTEND_DIR,
    states_path: Path = INDIA_BOUNDARY_PATH,
    facilities_path: Path = GEM_FACILITIES_PATH,
    industrial_context_path: Path = NATIONAL_INDUSTRIAL_CONTEXT_PATH,
) -> dict:
    started = time.perf_counter()
    dist = Path(dist)
    meta = json.loads(Path(meta_path).read_text())
    data_dir = dist / "data"
    if data_dir.exists():
        shutil.rmtree(data_dir)
    data_dir.mkdir(parents=True)

    # --- site files (keep dist/.vercel -- the project link) ---
    for rel in _SITE_FILES:
        target = dist / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(Path(frontend_dir) / rel, target)
    (dist / "vercel.json").write_text(json.dumps(_VERCEL_JSON, indent=2))

    # --- points + meta + classes ---
    shutil.copy2(Path(points_path).with_name(Path(points_path).name + ".gz"), data_dir / "points.bin.gz")
    public_meta = {k: v for k, v in meta.items() if k != "sources"}
    public_meta["exported_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    public_meta["detail_shard_rows"] = DETAIL_SHARD_ROWS
    (data_dir / "meta.json").write_text(json.dumps(public_meta, indent=1))
    (data_dir / "classes.json").write_text(json.dumps(
        {"classes": list(CLASSES), "colors": CLASS_COLORS, "colors_dark": CLASS_COLORS_DARK,
         "display": CLASS_DISPLAY_NAMES}, indent=1, ensure_ascii=False
    ), encoding="utf-8")

    # --- all rows, in row-id order (the pack's source order) ---
    frames = []
    for source in meta["sources"]:
        path = Path(source["path"]) if Path(source["path"]).is_absolute() else ROOT_DIR / source["path"]
        present = set(pq.read_schema(path).names)
        frame = pd.read_parquet(path, columns=[c for c in _DETAIL_COLUMNS if c in present])
        frames.append(frame.reindex(columns=_DETAIL_COLUMNS))  # absent columns -> missing values
    rows = pd.concat(frames, ignore_index=True)
    assert len(rows) == meta["count"], "pack is out of date with its sources -- repack first"
    base = pd.Timestamp(meta["base_date"])
    day = (pd.to_datetime(rows["acq_date"].astype(str).str[:10]) - base).dt.days.to_numpy()
    class_id = rows["label"].map({c: i for i, c in enumerate(CLASSES)}).to_numpy()

    # --- stats: detections per (day, class, state) ---
    state = assign_state(rows[["latitude", "longitude"]], states_path).fillna(OFFSHORE_STATE)
    states = sorted(state.unique())
    assert len(states) < 256, "state index is stored as uint8"
    state_id = state.map({s: i for i, s in enumerate(states)}).to_numpy().astype(np.uint8)
    _write_gz_bytes(state_id.tobytes(), data_dir / "point_state.bin.gz")  # indexed by row id
    _write_gz_json(_state_outlines(states_path, states), data_dir / "states.json.gz")
    _write_gz_json(_encode_facilities(_facility_frame(facilities_path, industrial_context_path)),
                   data_dir / "facilities.json.gz")
    table = pd.DataFrame({
        "day": day, "cls": class_id, "state": state_id,
        "anom": rows["is_anomalous"].fillna(False).astype(bool).to_numpy(),
    }).groupby(["day", "cls", "state"]).agg(n=("anom", "size"), anom=("anom", "sum")).reset_index()
    _write_gz_json({
        "base_date": meta["base_date"], "classes": list(CLASSES), "states": states,
        "day": table["day"].tolist(), "cls": table["cls"].tolist(), "state": table["state"].tolist(),
        "n": table["n"].tolist(), "anom": table["anom"].astype(int).tolist(),
    }, data_dir / "stats.json.gz")

    # --- alerts ---
    if Path(alerts_history).exists():
        shutil.copy2(alerts_history, data_dir / "alerts_history.csv")

    # --- detail shards ---
    dicts = {}
    codes = {}
    for field in _DICT_FIELDS:
        values = rows[field].astype("string").fillna("")
        uniques = sorted(set(values) - {""})
        dicts[field] = uniques
        lookup = {v: i for i, v in enumerate(uniques)}
        codes[field] = values.map(lambda v: lookup.get(v, -1)).to_numpy()  # -1 = missing
    encoded = {
        "lat": _nullable_ints(rows["latitude"], 1e5),
        "lon": _nullable_ints(rows["longitude"], 1e5),
        "day": day.tolist(),
        "frp": _nullable_ints(rows["frp"], 100),  # centi-MW
        "anom": rows["is_anomalous"].fillna(False).astype(int).tolist(),
        **{f: _nullable_ints(rows[f]) for f in _INT_FIELDS},
        **{f: codes[f].tolist() for f in _DICT_FIELDS},
    }
    shards = (len(rows) + DETAIL_SHARD_ROWS - 1) // DETAIL_SHARD_ROWS
    for shard in range(shards):
        lo, hi = shard * DETAIL_SHARD_ROWS, min(len(rows), (shard + 1) * DETAIL_SHARD_ROWS)
        _write_gz_json({k: v[lo:hi] for k, v in encoded.items()}, data_dir / "details" / f"{shard:04d}.json.gz")
    (data_dir / "details" / "index.json").write_text(json.dumps({
        "shard_rows": DETAIL_SHARD_ROWS, "shards": shards, "base_date": meta["base_date"],
        "scales": {"lat": 1e5, "lon": 1e5, "frp": 100}, "dicts": dicts,
    }))

    size = sum(p.stat().st_size for p in dist.rglob("*") if p.is_file() and ".vercel" not in p.parts)
    files = sum(1 for p in dist.rglob("*") if p.is_file() and ".vercel" not in p.parts)
    return {"rows": len(rows), "shards": shards, "bytes": size, "files": files,
            "seconds": round(time.perf_counter() - started, 1), "data_through": meta["max_date"]}


def is_stale(
    reference: Path,
    meta_path: Path = MAP_POINTS_META_PATH,
    alerts_history: Path = ALERTS_HISTORY_PATH,
    frontend_dir: Path = FRONTEND_DIR,
) -> bool:
    """True if the pack, the alert history, or a page file changed since `reference` was
    written (dist/data/meta.json for an export, dist/.deployed for a deploy -- so a deploy
    that was skipped or failed is retried on the next run)."""
    if not Path(reference).exists():
        return True
    built = Path(reference).stat().st_mtime
    inputs = [Path(meta_path), Path(alerts_history), *(Path(frontend_dir) / rel for rel in _SITE_FILES)]
    return any(p.exists() and p.stat().st_mtime > built for p in inputs)


def deploy(dist: Path = DIST_DIR) -> str:
    """vercel deploy --prod of dist/ (links the project on first use). Returns the URL."""
    vercel = shutil.which("vercel") or shutil.which("vercel.cmd")
    if vercel is None:
        raise SystemExit("Vercel CLI not found -- npm i -g vercel, then vercel login")
    env = {**os.environ, "NO_COLOR": "1", "NO_UPDATE_NOTIFIER": "1"}
    if not (dist / ".vercel" / "project.json").exists():
        subprocess.run([vercel, "link", "--yes", "--project", VERCEL_PROJECT, "--cwd", str(dist)],
                       check=True, env=env, capture_output=True, text=True)
    result = subprocess.run([vercel, "deploy", "--prod", "--yes", "--cwd", str(dist)],
                            check=True, env=env, capture_output=True, text=True)
    urls = re.findall(r"https://[\w.-]+\.vercel\.app", result.stdout + result.stderr)
    (dist / DEPLOY_MARKER).write_text(datetime.now(timezone.utc).isoformat(timespec="seconds"))
    return urls[0] if urls else "(deployed; URL not found in CLI output)"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--deploy", action="store_true", help="deploy dist/ to Vercel (production) after exporting")
    parser.add_argument("--if-changed", action="store_true", help="do nothing unless the data changed since the last export")
    return parser.parse_args()


if __name__ == "__main__":
    from training.site_preflight import preflight

    args = _parse_args()
    stamp = lambda: datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    reference = DIST_DIR / DEPLOY_MARKER if args.deploy else DIST_DIR / "data" / "meta.json"
    if args.if_changed and not is_stale(reference):
        print(f"[{stamp()}] Site export skipped: nothing changed since the last {'deploy' if args.deploy else 'export'}")
        raise SystemExit(0)
    summary = export()
    print(f"[{stamp()}] Exported {summary['rows']:,} detections through {summary['data_through']} -> {DIST_DIR} "
          f"({summary['files']} files, {summary['bytes'] / 1e6:.1f} MB, {summary['shards']} detail shards) "
          f"in {summary['seconds']} s")
    if args.deploy:
        failures = preflight(DIST_DIR)
        if failures:
            print(f"[{stamp()}] Deploy skipped -- pre-deploy checks failed:")
            for failure in failures:
                print(f"  - {failure}")
            raise SystemExit(1)
        print(f"[{stamp()}] Pre-deploy checks passed")
        print(f"[{stamp()}] Deployed:", deploy())
