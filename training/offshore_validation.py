"""Validation of the offshore gas-flare rule, run BEFORE it is adopted (see README, "Offshore gas flares").

    python -m training.offshore_validation sample      # 30 recurring offshore cells + 15 (+15 spare) open-sea controls
    python -m training.offshore_validation swir        # Sentinel-2 B12 hot-pixel check on both
    python -m training.offshore_validation compare     # cell locations vs OSM offshore platforms and GEM oil & gas fields
    python -m training.offshore_validation clusters    # other persistent offshore clusters (review list, not rules)

PASS RULE (fixed before any result was seen; not tuned afterwards): a hot pixel within 200 m in
at least 40% of the offshore cells that have two or more clear scenes, AND a hot pixel within
200 m in at most 1 of the 15 open-sea controls. 40% sits below the onshore flare hit rate
(64%, README "Five ways to separate a flare...") because offshore flares may be smaller.

Choices made before running, so the check can't be bent to the outcome:
  * cells: ~375 m grid cells (grid_keys) of the offshore history with recurrence_count >= 5
    (OFFSHORE_MIN_RECURRENCE), 30 drawn without replacement, seed SEED. Location = mean of
    the cell's detections. Neighbouring cells of one platform are separate cells.
  * controls: uniform random points in the zone boxes (weighted by area), each >= 10 km
    beyond the India boundary (the offshore definition) and >= CONTROL_MIN_GAP_KM from every
    offshore detection ever recorded. Each gets the dates of random offshore detections in
    its zone. 15 are drawn; a control without two clear scenes is replaced from the same
    seeded stream (offshore cells are NOT replaced: they leave the denominator, as the
    rule says).
  * scenes: for up to three anchor dates (three random detection dates of the cell), the
    Sentinel-2 L2A scene nearest the anchor within +-SCENE_WINDOW_DAYS whose 380 m box at
    20 m is at least 80% clear (SCL 4-7). Two or more distinct clear scenes = evaluable.
  * hot pixel: the onshore method (README): B12 reflectance > 0.30 and > median + 4 x MAD of the
    box's clear pixels; a hit needs one within 200 m of the point in at least one scene.
"""

from __future__ import annotations

import argparse
import io
import json
import math
import os
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import requests

from backend.config import (
    GEM_MULTISECTOR_FACILITIES_PATH,
    INDIA_BOUNDARY_PATH,
    NATIONAL_PROJECTED_CRS,
    OFFSHORE_DIR,
    OFFSHORE_HISTORY_PATH,
    OFFSHORE_MIN_DISTANCE_KM,
    OFFSHORE_MIN_RECURRENCE,
    OFFSHORE_ZONES,
)
from backend.ingestion.firms_client import _env
from backend.ingestion.offshore import offshore_columns
from training.spatial_features import grid_keys

SEED = 20260930
N_CELLS = 30
N_CONTROLS = 15
CONTROL_POOL = 40  # seeded stream length: 15 needed, the rest are replacements
CONTROL_MIN_GAP_KM = 5.0
ANCHORS = 3
SCENE_WINDOW_DAYS = 3
MIN_CLEAR_FRACTION = 0.8
MIN_CLEAR_SCENES = 2
HOT_B12_MIN = 0.30
HOT_MAD_MULTIPLE = 4.0
HIT_RADIUS_M = 200.0
BOX_HALF_M = 190.0  # 380 m box
PIXEL_M = 20.0
PASS_CELL_HIT_RATE = 0.40
PASS_MAX_CONTROL_HITS = 1
CLEAR_SCL = (4, 5, 6, 7)

SAMPLE_PATH = OFFSHORE_DIR / "validation_sample.json"
SWIR_PATH = OFFSHORE_DIR / "validation_swir.json"

TOKEN_URL = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"
PROCESS_URL = "https://sh.dataspace.copernicus.eu/api/v1/process"
CATALOG_URL = "https://sh.dataspace.copernicus.eu/api/v1/catalog/1.0.0/search"
EVALSCRIPT = """//VERSION=3
function setup() { return {input: [{bands: ["B12", "SCL", "dataMask"]}], output: {bands: 3, sampleType: "FLOAT32"}}; }
function evaluatePixel(s) { return [s.B12, s.SCL, s.dataMask]; }
"""


# --- sampling -----------------------------------------------------------------------------

def load_cells(history_path: Path = OFFSHORE_HISTORY_PATH) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(all detections with a cell key, one row per cell)."""
    rows = pd.read_parquet(history_path)
    grid_lat, grid_lon = grid_keys(rows)
    rows["cell"] = grid_lat.astype(str) + "_" + grid_lon.astype(str)
    cells = rows.groupby("cell").agg(
        lat=("latitude", "mean"), lon=("longitude", "mean"), zone=("offshore_zone", "first"),
        active_days=("acq_date", "nunique"), detections=("acq_date", "size"),
        dist_offshore_km=("dist_offshore_km", "mean"), dates=("acq_date", lambda s: sorted(set(s))),
        night_share=("daynight", lambda s: float((s == "N").mean())), frp_median=("frp", "median"),
    ).reset_index()
    return rows, cells


def _haversine_km(lat1, lon1, lat2, lon2):
    p = math.pi / 180
    a = np.sin((lat2 - lat1) * p / 2) ** 2 + np.cos(lat1 * p) * np.cos(lat2 * p) * np.sin((lon2 - lon1) * p / 2) ** 2
    return 12742 * np.arcsin(np.sqrt(a))


def draw_sample(history_path: Path = OFFSHORE_HISTORY_PATH) -> dict:
    rows, cells = load_cells(history_path)
    rng = np.random.default_rng(SEED)
    recurring = cells[cells["active_days"] >= OFFSHORE_MIN_RECURRENCE].sort_values("cell").reset_index(drop=True)
    chosen = recurring.iloc[np.sort(rng.choice(len(recurring), size=min(N_CELLS, len(recurring)), replace=False))]

    names = sorted(OFFSHORE_ZONES)
    areas = np.array([(OFFSHORE_ZONES[n][2] - OFFSHORE_ZONES[n][0]) * (OFFSHORE_ZONES[n][3] - OFFSHORE_ZONES[n][1]) for n in names])
    det_lat, det_lon = rows["latitude"].to_numpy(), rows["longitude"].to_numpy()
    pool_dates = {z: sorted(rows.loc[rows["offshore_zone"] == z, "acq_date"].unique()) for z in names}
    controls: list[dict] = []
    tries = 0
    while len(controls) < CONTROL_POOL and tries < 50_000:
        tries += 1
        zone = names[int(rng.choice(len(names), p=areas / areas.sum()))]
        west, south, east, north = OFFSHORE_ZONES[zone]
        lon, lat = float(rng.uniform(west, east)), float(rng.uniform(south, north))
        gap = float(_haversine_km(lat, lon, det_lat, det_lon).min())
        if gap < CONTROL_MIN_GAP_KM:
            continue
        probe = pd.DataFrame({"longitude": [lon], "latitude": [lat]})
        if offshore_columns(probe)["offshore_zone"].isna().iat[0]:  # not >= 10 km beyond the boundary
            continue
        dates = sorted(rng.choice(pool_dates[zone], size=ANCHORS, replace=False).tolist())
        controls.append({"id": f"C{len(controls) + 1:02d}", "zone": zone, "lat": round(lat, 5), "lon": round(lon, 5),
                         "nearest_detection_km": round(gap, 1), "anchor_dates": dates})
    return {
        "seed": SEED, "n_recurring_cells": int(len(recurring)),
        "cells": [
            {"id": f"O{i + 1:02d}", "cell": r.cell, "zone": r.zone, "lat": round(float(r.lat), 5), "lon": round(float(r.lon), 5),
             "active_days": int(r.active_days), "detections": int(r.detections),
             "dist_offshore_km": round(float(r.dist_offshore_km), 1),
             "anchor_dates": sorted(rng.choice(r.dates, size=min(ANCHORS, len(r.dates)), replace=False).tolist())}
            for i, r in enumerate(chosen.itertuples())
        ],
        "controls": controls,
    }


# --- Sentinel-2 ---------------------------------------------------------------------------

def _credentials() -> tuple[str, str]:
    cid, secret = _env("SH_CLIENT_ID"), _env("SH_CLIENT_SECRET")
    if not cid or not secret:
        sys.exit("Set SH_CLIENT_ID and SH_CLIENT_SECRET in .env")
    return cid, secret


class Sentinel:
    def __init__(self):
        self.session = requests.Session()
        self._token, self._issued = None, 0.0

    def _headers(self) -> dict:
        if self._token is None or time.time() - self._issued > 3000:
            cid, secret = _credentials()
            response = self.session.post(TOKEN_URL, data={"grant_type": "client_credentials", "client_id": cid,
                                                          "client_secret": secret}, timeout=60)
            response.raise_for_status()
            self._token, self._issued = response.json()["access_token"], time.time()
        return {"Authorization": f"Bearer {self._token}"}

    def scenes_near(self, lat: float, lon: float, anchor: str) -> list[str]:
        """Dates of L2A scenes over the point within +-SCENE_WINDOW_DAYS of anchor, nearest first."""
        day = datetime.strptime(anchor, "%Y-%m-%d")
        body = {"collections": ["sentinel-2-l2a"], "limit": 30,
                "bbox": [lon - 0.002, lat - 0.002, lon + 0.002, lat + 0.002],
                "datetime": f"{(day - timedelta(days=SCENE_WINDOW_DAYS)):%Y-%m-%d}T00:00:00Z/"
                            f"{(day + timedelta(days=SCENE_WINDOW_DAYS)):%Y-%m-%d}T23:59:59Z"}
        for attempt in range(4):
            response = self.session.post(CATALOG_URL, json=body, headers=self._headers(), timeout=120)
            if response.status_code == 429:
                time.sleep(5 * (attempt + 1))
                continue
            response.raise_for_status()
            break
        dates = {f["properties"]["datetime"][:10] for f in response.json().get("features", [])}
        return sorted(dates, key=lambda d: (abs((datetime.strptime(d, "%Y-%m-%d") - day).days), d))

    def box(self, lat: float, lon: float, date: str) -> np.ndarray | None:
        """B12, SCL, dataMask over the 380 m box at 20 m for one acquisition day, shape (3, 19, 19)."""
        import rasterio
        from rasterio.io import MemoryFile

        dlat = BOX_HALF_M / 111_320.0
        dlon = BOX_HALF_M / (111_320.0 * math.cos(math.radians(lat)))
        size = int(round(2 * BOX_HALF_M / PIXEL_M))
        body = {
            "input": {"bounds": {"bbox": [lon - dlon, lat - dlat, lon + dlon, lat + dlat],
                                 "properties": {"crs": "http://www.opengis.net/def/crs/EPSG/0/4326"}},
                      "data": [{"type": "sentinel-2-l2a", "dataFilter": {
                          "timeRange": {"from": f"{date}T00:00:00Z", "to": f"{date}T23:59:59Z"},
                          "mosaickingOrder": "leastCC"}}]},
            "output": {"width": size, "height": size, "responses": [{"identifier": "default", "format": {"type": "image/tiff"}}]},
            "evalscript": EVALSCRIPT,
        }
        for attempt in range(4):
            response = self.session.post(PROCESS_URL, json=body, headers=self._headers(), timeout=180)
            if response.status_code == 429:
                time.sleep(5 * (attempt + 1))
                continue
            if response.status_code >= 400:
                return None
            with MemoryFile(response.content) as memory, memory.open() as raster:
                return raster.read().astype("float64")
        return None


def analyse_box(cube: np.ndarray) -> dict:
    """Clear fraction and the hot pixel nearest the box centre (onshore method, README)."""
    b12, scl, mask = cube
    clear = (mask > 0) & np.isin(np.rint(scl), CLEAR_SCL)
    clear_fraction = float(clear.mean())
    out = {"clear_frac": round(clear_fraction, 3), "hot_found": False, "hot_dist_m": None, "hot_b12": None,
           "max_b12": None}
    if clear_fraction < MIN_CLEAR_FRACTION:
        return out
    values = b12[clear]
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    hot = clear & (b12 > HOT_B12_MIN) & (b12 > median + HOT_MAD_MULTIPLE * mad)
    out["max_b12"] = round(float(values.max()), 3)
    if hot.any():
        rows, cols = np.nonzero(hot)
        centre = (b12.shape[0] - 1) / 2.0
        distance = PIXEL_M * np.hypot(rows - centre, cols - centre)
        nearest = int(np.argmin(distance))
        out.update(hot_found=True, hot_dist_m=round(float(distance[nearest]), 1), hot_b12=round(float(b12[rows[nearest], cols[nearest]]), 3))
    return out


def check_point(client: Sentinel, point: dict) -> dict:
    """Up to ANCHORS clear scenes (one per anchor date, nearest first) and whether any has a hot pixel within HIT_RADIUS_M."""
    used, scenes = set(), []
    for anchor in point["anchor_dates"]:
        for date in client.scenes_near(point["lat"], point["lon"], anchor):
            if date in used:
                continue
            used.add(date)
            cube = client.box(point["lat"], point["lon"], date)
            if cube is None:
                continue
            result = analyse_box(cube)
            if result["clear_frac"] >= MIN_CLEAR_FRACTION:
                scenes.append({"date": date, "anchor": anchor, **result})
                break
    hit = any(s["hot_found"] and s["hot_dist_m"] <= HIT_RADIUS_M for s in scenes)
    return {"id": point["id"], "n_clear_scenes": len(scenes), "evaluable": len(scenes) >= MIN_CLEAR_SCENES,
            "hit": hit, "scenes": scenes}


def run_swir(sample: dict) -> dict:
    client = Sentinel()
    cells = []
    for cell in sample["cells"]:
        result = {**cell, **check_point(client, cell)}
        cells.append(result)
        print(f"  {cell['id']} {cell['zone']:<12} {cell['lat']:.4f},{cell['lon']:.4f}  clear scenes {result['n_clear_scenes']}  "
              f"{'HIT' if result['hit'] else '-'}", flush=True)
    controls, evaluable = [], 0
    for control in sample["controls"]:
        if evaluable >= N_CONTROLS:
            break
        result = {**control, **check_point(client, control)}
        controls.append(result)
        evaluable += int(result["evaluable"])
        print(f"  {control['id']} {control['zone']:<12} {control['lat']:.4f},{control['lon']:.4f}  clear scenes "
              f"{result['n_clear_scenes']}  {'HIT' if result['hit'] else '-'}"
              f"{'' if result['evaluable'] else '  (replaced: < 2 clear scenes)'}", flush=True)
    return {"cells": cells, "controls": controls, "summary": summarise(cells, controls)}


def summarise(cells: list[dict], controls: list[dict]) -> dict:
    evaluable_cells = [c for c in cells if c["evaluable"]]
    used_controls = [c for c in controls if c["evaluable"]][:N_CONTROLS]
    cell_hits = sum(c["hit"] for c in evaluable_cells)
    control_hits = sum(c["hit"] for c in used_controls)
    rate = cell_hits / len(evaluable_cells) if evaluable_cells else 0.0
    return {
        "cells_drawn": len(cells), "cells_evaluable": len(evaluable_cells), "cells_with_hit": int(cell_hits),
        "cell_hit_rate": round(rate, 3), "cell_pass": bool(rate >= PASS_CELL_HIT_RATE and evaluable_cells),
        "controls_evaluated": len(used_controls), "controls_replaced": len(controls) - len(used_controls),
        "controls_with_hit": int(control_hits), "control_pass": bool(control_hits <= PASS_MAX_CONTROL_HITS),
        "passed": bool(evaluable_cells and rate >= PASS_CELL_HIT_RATE and control_hits <= PASS_MAX_CONTROL_HITS
                       and len(used_controls) == N_CONTROLS),
        "rule": f"hot pixel within {HIT_RADIUS_M:g} m in >= {PASS_CELL_HIT_RATE:.0%} of offshore cells with "
                f">= {MIN_CLEAR_SCENES} clear scenes, and in <= {PASS_MAX_CONTROL_HITS} of {N_CONTROLS} controls",
        "by_zone": {z: {"evaluable": sum(1 for c in evaluable_cells if c["zone"] == z),
                        "hits": sum(1 for c in evaluable_cells if c["zone"] == z and c["hit"])}
                    for z in sorted({c["zone"] for c in cells})},
    }


# --- comparison with OSM platforms and GEM fields -----------------------------------------

OVERPASS_URL = "https://overpass-api.de/api/interpreter"


def fetch_osm_platforms() -> gpd.GeoDataFrame:
    """OSM man_made=offshore_platform features (nodes and ways) inside the zone boxes, via Overpass."""
    from shapely.geometry import Point

    records = []
    for zone, (west, south, east, north) in OFFSHORE_ZONES.items():
        query = (f'[out:json][timeout:120];(node["man_made"="offshore_platform"]({south},{west},{north},{east});'
                 f'way["man_made"="offshore_platform"]({south},{west},{north},{east}););out center tags;')
        response = requests.post(OVERPASS_URL, data={"data": query}, timeout=180)
        response.raise_for_status()
        for element in response.json().get("elements", []):
            lat = element.get("lat") or element.get("center", {}).get("lat")
            lon = element.get("lon") or element.get("center", {}).get("lon")
            if lat is None:
                continue
            tags = element.get("tags", {})
            records.append({"osm_id": f"{element['type']}/{element['id']}", "name": tags.get("name"),
                            "operator": tags.get("operator"), "zone": zone, "geometry": Point(lon, lat)})
    return gpd.GeoDataFrame(records, geometry="geometry", crs="EPSG:4326")


def compare_locations(cells: pd.DataFrame, platforms: gpd.GeoDataFrame, gem: pd.DataFrame) -> pd.DataFrame:
    """Distance (km) from each cell to the nearest OSM offshore platform and GEM oil & gas field.
    Descriptive only -- neither source is an input to the rule."""
    pts = gpd.GeoSeries(gpd.points_from_xy(cells["lon"], cells["lat"]), crs="EPSG:4326").to_crs(NATIONAL_PROJECTED_CRS)
    out = cells.copy()
    if len(platforms):
        plat = platforms.to_crs(NATIONAL_PROJECTED_CRS)
        out["osm_platform_km"] = [round(float(plat.geometry.distance(p).min()) / 1000, 2) for p in pts]
        out["osm_platform"] = [plat.iloc[int(plat.geometry.distance(p).values.argmin())]["name"] for p in pts]
    else:
        out["osm_platform_km"], out["osm_platform"] = np.nan, None
    fields = gem[gem["facility_type"] == "oil_gas_field"].dropna(subset=["latitude", "longitude"])
    if len(fields):
        gp = gpd.GeoSeries(gpd.points_from_xy(fields["longitude"], fields["latitude"]), crs="EPSG:4326").to_crs(NATIONAL_PROJECTED_CRS)
        names = fields["facility_name"].to_numpy()
        out["gem_field_km"] = [round(float(gp.distance(p).min()) / 1000, 1) for p in pts]
        out["gem_field"] = [names[int(gp.distance(p).values.argmin())] for p in pts]
    else:
        out["gem_field_km"], out["gem_field"] = np.nan, None
    return out


# --- CLI ----------------------------------------------------------------------------------

def _main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("step", choices=["sample", "swir", "compare"])
    args = parser.parse_args()
    OFFSHORE_DIR.mkdir(parents=True, exist_ok=True)
    if args.step == "sample":
        if SAMPLE_PATH.exists():
            sys.exit(f"{SAMPLE_PATH} exists: the sample is drawn once, before the check, and never redrawn")
        sample = draw_sample()
        SAMPLE_PATH.write_text(json.dumps(sample, indent=1))
        print(f"{len(sample['cells'])} cells of {sample['n_recurring_cells']} recurring; {len(sample['controls'])} control candidates")
    elif args.step == "swir":
        result = run_swir(json.loads(SAMPLE_PATH.read_text()))
        SWIR_PATH.write_text(json.dumps(result, indent=1))
        print(json.dumps(result["summary"], indent=1))
    elif args.step == "compare":
        rows, cells = load_cells()
        sample = json.loads(SAMPLE_PATH.read_text())
        drawn = cells[cells["cell"].isin([c["cell"] for c in sample["cells"]])]
        compared = compare_locations(cells[cells["active_days"] >= OFFSHORE_MIN_RECURRENCE].assign(),
                                     fetch_osm_platforms(), pd.read_csv(GEM_MULTISECTOR_FACILITIES_PATH))
        compared.drop(columns=["dates"]).to_csv(OFFSHORE_DIR / "validation_platform_comparison.csv", index=False)
        print(compared[["osm_platform_km", "gem_field_km"]].describe().to_string())
        print("cells drawn for SWIR:", len(drawn))


if __name__ == "__main__":
    _main()
