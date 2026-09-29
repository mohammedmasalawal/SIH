"""
training/ndvi_reference.py -- an independent, automatic reference for whether a
detection's surroundings behave like cropland or like natural vegetation, built from
a full year of Sentinel-2 NDVI, with NO manual review and NO access to any silver or
verified label while classifying.

Why NDVI seasonality: cropland is farmed in cycles -- planted, grown, harvested (NDVI
crashes toward bare soil), sometimes replanted the same year -- while natural
vegetation (forest, shrubland, grassland) follows one broader seasonal cycle (usually
monsoon-driven in India) without crashing to bare-soil NDVI between cycles. Two
signals capture that: how many distinct green-up peaks a year shows, and how far NDVI
drops in its troughs.

ALL THRESHOLDS BELOW ARE FIXED BEFORE ANY CALIBRATION OR SCORING RUN AND MUST NOT BE
RETUNED AFTER SEEING RESULTS -- that is what makes this an honest test of the
cropland_share_1km rule rather than a curve fit to it. If calibration
(--calibrate, see below) comes in under CALIBRATION_MIN_AGREEMENT, the right response
is to stop and report that, not to adjust a number here and rerun.

Method, fixed in advance:
  1. Per point, fetch NDVI = (B08 - B04) / (B08 + B04) from Sentinel Hub's Statistical
     API, Sentinel-2 L2A, over a WINDOW_DAYS-day window ending at the detection's
     acq_date, as ten-day (P10D) composites -- N_DEKADS = WINDOW_DAYS // DEKAD_DAYS of
     them. A BUFFER_M-half-width box (matching the "100 m box" spec) is sampled;
     pixels are kept only where the scene classification layer (SCL) marks them
     vegetation/bare soil/water/unclassified (codes 4/5/6/7) -- clouds, shadow, snow
     and no-data are dropped, same convention dnbr_check.py uses. A dekad is discarded
     if fewer than 1-MAX_CLOUD_FRAC of its sampled pixels pass that mask.
  2. The kept dekads are placed on a fixed 0..N_DEKADS-1 grid by their date. If fewer
     than MIN_VALID_FRACTION of the grid has a kept dekad, the point is
     "insufficient_data" and contributes to no score.
  3. Interior gaps (a kept dekad both before and after) are linearly interpolated so
     peak-finding sees a continuous series; leading/trailing gaps are never
     extrapolated and are excluded from every statistic below.
  4. scipy.signal.find_peaks locates green-up peaks on that series, requiring
     prominence >= PEAK_PROMINENCE and at least PEAK_MIN_DISTANCE_DEKADS between
     peaks (so a single noisy plateau isn't double-counted).
  5. Classification:
       - 2+ peaks in the year -> "crop_like" (a second cropping cycle).
       - exactly 1 peak, whose amplitude (max-min of the observed span) is >=
         AMPLITUDE_CROP_MIN and whose trough is <= TROUGH_BARE_SOIL_MAX (i.e. the
         low season crashes to near-bare-soil, not just to dormant/dry natural
         vegetation) -> "crop_like".
       - everything else -> "natural_like" (one broad seasonal peak without a
         bare-soil crash, a flat/low profile, or a consistently high, non-seasonal
         profile such as dense forest).

This produces "crop_like" / "natural_like" / "insufficient_data", never a fire-type
label, and never "accuracy" against it -- always "agreement with an independent
satellite reference" (this reference's own accuracy is itself only checked, not
proven, by the --calibrate step below).

Setup: same OAuth client as dnbr_check.py -- SH_CLIENT_ID / SH_CLIENT_SECRET in .env.

Usage:
    python -m training.ndvi_reference --in training/data/some_sample.csv \
                                      --out training/data/ndvi_reference_results.csv
    python -m training.ndvi_reference --in ... --out ... --limit 5   # try a few first

    python -m training.ndvi_reference --calibrate \
        --gold training/data/gold_verified_agri_wildfire.csv \
        --sample training/data/gold_sample.csv \
        --out training/data/ndvi_calibration_results.csv
        # scores this reference against the hand-verified agricultural burning /
        # wildfire rows, prints a confusion table, and stops (never proceeds to a
        # 300-row run) if agreement < CALIBRATION_MIN_AGREEMENT.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from scipy.signal import find_peaks

TOKEN_URL = ("https://identity.dataspace.copernicus.eu/auth/realms/CDSE"
             "/protocol/openid-connect/token")
STATS_URL = "https://sh.dataspace.copernicus.eu/api/v1/statistics"

# --- fetch parameters (fixed before any run) -----------------------------------------
DEKAD_DAYS = 10
WINDOW_DAYS = 360          # ~12 months = 36 ten-day dekads, ending at acq_date
N_DEKADS = WINDOW_DAYS // DEKAD_DAYS
BUFFER_M = 50              # half-width -> a 100m box, as specified
MAX_CLOUD_FRAC = 0.3       # a dekad composite is dropped if >30% of sampled pixels are cloud/shadow/snow/nodata
PAUSE_S = 0.5

# --- classification thresholds (fixed before any run -- see module docstring) --------
MIN_VALID_FRACTION = 0.35
PEAK_PROMINENCE = 0.15
PEAK_MIN_DISTANCE_DEKADS = 4     # >= 40 days between counted peaks
AMPLITUDE_CROP_MIN = 0.35
TROUGH_BARE_SOIL_MAX = 0.25

CALIBRATION_MIN_AGREEMENT = 0.80  # stop and report, never proceed, if below this
ADOPTION_MARGIN_POINTS = 10.0     # proposed must beat current by this many pp to be adopted

# NDVI, plus a validity mask from the scene classification layer (SCL). Same kept/
# dropped SCL codes as dnbr_check.py: keep 4 vegetation, 5 bare soil, 6 water, 7
# unclassified; drop 3 cloud shadow, 8/9/10 cloud/cirrus, 11 snow, 0/1/2 no-data.
EVALSCRIPT = """
//VERSION=3
function setup() {
  return {
    input: [{bands: ["B08", "B04", "SCL", "dataMask"]}],
    output: [
      {id: "ndvi", bands: 1, sampleType: "FLOAT32"},
      {id: "dataMask", bands: 1}
    ]
  };
}
function evaluatePixel(s) {
  var good = [4, 5, 6, 7].indexOf(s.SCL) >= 0 ? 1 : 0;
  var denom = s.B08 + s.B04;
  var ndvi = denom === 0 ? 0 : (s.B08 - s.B04) / denom;
  return {ndvi: [ndvi], dataMask: [s.dataMask * good]};
}
"""


# --- auth / request plumbing (same conventions as dnbr_check.py) ---------------------

def get_credentials() -> tuple[str, str]:
    cid = os.environ.get("SH_CLIENT_ID", "").strip()
    sec = os.environ.get("SH_CLIENT_SECRET", "").strip()
    env = Path(".env")
    if (not cid or not sec) and env.exists():
        for line in env.read_text().splitlines():
            if "=" not in line or line.strip().startswith("#"):
                continue
            k, v = line.split("=", 1)
            v = v.strip().strip('"').strip("'")
            if k.strip() == "SH_CLIENT_ID" and not cid:
                cid = v
            elif k.strip() == "SH_CLIENT_SECRET" and not sec:
                sec = v
    if not cid or not sec:
        sys.exit("Set SH_CLIENT_ID and SH_CLIENT_SECRET in .env "
                 "(create an OAuth client in the Sentinel Hub dashboard).")
    return cid, sec


def get_token(session: requests.Session, cid: str, secret: str) -> str:
    r = session.post(TOKEN_URL, data={"grant_type": "client_credentials",
                                      "client_id": cid, "client_secret": secret},
                     timeout=60)
    if r.status_code == 401:
        sys.exit("Sentinel Hub rejected the credentials. Check SH_CLIENT_ID / SH_CLIENT_SECRET.")
    r.raise_for_status()
    return r.json()["access_token"]


def bbox_around(lat: float, lon: float, metres: int = BUFFER_M) -> list[float]:
    dlat = metres / 111_320.0
    dlon = metres / (111_320.0 * max(math.cos(math.radians(lat)), 1e-6))
    return [lon - dlon, lat - dlat, lon + dlon, lat + dlat]


def build_request(lat: float, lon: float, start: datetime, end: datetime) -> dict:
    return {
        "input": {
            "bounds": {"bbox": bbox_around(lat, lon),
                       "properties": {"crs": "http://www.opengis.net/def/crs/EPSG/0/4326"}},
            "data": [{
                "type": "sentinel-2-l2a",
                "dataFilter": {"mosaickingOrder": "leastCC"},
            }],
        },
        "aggregation": {
            "timeRange": {"from": start.strftime("%Y-%m-%dT00:00:00Z"),
                          "to": end.strftime("%Y-%m-%dT23:59:59Z")},
            "aggregationInterval": {"of": f"P{DEKAD_DAYS}D"},
            "resx": 0.0001, "resy": 0.0001,   # ~10 m, native Sentinel-2
            "evalscript": EVALSCRIPT,
        },
        "calculations": {"ndvi": {"statistics": {"default": {}}}},
    }


def parse_dekads(payload: dict) -> list[dict]:
    """One record per kept ten-day composite: {date, ndvi, frac_valid}."""
    out = []
    for item in payload.get("data", []):
        interval = item.get("interval", {})
        stats = item.get("outputs", {}).get("ndvi", {}).get("bands", {}).get("B0", {}).get("stats")
        if not stats or stats.get("sampleCount", 0) == 0:
            continue
        valid = stats.get("sampleCount", 0) - stats.get("noDataCount", 0)
        frac_valid = valid / stats["sampleCount"]
        if frac_valid < (1 - MAX_CLOUD_FRAC):
            continue
        out.append({"date": interval.get("from", "")[:10], "ndvi": stats.get("mean"), "frac_valid": round(frac_valid, 2)})
    return sorted(out, key=lambda d: d["date"])


# --- classification: pure, offline, no network (unit-testable on synthetic series) ---

def classify_series(dekads: list[dict], window_start: datetime) -> dict:
    """dekads: parse_dekads' output. window_start: the fetch window's start date (the
    grid's dekad 0). Returns classification (crop_like / natural_like /
    insufficient_data) plus the diagnostics it was computed from."""
    grid = np.full(N_DEKADS, np.nan)
    for d in dekads:
        date = datetime.strptime(d["date"], "%Y-%m-%d")
        i = (date - window_start).days // DEKAD_DAYS
        if 0 <= i < N_DEKADS:
            grid[i] = d["ndvi"] if np.isnan(grid[i]) else (grid[i] + d["ndvi"]) / 2

    valid_fraction = float(np.count_nonzero(~np.isnan(grid)) / N_DEKADS)
    if valid_fraction < MIN_VALID_FRACTION:
        return {"classification": "insufficient_data", "valid_fraction": round(valid_fraction, 2),
                "n_peaks": None, "amplitude": None, "trough": None}

    series = pd.Series(grid)
    filled = series.interpolate(limit_area="inside")  # never extrapolate leading/trailing gaps
    observed = filled.dropna()
    first_i, last_i = int(observed.index.min()), int(observed.index.max())

    amplitude = float(observed.max() - observed.min())
    trough = float(observed.min())
    peak_positions, _ = find_peaks(filled.to_numpy(), prominence=PEAK_PROMINENCE, distance=PEAK_MIN_DISTANCE_DEKADS)
    n_peaks = int(np.sum((peak_positions >= first_i) & (peak_positions <= last_i)))

    if n_peaks >= 2:
        classification = "crop_like"
    elif n_peaks == 1 and amplitude >= AMPLITUDE_CROP_MIN and trough <= TROUGH_BARE_SOIL_MAX:
        classification = "crop_like"
    else:
        classification = "natural_like"
    return {"classification": classification, "valid_fraction": round(valid_fraction, 2),
            "n_peaks": n_peaks, "amplitude": round(amplitude, 3), "trough": round(trough, 3)}


def check_point(session: requests.Session, token: str, lat: float, lon: float, acq_date: str) -> dict:
    end = datetime.strptime(acq_date, "%Y-%m-%d")
    start = end - timedelta(days=WINDOW_DAYS)
    req = build_request(lat, lon, start, end)
    r = session.post(STATS_URL, json=req, headers={"Authorization": f"Bearer {token}"}, timeout=180)
    if r.status_code == 429:
        return {"status": "rate limited, retry later"}
    if r.status_code >= 400:
        return {"status": f"HTTP {r.status_code}: {r.text[:150]}"}
    dekads = parse_dekads(r.json())
    result = classify_series(dekads, start)
    result["status"] = "ok"
    result["n_dekads_kept"] = len(dekads)
    return result


# --- driving the classifier over a CSV of points --------------------------------------

def run_reference(df: pd.DataFrame, pause: float = PAUSE_S, limit: int | None = None) -> pd.DataFrame:
    """df needs sample_id, latitude, longitude, acq_date. One Statistical API call per row."""
    if limit:
        df = df.head(limit)
    session = requests.Session()
    cid, secret = get_credentials()
    token = get_token(session, cid, secret)
    issued = time.time()

    rows = []
    for i, row in enumerate(df.itertuples(), 1):
        if time.time() - issued > 3000:  # tokens last an hour; refresh early
            token = get_token(session, cid, secret)
            issued = time.time()
        res = check_point(session, token, row.latitude, row.longitude, str(row.acq_date))
        rows.append({"sample_id": row.sample_id, "latitude": row.latitude, "longitude": row.longitude,
                    "acq_date": row.acq_date, **res})
        note = res["classification"] if res.get("status") == "ok" else res["status"]
        print(f"[{i:>4}/{len(df)}] {row.sample_id}  {note}", flush=True)
        time.sleep(pause)
    return pd.DataFrame(rows)


# --- calibration against gold_verified_agri_wildfire.csv ------------------------------

_LABEL_TO_CLASS = {"agricultural burning": "crop_like", "wildfire": "natural_like"}


def calibration_rows(gold_path: str | Path, sample_path: str | Path) -> pd.DataFrame:
    """The decided (agricultural burning / wildfire) rows of gold_path joined onto
    sample_path for coordinates, with the reference's expected class attached."""
    gold = pd.read_csv(gold_path)
    gold = gold[gold["verified_label"].isin(_LABEL_TO_CLASS)]
    sample = pd.read_csv(sample_path, usecols=["sample_id", "latitude", "longitude", "acq_date"])
    joined = gold.merge(sample, on="sample_id", how="left")
    missing = joined[joined["latitude"].isna()]
    if not missing.empty:
        sys.exit(f"{len(missing)} gold rows have no matching sample_id in {sample_path}: "
                 f"{missing['sample_id'].tolist()}")
    joined["expected_class"] = joined["verified_label"].map(_LABEL_TO_CLASS)
    return joined[["sample_id", "latitude", "longitude", "acq_date", "verified_label", "expected_class"]]


def score_agreement(rule_class: pd.Series, reference_class: pd.Series) -> dict:
    """% agreement excluding insufficient_data rows, plus n scored / n total."""
    scoreable = reference_class != "insufficient_data"
    n_total, n_scored = len(reference_class), int(scoreable.sum())
    if n_scored == 0:
        return {"agreement_pct": None, "n_scored": 0, "n_total": n_total}
    agree = (rule_class[scoreable] == reference_class[scoreable]).sum()
    return {"agreement_pct": round(agree / n_scored * 100, 1), "n_scored": n_scored, "n_total": n_total}


def confusion_table(expected: pd.Series, actual: pd.Series) -> pd.DataFrame:
    return pd.crosstab(expected, actual, rownames=["hand-verified"], colnames=["ndvi_reference"], dropna=False)


def run_calibration(gold_path: str | Path, sample_path: str | Path, out_path: str | Path,
                    pause: float = PAUSE_S, limit: int | None = None) -> tuple[pd.DataFrame, dict]:
    rows = calibration_rows(gold_path, sample_path)
    print(f"Calibrating on {len(rows)} hand-verified agricultural burning / wildfire rows "
         f"from {gold_path}\n(this reference sees only sample_id/lat/lon/acq_date -- never verified_label)\n")
    result = run_reference(rows[["sample_id", "latitude", "longitude", "acq_date"]], pause=pause, limit=limit)
    merged = rows.merge(result, on=["sample_id", "latitude", "longitude", "acq_date"])
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out_path, index=False)

    score = score_agreement(merged["expected_class"], merged["classification"])
    table = confusion_table(merged["verified_label"], merged["classification"])
    n_insufficient = score["n_total"] - score["n_scored"]
    insufficient_note = f", {n_insufficient} insufficient_data" if n_insufficient else ""
    print(f"\nAgreement with hand-verified labels (an independent satellite reference "
         f"checked against ground truth, not a rule's precision): "
         f"{score['agreement_pct']}% ({score['n_scored']}/{score['n_total']} scored{insufficient_note})")
    print("\nConfusion table:")
    print(table.to_string())
    print(f"\nResults written to {out_path}")
    if score["agreement_pct"] is None or score["agreement_pct"] < CALIBRATION_MIN_AGREEMENT * 100:
        print(f"\nCalibration agreement is below {CALIBRATION_MIN_AGREEMENT * 100:.0f}% -- STOPPING. "
             "Per the fixed-thresholds rule, these cut-offs are not retuned after seeing this result. "
             "Do not proceed to the 300-row scoring run.")
    else:
        print(f"\nCalibration agreement meets the {CALIBRATION_MIN_AGREEMENT * 100:.0f}% bar -- "
             "safe to proceed to the 300-row scoring run.")
    return merged, score


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="infile", help="CSV with sample_id, latitude, longitude, acq_date")
    ap.add_argument("--out", dest="outfile", required=True)
    ap.add_argument("--limit", type=int, help="only process the first N rows (for testing)")
    ap.add_argument("--pause", type=float, default=PAUSE_S, help="seconds to sleep between API calls (raise this on 429s)")
    ap.add_argument("--calibrate", action="store_true", help="run calibration against gold_verified_agri_wildfire.csv instead")
    ap.add_argument("--gold", default="training/data/gold_verified_agri_wildfire.csv")
    ap.add_argument("--sample", default="training/data/gold_sample.csv")
    return ap.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    if args.calibrate:
        run_calibration(args.gold, args.sample, args.outfile, pause=args.pause, limit=args.limit)
    else:
        if not args.infile:
            sys.exit("--in is required unless --calibrate is given")
        df = pd.read_csv(args.infile)
        for col in ("sample_id", "latitude", "longitude", "acq_date"):
            if col not in df.columns:
                sys.exit(f"{args.infile} has no '{col}' column")
        out = run_reference(df, pause=args.pause, limit=args.limit)
        Path(args.outfile).parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(args.outfile, index=False)
        ok = out[out["status"] == "ok"]
        print(f"\n{len(ok)}/{len(out)} points resolved -> {args.outfile}")
        if not ok.empty:
            print(ok["classification"].value_counts().to_string())
