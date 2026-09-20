"""
dnbr_check.py - burn-scar evidence for gold-set verification (AgniNetra, SIH 2026).

For each detection in gold_sample.csv, asks Sentinel Hub's Statistical API for the
Normalized Burn Ratio (NBR) at that point before and after the detection date, and
reports the difference (dNBR). A real vegetation burn drops NBR sharply, so dNBR
comes out positive.

    NBR   = (B08 - B12) / (B08 + B12)
    dNBR  = NBR_before - NBR_after

This produces EVIDENCE, not labels. You still decide verified_label yourself --
that is the whole point of a gold set. dNBR only tells you whether the ground
changed the way a fire would change it.

Interpretation (Key & Benson severity bands, widely used for dNBR):
    < 0.10   no burn detected
    0.10-0.27  low severity  -> consistent with a light crop-residue burn
    0.27-0.66  moderate
    > 0.66     high severity

Caveats worth remembering when you read the output:
  - A harvest also drops NBR. If the "before" scene already shows bare soil, a
    positive dNBR may just be normal field work. Check nbr_before: healthy crop
    is roughly > 0.3, bare soil is nearer 0.
  - Small fires (under ~1 ha) may leave no trace at 10 m resolution.
  - Gas flares and industrial heat produce NO scar at all. dNBR near zero at a
    refinery is expected and says nothing against the detection.

`dnbr_informative` (added on top of what dNBR itself computes): False for any
point inside an OSM industrial polygon or close to a flare-capable/heat-industry
GEM facility (same osm_industrial_tag / distance columns and thresholds rules.py
uses for the gas-flare/industrial rules) -- those points are skipped entirely, no
Statistical API call is made for them, since a refinery has no vegetation to leave
a scar regardless of what dNBR comes back as. This does not read or write the
silver label -- only osm_industrial_tag and the two GEM distance columns.

Setup:
    pip install requests pandas
    Create an OAuth client at https://shapps.dataspace.copernicus.eu/dashboard/
    (User Settings -> OAuth clients -> Create). Then add to .env:
        SH_CLIENT_ID=...
        SH_CLIENT_SECRET=...

Usage:
    python dnbr_check.py --in training/data/gold_sample.csv \
                         --out training/data/dnbr_results.csv
    python dnbr_check.py --in ... --out ... --limit 5     # try a few first
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import requests

TOKEN_URL = ("https://identity.dataspace.copernicus.eu/auth/realms/CDSE"
             "/protocol/openid-connect/token")
STATS_URL = "https://sh.dataspace.copernicus.eu/api/v1/statistics"

WINDOW_DAYS = 30      # how far either side of the detection to look for scenes
BUFFER_M = 190        # half-width of the box sampled (-> ~380m box, matching the ~375m VIIRS footprint)
MAX_CLOUD_FRAC = 0.2  # skip a scene if >20% of sampled pixels are cloud/shadow
PAUSE_S = 0.5

# Same industrial-proximity thresholds rules.py uses for is_gas_flare /
# is_industrial -- a point this close to flare/heat infrastructure, or inside an
# OSM industrial polygon at all, has no vegetation to leave a burn scar.
FLARE_PROXIMITY_M = 1000
HEAT_INDUSTRY_PROXIMITY_M = 2000

# NBR, plus a validity mask from the scene classification layer (SCL).
# SCL codes kept: 4 vegetation, 5 bare soil, 6 water, 7 unclassified.
# Dropped: 3 cloud shadow, 8/9/10 cloud and cirrus, 11 snow, 0/1/2 no-data.
EVALSCRIPT = """
//VERSION=3
function setup() {
  return {
    input: [{bands: ["B08", "B12", "SCL", "dataMask"]}],
    output: [
      {id: "nbr", bands: 1, sampleType: "FLOAT32"},
      {id: "dataMask", bands: 1}
    ]
  };
}
function evaluatePixel(s) {
  var good = [4, 5, 6, 7].indexOf(s.SCL) >= 0 ? 1 : 0;
  var denom = s.B08 + s.B12;
  var nbr = denom === 0 ? 0 : (s.B08 - s.B12) / denom;
  return {nbr: [nbr], dataMask: [s.dataMask * good]};
}
"""


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
    """Small square in WGS84 degrees around a point."""
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
            "aggregationInterval": {"of": "P1D"},   # one entry per acquisition day
            "resx": 0.0001, "resy": 0.0001,          # ~10 m, native Sentinel-2
            "evalscript": EVALSCRIPT,
        },
        "calculations": {"nbr": {"statistics": {"default": {
            "percentiles": {"k": [10, 50], "smoothedPercentiles": True},
        }}}},
    }


def parse_intervals(payload: dict) -> list[dict]:
    """Flatten the Statistical API response into one record per usable day."""
    out = []
    for item in payload.get("data", []):
        interval = item.get("interval", {})
        stats = item.get("outputs", {}).get("nbr", {}).get("bands", {}).get("B0", {}).get("stats")
        if not stats or stats.get("sampleCount", 0) == 0:
            continue
        valid = stats.get("sampleCount", 0) - stats.get("noDataCount", 0)
        frac_valid = valid / stats["sampleCount"]
        if frac_valid < (1 - MAX_CLOUD_FRAC):
            continue  # too much cloud/shadow over the point
        percentiles = stats.get("percentiles", {})
        out.append({"date": interval.get("from", "")[:10],
                    "nbr": stats.get("mean"),
                    "nbr_p10": percentiles.get("10.0"),
                    "nbr_p50": percentiles.get("50.0"),
                    "frac_valid": round(frac_valid, 2)})
    return sorted(out, key=lambda d: d["date"])


def severity(dnbr: float | None) -> str:
    if dnbr is None:
        return ""
    if dnbr < 0.10:
        return "no burn detected"
    if dnbr < 0.27:
        return "low severity"
    if dnbr < 0.66:
        return "moderate severity"
    return "high severity"


def check_point(session: requests.Session, token: str, lat: float, lon: float,
                acq_date: str) -> dict:
    """Nearest clear scene before and after the detection, and their NBR difference."""
    d = datetime.strptime(acq_date, "%Y-%m-%d")
    req = build_request(lat, lon, d - timedelta(days=WINDOW_DAYS),
                        d + timedelta(days=WINDOW_DAYS))
    r = session.post(STATS_URL, json=req,
                     headers={"Authorization": f"Bearer {token}"}, timeout=180)
    if r.status_code == 429:
        return {"status": "rate limited, retry later"}
    if r.status_code >= 400:
        return {"status": f"HTTP {r.status_code}: {r.text[:120]}"}

    days = parse_intervals(r.json())
    before = [x for x in days if x["date"] < acq_date]
    after = [x for x in days if x["date"] >= acq_date]
    if not before or not after:
        missing = "before" if not before else "after"
        return {"status": f"no clear scene {missing} detection",
                "n_clear_scenes": len(days)}

    b, a = before[-1], after[0]
    dnbr = b["nbr"] - a["nbr"]
    # A small, localised scar can sit inside the box without moving its mean much --
    # compare the (stable) before-scene median against the after-scene's low tail,
    # which a burned patch pulls down even when most of the box is untouched.
    dnbr_p10 = None
    if b.get("nbr_p50") is not None and a.get("nbr_p10") is not None:
        dnbr_p10 = round(b["nbr_p50"] - a["nbr_p10"], 4)
    return {
        "status": "ok",
        "nbr_before": round(b["nbr"], 4), "date_before": b["date"],
        "nbr_after": round(a["nbr"], 4), "date_after": a["date"],
        "days_before": (d - datetime.strptime(b["date"], "%Y-%m-%d")).days,
        "days_after": (datetime.strptime(a["date"], "%Y-%m-%d") - d).days,
        "dnbr": round(dnbr, 4), "burn_evidence": severity(dnbr),
        "nbr_before_p50": round(b["nbr_p50"], 4) if b.get("nbr_p50") is not None else None,
        "nbr_after_p10": round(a["nbr_p10"], 4) if a.get("nbr_p10") is not None else None,
        "dnbr_p10": dnbr_p10,
        "n_clear_scenes": len(days),
    }


def load_industrial_context(labeled_csv: str | Path) -> pd.DataFrame:
    """osm_industrial_tag + GEM distance columns, keyed for a join onto gold_sample.

    Reads only these columns -- never `label` -- so the silver label can't reach
    dnbr_results.csv even transiently; verification stays blind.
    """
    return pd.read_csv(labeled_csv, usecols=[
        "latitude", "longitude", "acq_date", "acq_time",
        "osm_industrial_tag", "dist_to_flare_capable_m", "dist_to_heat_industry_m",
    ])


def is_dnbr_informative(row) -> bool:
    """False inside an OSM industrial polygon or near flare/heat-industry infrastructure.

    A refinery or kiln has no vegetation to leave a burn scar, so dNBR there is
    uninformative by construction -- not just usually zero -- and the Statistical
    API call for it is skipped rather than spent.
    """
    osm_tag = row.get("osm_industrial_tag")
    if isinstance(osm_tag, str) and osm_tag.strip():
        return False
    flare_dist = row.get("dist_to_flare_capable_m")
    if pd.notna(flare_dist) and flare_dist <= FLARE_PROXIMITY_M:
        return False
    heat_dist = row.get("dist_to_heat_industry_m")
    if pd.notna(heat_dist) and heat_dist <= HEAT_INDUSTRY_PROXIMITY_M:
        return False
    return True


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--in", dest="infile", default="training/data/gold_sample.csv")
    ap.add_argument("--out", dest="outfile", default="training/data/dnbr_results.csv")
    ap.add_argument("--labeled", dest="labeled_csv", default="training/data/labeled_hotspots_12mo_final.csv",
                    help="labeled hotspots CSV to source osm_industrial_tag/distance columns from")
    ap.add_argument("--limit", type=int, help="only process the first N rows (for testing)")
    ap.add_argument("--pause", type=float, default=PAUSE_S, help="seconds to sleep between API calls (raise this on 429s)")
    a = ap.parse_args()

    df = pd.read_csv(a.infile)
    for col in ("sample_id", "latitude", "longitude", "acq_date"):
        if col not in df.columns:
            sys.exit(f"{a.infile} has no '{col}' column")
    if a.limit:
        df = df.head(a.limit)

    context = load_industrial_context(a.labeled_csv)
    df = df.merge(context, on=["latitude", "longitude", "acq_date", "acq_time"], how="left")
    df["dnbr_informative"] = df.apply(is_dnbr_informative, axis=1)

    session = requests.Session()
    cid, secret = get_credentials()
    token = get_token(session, cid, secret)
    issued = time.time()

    rows = []
    for i, row in enumerate(df.itertuples(), 1):
        if not row.dnbr_informative:
            rows.append({"sample_id": row.sample_id, "latitude": row.latitude,
                        "longitude": row.longitude, "acq_date": row.acq_date,
                        "dnbr_informative": False,
                        "status": "skipped: near industrial infrastructure (osm_industrial_tag/distance)"})
            print(f"[{i:>3}/{len(df)}] {row.sample_id}  skipped (industrial)")
            continue

        if time.time() - issued > 3000:      # tokens last an hour; refresh early
            token = get_token(session, cid, secret)
            issued = time.time()
        res = check_point(session, token, row.latitude, row.longitude, str(row.acq_date))
        rows.append({"sample_id": row.sample_id, "latitude": row.latitude,
                     "longitude": row.longitude, "acq_date": row.acq_date,
                     "dnbr_informative": True, **res})
        note = (f"dNBR {res['dnbr']:+.3f}  {res['burn_evidence']}"
                if res.get("status") == "ok" else res["status"])
        print(f"[{i:>3}/{len(df)}] {row.sample_id}  {note}")
        time.sleep(a.pause)

    out = pd.DataFrame(rows)
    Path(a.outfile).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(a.outfile, index=False)

    ok = out[out["status"] == "ok"]
    skipped = out[~out["dnbr_informative"]]
    print(f"\n{len(ok)}/{len(out)} points resolved, {len(skipped)} skipped (industrial) -> {a.outfile}")
    if not ok.empty:
        print(ok["burn_evidence"].value_counts().to_string())
        print("\nReminder: dNBR is evidence, not a label. A burn scar supports "
              "agricultural burning or wildfire; its absence at a refinery means nothing.")


if __name__ == "__main__":
    main()
