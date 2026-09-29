"""Draw a blind gold sample of detections that are `wildfire` under the current rule
AND would become `agricultural burning` under the (still off-by-default)
CROPLAND_SHARE_FIX_ENABLED rule -- i.e. shrubland/grassland with cropland_share_1km >
CROPLAND_SHARE_THRESHOLD, from training.report_cropland_share_impact's per-detection
output. No label, silver or otherwise, is written to the sample file -- only
coordinates, timing, and dNBR burn-scar evidence (dnbr_check.py's method, applied
directly since is_wildfire already requires >2km from any industrial context, so the
industrial-proximity skip that script applies to a mixed sample is never triggered
here). You still decide verified_label yourself.

Draws at most one per ~375m grid cell (training.spatial_features.grid_keys), excludes
every cell already used in gold_sample.csv, gold_holdout.csv or
gold_brick_kiln_sample.csv, and spreads across states in PRIORITY_STATES first (each
shuffled under a fixed seed for reproducibility), filling any remainder from other
states with qualifying detections.

    python -m training.draw_cropland_share_gold_sample \
        --impact training/data/cropland_share_impact_report.csv \
        --out training/data/gold_cropland_share_sample.csv --n 15
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
import requests

import dnbr_check
from backend.config import MAP_POINTS_META_PATH, ROOT_DIR
from training.report_by_state import assign_state
from training.report_cropland_share_impact import _source_paths
from training.spatial_features import grid_keys

SEED = 20260929  # fixed for reproducibility, chosen once, not retried
PRIORITY_STATES = ["Andhra Pradesh", "Karnataka", "Gujarat", "Rajasthan"]
EXISTING_GOLD_FILES = ["training/data/gold_sample.csv", "training/data/gold_holdout.csv",
                       "training/data/gold_brick_kiln_sample.csv"]
_DETAIL_COLUMNS = ["latitude", "longitude", "acq_date", "acq_time", "satellite", "label", "landcover_class"]


def load_qualifying_with_timing(impact_csv: str | Path, meta_path=MAP_POINTS_META_PATH) -> pd.DataFrame:
    """The would_reclassify rows from the impact report, with acq_date/acq_time/
    satellite attached back on -- the impact report itself only carries the columns
    its national-scale WorldCover pass needed (see report_cropland_share_impact's
    _READ_COLUMNS), not full detection detail. cropland_share_1km is a function of
    location only, so it's valid to re-join it back onto the full-detail rows by
    (latitude, longitude, landcover_class) even though multiple distinct detections
    (different dates) can share it at a recurring site -- each such detection is a
    real, separate, independently drawable event."""
    impact = pd.read_csv(impact_csv)
    qualifying = impact[impact["would_reclassify"]][["latitude", "longitude", "landcover_class", "cropland_share_1km"]]
    if qualifying.empty:
        return qualifying.assign(acq_date=[], acq_time=[], satellite=[])

    frames = []
    for path in _source_paths(meta_path):
        present = set(pq.read_schema(path).names)
        columns = [c for c in _DETAIL_COLUMNS if c in present]
        frame = pd.read_parquet(path, columns=columns).reindex(columns=_DETAIL_COLUMNS)
        frames.append(frame[frame["label"] == "wildfire"])
    rows = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=_DETAIL_COLUMNS)
    merged = rows.merge(qualifying, on=["latitude", "longitude", "landcover_class"], how="inner")
    return merged.drop_duplicates(subset=["latitude", "longitude", "acq_date", "acq_time", "satellite"])


def _excluded_cells(paths: list[str]) -> set[tuple[int, int]]:
    cells = set()
    for path in paths:
        if not Path(path).exists():
            continue
        frame = pd.read_csv(path, usecols=["latitude", "longitude"])
        glat, glon = grid_keys(frame)
        cells |= set(zip(glat.tolist(), glon.tolist()))
    return cells


def _quota(n: int, k: int) -> list[int]:
    """n split as evenly as possible across k slots, remainder to the earliest slots
    (e.g. _quota(15, 4) == [4, 4, 4, 3])."""
    base, remainder = divmod(n, k)
    return [base + 1 if i < remainder else base for i in range(k)]


def draw_sample(candidates: pd.DataFrame, n: int, seed: int = SEED,
                priority_states: list[str] = PRIORITY_STATES,
                exclude_cells: set[tuple[int, int]] | None = None) -> pd.DataFrame:
    """One row per ~375m grid cell. Each priority_states gets an even target share of
    n (e.g. 15 across 4 states -> 4/4/4/3) drawn from a fixed-seed shuffle of its
    qualifying candidates; any state short of its target -- and any of n left over
    once every priority state is either filled or exhausted -- is filled by round-
    robin across every other state with a qualifying candidate, alphabetically, until
    n are drawn or candidates run out everywhere."""
    exclude_cells = exclude_cells or set()
    glat, glon = grid_keys(candidates)
    candidates = candidates.assign(_glat=glat, _glon=glon)
    key = pd.MultiIndex.from_arrays([candidates["_glat"], candidates["_glon"]])
    excluded_index = pd.MultiIndex.from_tuples(sorted(exclude_cells)) if exclude_cells else pd.MultiIndex.from_arrays([[], []])
    candidates = candidates[~key.isin(excluded_index)]
    candidates = candidates.drop_duplicates(subset=["_glat", "_glon"])  # one detection per cell, even within the draw
    state = assign_state(candidates[["latitude", "longitude"]]).fillna("(offshore/border)")
    candidates = candidates.assign(_state=state)

    other_states = sorted(set(candidates["_state"]) - set(priority_states) - {"(offshore/border)"})
    order = priority_states + other_states
    pools = {s: candidates[candidates["_state"] == s].sample(frac=1, random_state=seed) for s in order}
    cursors = {s: 0 for s in order}

    def take(state_name: str, count: int) -> list:
        available = min(count, len(pools[state_name]) - cursors[state_name])
        taken = [pools[state_name].iloc[cursors[state_name] + i] for i in range(available)]
        cursors[state_name] += available
        return taken

    picks = []
    for state_name, target in zip(priority_states, _quota(n, len(priority_states))):
        picks.extend(take(state_name, target))
    while len(picks) < n and any(cursors[s] < len(pools[s]) for s in order):
        for s in order:
            if len(picks) >= n:
                break
            picks.extend(take(s, 1))
    return pd.DataFrame(picks).drop(columns=["_glat", "_glon"]).reset_index(drop=True)


def with_dnbr_evidence(sample: pd.DataFrame, pause: float = dnbr_check.PAUSE_S) -> pd.DataFrame:
    """dnbr_check.py's burn-scar evidence for each row, computed directly (no
    industrial-proximity skip) -- every row here already satisfies is_wildfire's own
    dist_to_industrial_m > 2000m requirement by construction."""
    session, (cid, secret) = requests.Session(), dnbr_check.get_credentials()
    token = dnbr_check.get_token(session, cid, secret)
    issued = time.time()
    rows = []
    for i, row in enumerate(sample.itertuples(), 1):
        if time.time() - issued > 3000:
            token = dnbr_check.get_token(session, cid, secret)
            issued = time.time()
        res = dnbr_check.check_point(session, token, row.latitude, row.longitude, str(row.acq_date)[:10])
        rows.append(res)
        note = f"dNBR {res['dnbr']:+.3f} {res['burn_evidence']}" if res.get("status") == "ok" else res["status"]
        print(f"[{i:>3}/{len(sample)}] {row.sample_id}  {note}", flush=True)
        time.sleep(pause)
    return sample.reset_index(drop=True).join(pd.DataFrame(rows))


def build_sample_file(picked: pd.DataFrame, id_prefix: str = "CS") -> pd.DataFrame:
    picked = picked.reset_index(drop=True)
    picked["sample_id"] = [f"{id_prefix}{i + 1:03d}" for i in range(len(picked))]
    hhmm = picked["acq_time"].astype(str).str.zfill(4)
    utc = pd.to_datetime(picked["acq_date"].astype(str).str[:10]) + pd.to_timedelta(
        hhmm.str[:2].astype(int), unit="h"
    ) + pd.to_timedelta(hhmm.str[2:].astype(int), unit="m")
    ist = utc + pd.Timedelta(hours=5, minutes=30)
    picked["acq_time_ist"] = ist.dt.strftime("%H:%M")
    picked["maps_link"] = [f"https://www.google.com/maps?q={lat},{lon}&t=k" for lat, lon in zip(picked["latitude"], picked["longitude"])]
    picked["verified_label"], picked["structure_seen"], picked["confidence"] = "", "", ""
    # cropland_share_1km and landcover_class are withheld -- cropland_share_1km IS the
    # proposed rule's decision variable, so showing it would bias the blind review
    # (state doesn't: it's already implicit in the maps_link, so it's kept for context).
    return picked[[
        "sample_id", "latitude", "longitude", "acq_date", "acq_time", "acq_time_ist", "maps_link",
        "_state", "verified_label", "structure_seen", "confidence",
    ]].rename(columns={"_state": "state"})


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--impact", default="training/data/cropland_share_impact_report.csv")
    ap.add_argument("--out", default="training/data/gold_cropland_share_sample.csv")
    ap.add_argument("--n", type=int, default=15)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--no-dnbr", action="store_true", help="skip the Sentinel Hub dNBR pass (for a dry run)")
    return ap.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    candidates = load_qualifying_with_timing(args.impact)
    print(f"{len(candidates):,} qualifying detections (wildfire now, agricultural burning if the fix were on)")
    excluded = _excluded_cells([str(ROOT_DIR / p) for p in EXISTING_GOLD_FILES])
    picked = draw_sample(candidates, n=args.n, seed=args.seed, exclude_cells=excluded)
    print(f"Drew {len(picked)} of {args.n} requested, by state:")
    print(picked["_state"].value_counts().to_string())
    sample = build_sample_file(picked)
    if not args.no_dnbr:
        sample = with_dnbr_evidence(sample)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    sample.to_csv(args.out, index=False)
    print(f"\nWritten to {args.out} -- no label of any kind is included; fill in verified_label yourself.")
