# Agninetra FIRMS weak-label pipeline

Teammate 1 data pipeline for SIH 2026 PS 26162. It fetches NASA FIRMS VIIRS detections, caches OSM/GEM context, joins recurrence and spatial features, and emits the hand-off CSV consumed by training and serving.

## Quick start

```powershell
python -m pip install -r requirements.txt
$env:FIRMS_MAP_KEY = "your-free-firms-map-key"
python firms_client.py --bbox 69.5 21.9 70.5 22.8 --start 2026-01-01 --end 2026-01-31 --output data/raw/jamnagar_viirs.csv
```

The command above merges all three VIIRS satellites (NOAA-20, NOAA-21, SNPP) by default, resolving the correct archive (SP) vs near-real-time (NRT) source per satellite per sub-range from FIRMS' `data_availability` endpoint (NOAA-21 currently has no separate SP source at all). Fetching only one satellite — the old default — undercounts `recurrence_count` by roughly 3x and silently truncates the date range at that one satellite's SP/NRT cutover. Pass `--source VIIRS_NOAA20_NRT` (or similar) to opt back into a single named source for debugging.

Use `context_sources.fetch_osm_industrial` and `load_gem_facilities` to populate cached GeoJSON files, then run:

```powershell
python build_labels.py --firms data/raw/jamnagar_viirs.csv --industrial data/context/osm_industrial.geojson --facilities data/context/gem_facilities.geojson --worldcover data/raw/worldcover/ESA_WorldCover_10m_2021_v200_N21E069_Map.tif --output training/data/labeled_hotspots.csv
```

To (re)build the manual-review gold set from that output:

```powershell
python make_gold_sample.py --labeled training/data/labeled_hotspots.csv
```

`--worldcover` is optional; omit it and `landcover_class` stays null, which silently disables the `agricultural burning` and `wildfire` rules. Pick the tile(s) covering your bbox from `https://esa-worldcover.s3.eu-central-1.amazonaws.com/v200/2021/map/` (3°x3° tiles named by their SW corner, e.g. `N21E069` for Jamnagar) — verify the URL with a HEAD request before downloading, since a missing tile 404s silently different from a bad key.

`unknown` is an intentional output and should be excluded from supervised training until reviewed. Validation belongs in spatial-block and temporal holdouts, never random row splits.

`recurrence_count`, `first_seen`, `last_seen`, and `is_anomalous` are all computed over the full input FIRMS CSV, including whatever date range you pass in. That's correct for labelling, but if that CSV spans your eval period, Teammate 2 must recompute all four from pre-split (training-only) data before using them as model inputs, or they will leak future detections into the label/flag for past rows.

### Gold verification set

`training/data/gold_sample.csv` (150 detections, 30 per class, at most one per ~375m grid cell, 10 drawn from `is_anomalous` cells, built with a fixed seed by `make_gold_sample.py`) is for manual satellite-imagery review — it deliberately withholds the `label` column. The silver label each `sample_id` was drawn with lives separately in `training/data/gold_key.csv`, so a reviewer can't see the rule's answer while checking it.

`gold_sample.csv` columns: `sample_id`, `latitude`, `longitude`, `acq_date`, `acq_time` (UTC), `acq_time_ist` (UTC + 5:30, time-of-day only — near midnight IST this can belong to the following UTC date), `maps_link` (drops a pin at the exact point, satellite view), and three blank columns for the reviewer to fill in:
- `verified_label` — one of `agricultural burning`, `gas flare`, `industrial`, `wildfire`, `unknown`, `uncertain`
- `structure_seen` — free text, what's visible at the pin
- `confidence` — one of `high`, `medium`, `low`

Both `gold_sample.csv` and `gold_key.csv` are tracked in git deliberately, as an exception to `training/data/` being gitignored — hand-verification is manual work that can't be regenerated. **Because of that, do not re-run `make_gold_sample.py` once review has started**: it overwrites `gold_sample.csv` from scratch, blanking any `verified_label`/`structure_seen`/`confidence` already filled in. The seed only guarantees the same 150 detections come back, not that in-progress review survives.

`dnbr_check.py` supplements manual review with satellite evidence: it queries Sentinel Hub's Statistical API for the pre/post-detection change in NBR (burn ratio) at each point and writes `training/data/dnbr_results.csv` (also tracked in git deliberately, same reasoning as above — a paid API run isn't free to redo). It needs `SH_CLIENT_ID`/`SH_CLIENT_SECRET` in `.env` (gitignored, never commit it). It skips the API call entirely — marking `dnbr_informative = False` — for any point inside an OSM industrial polygon or near flare/heat-industry infrastructure, since a refinery or kiln has no vegetation to leave a burn scar; a burn scar supports `agricultural burning`/`wildfire`, but its absence at an industrial site says nothing. `dnbr_results.csv` never contains `label` or `verified_label` — dNBR is evidence for the reviewer, not a second label source, and joining the answer in would defeat blind verification.

Teammate 2 must exclude every gold-set cell, and the whole spatial block each one falls in, from both the training split and the recurrence/anomaly recomputation above — a gold cell inside a training block would let manually-verified ground truth leak into the model's own training signal, defeating the point of holding it out.

`daynight` is never used by any labelling rule, deliberately — it's kept as an independent sanity check (real agricultural burning should skew daytime, flares and industrial heat should skew nighttime; that's how this redesign was validated, not how it was built).

### Contract column changes since the original handoff schema

Removed: **`dist_to_facility_m`, `facility_type`** (single globally-nearest GEM facility, any type — replaced because it picked the wrong facility for large multi-point complexes, e.g. a detection inside the Reliance Jamnagar refinery resolving to a neighbouring cement plant's point instead, since GEM represents each facility as one lat/lon regardless of site size).

Added:
- `dist_to_flare_capable_m` / `nearest_flare_facility_type` — distance to and type of the nearest facility among `oil_gas_power_plant`, `oil_gas_field`, `lng_terminal`, `chemical_plant` (drives `gas flare`).
- `dist_to_heat_industry_m` / `nearest_heat_facility_type` — distance to and type of the nearest facility among `cement_plant`, `steel_plant`, `coal_power_plant`, `coal_mine`, `chemical_plant` (drives `industrial`).
- `is_anomalous` (bool) — see below.
- `osm_industrial_tag` — lowercased `industrial=*` tag of the OSM polygon a detection falls inside (`refinery`, `yes` for untagged `landuse=industrial`, etc.), empty if not inside one. Drives the `industrial`/`gas flare` polygon-containment paths — see above.

Both distance columns are still point-to-point nearest-neighbour, so the same wrong-facility failure mode can recur for any sprawling site not well-represented by a single GEM point — treat them as "nearest tagged facility of this type," not "distance to the site boundary." 334 of the 365 hotspots inside the Reliance refinery polygon but outside any curated GEM point's 2km radius are a known, currently-unresolved instance of this.

**Label value renamed:** `industrial fire` → `industrial` (same column, `label`).

### `industrial` class membership (no FRP gate)

A hotspot is `industrial` if any of:
1. within 2000m of a heat-industry facility, not also within 1000m of a flare-capable facility, with `recurrence_count >= 2`, **or**
2. `recurrence_count == 1` and either inside an OSM industrial polygon or within 500m of a heat-industry facility, **or**
3. `recurrence_count >= 2` and inside an OSM industrial polygon that isn't tagged `industrial=refinery` (see `osm_industrial_tag` below), provided the row doesn't already qualify as `gas flare`.

There is no absolute FRP threshold in this rule — a small, steady heat source (e.g. a cement kiln) and a large furnace are both legitimate industrial signatures, and gating on FRP magnitude structurally excluded the small ones. `gas flare` requires either `recurrence_count >= 5` within 1000m of a flare-capable GEM facility, **or** `recurrence_count >= 2` inside an OSM polygon tagged `industrial=refinery`. The two classes cannot both fire on the same row, by construction: industrial's path 1 explicitly excludes flare-blocked rows, path 2's `recurrence_count == 1` can never satisfy either gas-flare path's `>= 2`/`>= 5`, and path 3 explicitly defers to gas flare when it already matches.

`osm_industrial_tag` (new column) is the lowercased OSM `industrial=*` tag of the polygon a detection falls inside, or empty if it isn't inside any OSM industrial polygon. Plain `landuse=industrial` polygons with no subtype tag get `yes` — present but not a refinery, which is all the rules above need.

### `is_anomalous`

Per ~375m grid cell, per day/night (VIIRS reads FRP differently under solar illumination, so baselines are kept separate): compare a detection's FRP to that same cell's own **prior** history only — never same-day or future detections. Needs at least 5 prior active days at that cell; before that, always `False` (not "normal", just not yet judged). Baseline is the site's daily-max FRP (median + MAD, robust to outliers); flagged when the modified z-score `0.6745 * (frp - median) / MAD > 3.5`, or `frp > 3 * median` when MAD is 0. One-sided — a drop in FRP, or a missing detection, is never flagged.
