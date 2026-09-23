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

### `gold_sample.csv` is no longer a clean held-out set — use `gold_holdout.csv` for evaluation

The first batch of `gold_sample.csv` review (`training/data/gold_verified1.csv`, 69 rows, all `industrial`-adjacent) was used *diagnostically*: comparing verified labels against the rule engine's silver labels found `is_gas_flare`'s OSM-`industrial=refinery`-polygon path was wrong 28/30 times (generic refinery activity with no visible flare, not evidence of an actual flare), and `rules.py` was changed as a direct result — that branch now assigns `industrial` instead of `gas flare`, and `gas flare` requires GEM-distance proximity to a flare-capable facility (see the docstrings on `is_gas_flare`/`is_industrial` in `backend/classification/rules.py`).

That means `gold_sample.csv` fed back into a rule change and can no longer serve as an independent check of that rule — reporting accuracy against it would be validating the rules against the same data that shaped them. `training/data/gold_holdout.csv` (15 detections, drawn post-fix from cells the rule engine now labels `industrial`, at most one per ~375m grid cell, none overlapping any cell already in `gold_sample.csv`, blank `verified_label`/`structure_seen`/`confidence` — same shape as `gold_sample.csv`, no silver label included) is the independent evaluation set going forward: review it blind the same way, and its agreement rate is the number that actually tells you whether the fix generalizes, not `gold_sample.csv`'s.

**The ML model must not train on either file.** `training/train_model.py`'s gold-cell exclusion currently only reads `gold_sample.csv` (`GOLD_SAMPLE_PATH` in `backend/config.py`) — it does not yet know about `gold_holdout.csv`, and needs to before any training run happens against a `labeled_hotspots.csv` built after this holdout was drawn, or `gold_holdout.csv`'s cells (and therefore its usefulness as an independent check) will leak into training the same way an unexcluded `gold_sample.csv` cell would.

#### Pre-fix independent evaluation (`gold_holdout_verified.csv`, 15 rows, reviewed 2026-09-23)

Before the OSM flare-proximity fix described below, blind review of `gold_holdout.csv` against the rules then in `rules.py` (GEM-distance-only `is_gas_flare`, no OSM flare-proximity path) gave:

- **Rules**: 11/12 = 91.7% agreement on decided rows (excludes 3 `uncertain` rows), 11/15 = 73.3% including them.
- **Model** (`model_400k.pkl`): 10/12 = 83.3% agreement on decided rows, 10/15 = 66.7% including them.

Both misclassified the one verified `gas flare` row in the set (H014) as `industrial` — the GEM-distance rule doesn't fire because the nearest flare-capable GEM facility point is farther than `GAS_FLARE_MAX_DIST_M` from the pin, even though a flare stack is visible ~150-200m away in the imagery. This is the independent evidence (not `gold_sample.csv`, which already fed back into a prior fix — see below) that motivated the flare-vs-process-heat investigation below. It did not motivate a rule change: see "Five ways to separate a flare from refinery process heat, none of which worked" below for why.

`daynight` is never used by any labelling rule, deliberately — it's kept as an independent sanity check (real agricultural burning should skew daytime, flares and industrial heat should skew nighttime; that's how this redesign was validated, not how it was built).

#### Five ways to separate a flare from refinery process heat, none of which worked

`is_gas_flare` still requires GEM-distance proximity to a flare-capable facility, with no polygon or proximity fallback — despite `gold_verified1.csv`/`gold_holdout_verified.csv` review turning up real, visually-confirmed flares this rule misses (see above). Five candidate signals were tested against the verified gold rows to find something better, and none separated verified `gas flare` rows from verified `industrial` rows well enough to justify a rule change:

1. **GEM facility distance** (current rule) — 0/11 recall on `gold_verified1.csv`'s verified flares: `dist_to_flare_capable_m` is 1,938-3,416m for every one of them, because the GEM point is the facility's registered centroid, not the flare stack itself.
2. **OSM `industrial=refinery` polygon** (the old, removed path) — recovers 7/11 flares via `recurrence>=2`, but at the cost of *lower* overall rule agreement than the current GEM-only rule on both `gold_verified1.csv` (39.1%/48.2% vs 55.1%/67.9%, overall/decided) and `gold_holdout_verified.csv` (66.7%/83.3% vs 73.3%/91.7%) — it over-calls `gas flare` on plain refinery activity, the exact problem that got it removed originally.
3. **OSM `man_made=flare` node proximity (500m)** — 7/11 true positives, but 17/45 verified-`industrial` rows also fall within 500m of a mapped flare node (worse than 2:1 false-positive ratio); `recurrence_count` doesn't separate the two groups (both span 1-10).
4. **VIIRS `bright_ti4`/`bright_ti5`/`frp`/`daynight`** — gas flare's `bright_ti4` median is ~16K higher (326.5 vs 310.0K) but the distributions overlap heavily; `bright_ti5`, `frp`, and `daynight` show no separation at all; I4-band saturation (367.0K ceiling) hits both groups at similarly low rates (9.1% vs 2.2%, n too small to trust).
5. **Sentinel-2 SWIR (B12) hot-spot check** (`training/data/swir_flare_check_results.json`, method in git history) — nearest 2-3 clear L2A scenes per point, 380m box at 20m resolution, hot pixel = local robust outlier (>median + 4×MAD) and B12 > 0.30. Hit rate 63.6% (flare) vs 55.6% (industrial), hit distance 151m/144m mean vs 144m/152m mean — no separation, and the two rates aren't statistically distinguishable at n=11.

The common failure mode across all five: a 375m VIIRS pixel over a dense refinery complex (Reliance Jamnagar, Nayara Vadinar) contains both a flare stack *and* generic process heat (pipe racks, tank farms, FCC units, substations) within the same footprint or within a few hundred metres of it. Every distance/brightness/SWIR signal tested picks up the process heat as readily as the flare, because at this resolution they usually aren't spatially or spectrally distinct. `is_gas_flare` stays GEM-distance-only rather than adding a signal that trades false negatives (real flares called `industrial`) for a worse rate of false positives (plain refinery activity called `gas flare`) — the latter was the original, specifically-diagnosed failure mode that motivated removing the refinery-polygon path in the first place.

### Contract column changes since the original handoff schema

Removed: **`dist_to_facility_m`, `facility_type`** (single globally-nearest GEM facility, any type — replaced because it picked the wrong facility for large multi-point complexes, e.g. a detection inside the Reliance Jamnagar refinery resolving to a neighbouring cement plant's point instead, since GEM represents each facility as one lat/lon regardless of site size).

Added:
- `dist_to_flare_capable_m` / `nearest_flare_facility_type` — distance to and type of the nearest facility among `oil_gas_power_plant`, `oil_gas_field`, `lng_terminal`, `chemical_plant` (drives `gas flare`).
- `dist_to_heat_industry_m` / `nearest_heat_facility_type` — distance to and type of the nearest facility among `cement_plant`, `steel_plant`, `coal_power_plant`, `coal_mine`, `chemical_plant` (drives `industrial`).
- `is_anomalous` (bool) — see below.
- `osm_industrial_tag` — lowercased `industrial=*` tag of the OSM polygon a detection falls inside (`refinery`, `yes` for untagged `landuse=industrial`, etc.), empty if not inside one. Drives the `industrial`/`gas flare` polygon-containment paths — see above.
- `dist_to_coal_mine_m` / `nearest_coal_source` — distance to the nearest coal-mine evidence and which source supplied it (`gem` or `osm`); see "Coal-mine path" below.
- `dist_to_brick_kiln_m` — distance to the nearest OSM `industrial=brickyard`/`man_made=kiln` feature, OSM-only (no GEM equivalent); see "Brick-kiln path" below.

Both distance columns are still point-to-point nearest-neighbour, so the same wrong-facility failure mode can recur for any sprawling site not well-represented by a single GEM point — treat them as "nearest tagged facility of this type," not "distance to the site boundary." 334 of the 365 hotspots inside the Reliance refinery polygon but outside any curated GEM point's 2km radius are a known, currently-unresolved instance of this.

**Label value renamed:** `industrial fire` → `industrial` (same column, `label`).

### `industrial` class membership (no FRP gate)

A hotspot is `industrial` if any of:
1. within 2000m of a heat-industry facility, not also within 1000m of a flare-capable facility, with `recurrence_count >= 2`, **or**
2. `recurrence_count == 1` and either inside an OSM industrial polygon or within 500m of a heat-industry facility, **or**
3. `recurrence_count >= 2` and inside *any* OSM industrial polygon (including `industrial=refinery` — see below for why), provided the row doesn't already qualify as `gas flare`, **or**
4. coal-mine proximity (added after the 12-month national pull; see "Coal-mine path" below): `recurrence_count >= 2` within 3000m of a GEM-sourced coal mine, **or** `recurrence_count >= 3` within 1000m of an OSM-sourced one, **or**
5. brick-kiln proximity (see "Brick-kiln path" below): `recurrence_count >= 2` within 500m of an OSM `industrial=brickyard`/`man_made=kiln` feature.

There is no absolute FRP threshold in this rule — a small, steady heat source (e.g. a cement kiln) and a large furnace are both legitimate industrial signatures, and gating on FRP magnitude structurally excluded the small ones. `gas flare` requires `recurrence_count >= 5` within 1000m of a flare-capable GEM facility — no polygon-only path (an earlier version also matched `recurrence_count >= 2` inside an OSM `industrial=refinery` polygon with no distance requirement; gold-set verification found that branch wrong 28/30 times — generic refinery activity, not a visible flare — so it was removed and path 3 above no longer excludes refinery polygons either). `gas flare` and `industrial` cannot both fire on the same row, by construction: `industrial`'s path 1 explicitly excludes flare-blocked rows, path 2's `recurrence_count == 1` can never satisfy `gas flare`'s `>= 5`, and paths 3-4 explicitly defer to `gas flare` when it already matches (path 4's OSM branch doesn't need the same guard — it doesn't touch flare-related distances at all).

`osm_industrial_tag` is the lowercased OSM `industrial=*` tag of the polygon a detection falls inside, or empty if it isn't inside any OSM industrial polygon. Plain `landuse=industrial` polygons with no subtype tag get `yes` — present but not a refinery, which is all the rules above need.

#### Coal-mine path

Added after the 12-month national pull showed Jharkhand and West Bengal (India's coal belt) with `unknown` rates far above the national baseline: `HEAT_INDUSTRY_FACILITY_TYPES` already lists `coal_mine`, but `gem_facilities_india.csv` has zero `coal_mine` rows, so `dist_to_heat_industry_m` never actually saw a coal mine. `dist_to_coal_mine_m` / `nearest_coal_source` (`training.spatial_features.nearest_coal_mine_distance`) are a separate signal: GEM (boundary-preferred, falling back to point-distance scaled by `area_km2` for mines GEM only has a centroid for — `effective_distance = max(0, point_distance - sqrt(area_km2 * 1e6 / pi))`) is checked first and wins whenever it has *any* data, regardless of whether OSM's `industrial=mine` match happens to be closer. Closed and mothballed mines are deliberately kept in scope (`GEM_COAL_STATUSES_EXCLUDED` only drops `proposed`/`announced`/`cancelled`/`shelved`) — seam fires outlive active mining.

**As of this writing, neither `gem_coal_mine_boundaries_india.geojson` nor any `coal_mine` row in `gem_facilities_india.csv` exists** — GEM's Global Coal Mine Tracker isn't a direct-downloadable file (gated behind a form on their site, no public URL). `nearest_coal_source` currently reads `"osm"` for every match; the GEM path is implemented and ready, but inert, until one or both files are actually supplied. Re-running `training/recompute_coal_and_relabel.py` (patches an already-built `labeled_hotspots.csv`/Parquet in place — recomputes only `dist_to_coal_mine_m`/`nearest_coal_source` and re-applies `rules.apply_rules()`, reusing every other cached column) against the 12-month national pull moved ~9,800 rows nationally from `unknown` to `industrial` (all OSM-sourced), and Jharkhand's `unknown` rate from 51.4% to 42.1% — real, but understood to be a floor: populating the GEM side is expected to move it further.

#### Brick-kiln path

Same reasoning as the coal-mine path: `HEAT_INDUSTRY_FACILITY_TYPES` has no `brick_kiln` entry at all, because brick kilns aren't a GEM facility type — there's no GEM equivalent to fall back from, so this path is OSM-only, always. `dist_to_brick_kiln_m` (`training.spatial_features.nearest_brick_kiln_distance`, sourced from `backend.ingestion.context_sources.load_osm_brick_kilns`) doesn't reuse `osm_industrial_tag`'s polygon-containment logic, because `man_made=kiln` is typically mapped as an OSM node, not a polygon — `osm_industrial_tag` only reflects being *inside* a polygon, so a detection near (not literally inside) a kiln node would otherwise be invisible to every other `industrial` path. `industrial=brickyard`/`man_made=kiln` are specific tags (unlike `industrial=mine`, generic to any mine type), so this path is trusted at the same recurrence bar as a GEM match (`BRICK_KILN_MIN_RECURRENCE = 2`) despite being OSM-sourced, at a tighter 500m radius.

Rebuilt nationally via the targeted recompute (`training/recompute_industrial_and_relabel.py`, extended to also recompute `dist_to_brick_kiln_m` alongside the industrial-context/solar-wind-exclusion columns it already handled): **557 detections moved `unknown → industrial`** nationally, **zero from `agricultural burning`** — `dist_to_industrial_m` already excludes points this close to a kiln from `agricultural burning`'s `> 2000m` requirement (kilns were already inside `OSM_TARGET_TAGS`), so this path only resolves points that were already too close for `agricultural burning` but hadn't matched any `industrial` path either. Of the 1,277 detections within 500m of a mapped kiln, 619 (recurrence == 1) don't clear the recurrence bar.

**West Bengal (382) and Jharkhand (145) account for 94.6% of the 557 new detections.** Read this as OSM mapping density, not India's actual kiln geography — India's Zig-Zag/FCBTK kiln count is known to be much higher and more evenly spread across the Indo-Gangetic belt (UP, Bihar, Punjab, Haryana) than OSM's `industrial=brickyard`/`man_made=kiln` coverage currently reflects; West Bengal and Jharkhand are simply better-mapped in OSM for this tag, not more kiln-dense in reality. Don't read the state ranking as ground truth without checking OSM completeness for that state first.

Monthly distribution of the 557 matches the expected brick-kiln firing season closely: 95.0% (529/557) fall October-May, 5.0% June-September (monsoon) — a sharp, not gradual, drop-off.

### `is_anomalous`

Per ~375m grid cell, per day/night (VIIRS reads FRP differently under solar illumination, so baselines are kept separate): compare a detection's FRP to that same cell's own **prior** history only — never same-day or future detections. Needs at least 5 prior active days at that cell; before that, always `False` (not "normal", just not yet judged). Baseline is the site's daily-max FRP (median + MAD, robust to outliers); flagged when the modified z-score `0.6745 * (frp - median) / MAD > 3.5`, or `frp > 3 * median` when MAD is 0. One-sided — a drop in FRP, or a missing detection, is never flagged.
