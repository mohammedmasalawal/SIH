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

### National dashboard (Vercel)

`/` is a static dashboard. It has:

- every national detection (1.90M) on a GPU map;
- situation KPIs, a class-mix toggle, the top states and a daily timeline per class;
- the alerts panel;
- a **region selector**, which filters the map, KPIs and timeline to one state and outlines it. Its caption reads "Accuracy verified in Gujarat; other states not yet validated", and a badge marks each state as verified or not;
- a **Known facilities** map layer (off by default);
- a **Validation & Method** drawer. It explains the labelling rules and gives the Gujarat gold-set results, the parts not yet validated and the data sources. Its numbers are copied from the sections of this README below; update both together.

The `unknown` class is shown as **"Unclassified — needs review"** (`CLASS_DISPLAY_NAMES` in `backend/config.py`; stored labels are unchanged). Each point's popup opens with **Why**, the `label_source` reason: a rule match with the rule in words, "no labelling rule matched", or "more than one labelling rule matched". **Live: https://agninetra-one.vercel.app** (public). It needs no backend. `training/export_site.py` writes everything the browser reads into `dist/`, and any static host can serve that folder.

```powershell
python -m training.pack_map_points          # -> data/map/national_points.bin (+ .gz, .json sidecar)
python -m training.export_site              # -> dist/ (about 50 s)
python -m training.export_site --deploy     # ...and `vercel deploy --prod` it
python main.py                              # local: http://localhost:8000/ serves frontend/ + dist/data
```

`dist/data/` (about 57 MB, about 200 files):

| File | Contents |
|---|---|
| `points.bin.gz` | the packed points (below), 15.5 MB |
| `meta.json`, `classes.json` | pack layout, date range, export time; class names and colours |
| `stats.json.gz` | detections (and anomalous) per day × class × state, for the KPIs, states and timeline |
| `alerts_history.csv` | every live alert |
| `point_state.bin.gz` | one byte per row id: the detection's state index (`stats.json`'s state order), so the map can filter to a state on the GPU |
| `states.json.gz` | simplified state outlines and bounding boxes (Natural Earth, `INDIA_BOUNDARY_PATH`) |
| `facilities.json.gz` | the known-facilities layer, about 35k points. It comes from the same inputs the rules measure distances to: GEM facilities and the OSM industrial-context layer, with solar/wind plants excluded as in the pipeline. OSM polygons are shown as a point inside the outline, grouped as GEM, OSM refinery or fossil/nuclear power plant, or other OSM industrial site |
| `details/NNNN.json.gz` + `index.json` | full records, 10,000 rows per shard, columnar (coordinates ×1e5, FRP ×100, repeated strings as dictionary codes). A click on a point loads shard `row_id // 10000`, one fetch of about 250 KB, cached after that |

`.gz` files are served as plain `application/gzip`, and the browser decompresses them itself with `DecompressionStream`, so every host behaves the same. Gzip is written with a fixed timestamp, so unchanged shards produce identical bytes. Live data only changes the last shards and the stats, and a redeploy uploads only those. Vercel Hobby limits leave plenty of room: 100 MB per CLI upload, 15,000 files, 100 deploys/day. The project link lives in `dist/.vercel/` (gitignored with the rest of `dist/`), and the first `--deploy` needs `npm i -g vercel` and `vercel login`.

**Auto-deploy:** after each scheduled ingest, `training/run_ingest.cmd` runs `python -m training.export_site --deploy --if-changed`. That step exports and deploys only when something is newer than the last successful deploy (`dist/.deployed`): the map pack, `alerts_history.csv` or a page file. A skipped or failed deploy is therefore retried on the next run. Its output, timestamped, goes to `ingest.log`. A failed deploy never changes the task's result: the ingest exit code is returned.

**Deploy retries and failure logging:** a failed `vercel deploy` is retried after about 30 s and again after about 90 s (`DEPLOY_RETRY_WAITS_S`), and each CLI call times out after 15 min. If all three attempts fail, `ingest.log` gets `Deploy FAILED` followed by every attempt's full CLI stdout and stderr, with the token replaced by `***`. The next run tries again, because `dist/.deployed` isn't updated. Before this change, two scheduled deploys on 26 Sep failed without a logged reason. Vercel has no record of either, so the CLI failed locally before uploading. A manual retry of the same command worked.

**Vercel token:** scheduled deploys authenticate with `VERCEL_TOKEN`, not the interactive `vercel login`, because a login session can expire or be logged out. `run_ingest.cmd` passes `--require-token`, so if the token is missing the deploy fails with `VERCEL_TOKEN is not set` rather than quietly falling back to the login. The token is read from the environment or `.env` (gitignored), the same way as `FIRMS_MAP_KEY`. It is handed to the CLI in its environment, never as `--token`, so it can't appear in a process listing or a traceback. To create one:

1. Sign in at vercel.com and open **Account Settings → Tokens** (https://vercel.com/account/settings/tokens).
2. **Create Token**:
   - **Name:** e.g. `agninetra-laptop-deploy`.
   - **Scope:** the team that owns the `agninetra` project (the one shown by `vercel ls`).
   - **Expiration:** 1 year is reasonable. When it expires, deploys fail with `Not authorized` in `ingest.log`.
3. Copy the token. Vercel shows it only once.
4. Add a line `VERCEL_TOKEN=<token>` to `.env` in the repo root. Never commit it; `.env` is gitignored.
5. Test it: `python -m training.export_site --deploy --require-token`. This exports, runs the checks and deploys using only the token. The next scheduled run picks it up automatically.

If the token leaks, delete it on the same Tokens page and create a new one.

**Pre-deploy checks** (`training/site_preflight.py`, runnable alone as `python -m training.site_preflight`). They run before every `--deploy`. If any check fails, the deploy is skipped and each reason is logged (`Deploy skipped -- pre-deploy checks failed:`):

1. `data/points.bin.gz` exists, is non-empty, and decompresses to the byte count `meta.json` promises.
2. `data/alerts_history.csv` parses and has the panel's columns. Every row has a numeric coordinate, a valid date and a known `alert_type`, and there are no duplicate `alert_id`s.
3. The page loads in headless Chrome, served from `dist/` over local HTTP. It must render its points, fill the alerts count and decode one detail record, with no JavaScript exception, console error or failed request. This takes about 6 s. The check looks for Chrome or Edge in the standard install paths; set `AGNINETRA_CHROME` to point elsewhere.

Deliberately breaking the export confirmed that the checks block a deploy for a thrown script error and for a missing data file. The local server's `/data` also comes from `dist/data`, so it refreshes on the same step.

- **Points:** `training/pack_map_points.py` packs these fields, 16 bytes per detection: lon/lat (float32), row id (uint32), day index (uint16), class (uint8) and `is_anomalous` (uint8). Points are stored in draw order so rare classes paint on top: unknown → agricultural burning → wildfire → industrial → gas flare. The row id maps each point back to its parquet row. Repack whenever the parquets change, then re-export: row ids are only valid against the files listed in the sidecar.
- **Rendering:** MapLibre GL (CARTO dark-matter basemap) with a deck.gl `ScatterplotLayer`. Month range and class toggles filter on the GPU via `DataFilterExtension`, so filter changes never re-upload points. URL parameters `month=YYYY-MM`, `all=1`, `lat`, `lng` and `zoom` set the initial view. Times are shown in IST, with UTC in the popup.
- **Alerts panel:** lists `data/alerts_history.csv` newest first, with a type badge, a type filter, and the site, date, facility and metric for each alert. Clicking an alert flies the map there, opens its popup and switches the month.
- **Performance:** measured with all 1.89M points on screen at 1400×900: about 116 fps panning on an RTX 3050 laptop GPU, but about 19 fps on the same laptop's Intel UHD integrated GPU (about 38 fps with one month shown). Chrome on Windows laptops uses the integrated GPU by default, so for smooth panning set Chrome to "High performance" under Windows Settings → System → Display → Graphics. The Vercel page loads in about 4 s.
- **Local only:** the FastAPI server still serves `/api/map/*`, `/api/detection/{row_id}` (DuckDB row lookup), `/api/classify` and the old Leaflet demo page at `/legacy`. None of these exist on Vercel.

### Live ingestion

`training/ingest_latest.py` keeps the national data current. Each run pulls the last N days (default 2, UTC) of VIIRS SNPP / NOAA-20 / NOAA-21 **NRT** detections over India, clips them to India's boundary, drops duplicates on `(latitude, longitude, acq_date, acq_time, satellite)` against everything already stored (so re-running a window adds nothing), and then:

- computes the context columns with `build_labels.add_context_features` — the same cached OSM / GEM / coal-mine / WorldCover inputs as the historical build;
- computes `recurrence_count` and `is_anomalous` against the historical + live store using **earlier detections only**. Recurrence counts a site's active days on or before the detection's date within `RECURRENCE_LOOKBACK_DAYS` (90) — the historical files counted within each 1-3-month period file, so an unbounded look-back would inflate recurrence relative to what the rules were checked against (`--lookback-days 0` = all history). `is_anomalous` uses the site's full prior history, as it always has;
- applies the rules unchanged and appends to `training/data/live/labeled_hotspots_india_live_YYYY-MM.parquet` (one file per month; existing rows never move). The historical 12-month parquets are only read;
- writes `training/data/live/alerts_latest.csv` — this run's `is_anomalous` detections labelled `industrial` or `gas flare`, with FRP vs the site's normal (median prior daily-max FRP) and the facility behind the label; it is empty when a run finds nothing new, so every alert is also appended to `alerts_history.csv` beside it — then repacks the map; `run_ingest.cmd` then re-exports and redeploys the dashboard (see "National dashboard").

```powershell
python -m training.ingest_latest                                   # last 2 days
python -m training.ingest_latest --start 2026-09-01 --end 2026-09-23   # backfill
```

A lock file (`training/data/live/ingest.lock`) stops overlapping runs.

**Run it every 6 hours (Windows Task Scheduler)** — `training/run_ingest.cmd` sets the working directory, pulls the **last 3 days** on every run (overlapping windows cost nothing — stored detections are skipped — and a missed run or a day with the machine off leaves no gap, as long as it's back within ~3 days), and appends output to `training/data/live/logs/ingest.log`. Register it from PowerShell:

```powershell
$action    = New-ScheduledTaskAction -Execute "C:\Windows\System32\conhost.exe" `
               -Argument '--headless cmd.exe /c "C:\Users\Admin\Documents\Agninetra\training\run_ingest.cmd"' `
               -WorkingDirectory "C:\Users\Admin\Documents\Agninetra"
$trigger   = New-ScheduledTaskTrigger -Once -At "00:15" -RepetitionInterval (New-TimeSpan -Hours 6)
$settings  = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -StartWhenAvailable `
               -MultipleInstances IgnoreNew -ExecutionTimeLimit (New-TimeSpan -Hours 2)
$principal = New-ScheduledTaskPrincipal -UserId "$env:USERDOMAIN\$env:USERNAME" -LogonType Interactive
Register-ScheduledTask -TaskName "Agninetra live ingest" -Action $action -Trigger $trigger -Settings $settings -Principal $principal
Get-ScheduledTaskInfo -TaskName "Agninetra live ingest"   # last/next run and result
```

Why these settings: `-AllowStartIfOnBatteries`/`-DontStopIfGoingOnBatteries` (Task Scheduler's defaults skip or kill runs on a laptop unplugged), `-StartWhenAvailable` (a run missed while the machine was asleep or off starts as soon as it's back), `-MultipleInstances IgnoreNew` (never two runs at once; the ingest lock file is the second guard), and `conhost.exe --headless` so no console window opens during a run. `Interactive` means it runs only while you're logged in; running logged-out (`-LogonType S4U`) needs an administrator PowerShell. The FIRMS key is read from `.env`, so the task needs no extra environment. A running API server picks up each repack on the next request (the sidecar's modification time is checked).

### Live alerts

`training/alerts.py` raises four alert types on each ingest run (every alert has an `alert_type` and a stable `alert_id`, so re-evaluating a window never raises the same alert twice; alerts never change a label). **Every threshold below is an untested starting value** — none has been checked against verified ground truth. All live in `backend/config.py`.

| `alert_type` | Fires when | Thresholds (config) | Extra fields |
|---|---|---|---|
| `industrial_anomaly` | an `is_anomalous` detection labelled `industrial` or `gas flare` (unchanged from the first version) | — (the `is_anomalous` rule) | FRP vs the site's normal, facility |
| `new_activity_at_critical_site` | a detection, **any label**, within `CRITICAL_SITE_MAX_DIST_M` (2 km) of a critical facility — GEM refinery, chemical, thermal (oil/gas, coal) power, LNG terminal, oil/gas field, cement or steel plant (`CRITICAL_SITE_GEM_FACILITY_TYPES`), OSM `industrial=refinery` (`CRITICAL_SITE_OSM_INDUSTRIAL_TAGS`), or OSM `power=plant` whose `plant:source` includes coal/gas/oil/diesel/nuclear (`CRITICAL_SITE_OSM_POWER_SOURCES`; solar, wind, hydro, biomass, waste and untagged plants never count) — whose **~1 km neighbourhood** had no other active day in the previous `CRITICAL_SITE_LOOKBACK_DAYS` (90) (`CRITICAL_SITE_MAX_ACTIVE_DAYS` = 1, the detection's own day) | 2 km; quiet ~1 km area for 90 days | facility name, kind, distance |
| `new_unmapped_source` | a ~375 m cell with no known facility (OSM industrial, GEM heat/flare, coal mine, brick kiln) within `UNMAPPED_NO_FACILITY_WITHIN_M` that reaches `UNMAPPED_MIN_ACTIVE_DAYS` active days in the trailing `UNMAPPED_WINDOW_DAYS`, with at most `UNMAPPED_MAX_PRIOR_ACTIVE_DAYS` before that window; once per cell, ever | 2 km, 5 of 30 days, ≤1 day before | active days in 30, total active days, first seen |
| `large_fire_event` | `LARGE_FIRE_MIN_DETECTIONS`+ `agricultural burning`/`wildfire` detections (`LARGE_FIRE_LABELS`) within `LARGE_FIRE_RADIUS_M` on the same day (DBSCAN, so a cluster can chain beyond 5 km), after dropping detections whose **~1 km neighbourhood** was active on more than `LARGE_FIRE_PERSISTENT_MAX_ACTIVE_DAYS` of the previous `LARGE_FIRE_PERSISTENT_WINDOW_DAYS` (persistent industrial sites, coal-seam fires); judged only on complete UTC days | 10 within 5 km; >5 of 30 days = persistent | detection count, dominant class, total FRP |

**Neighbourhood** = the detection's ~375 m grid cell plus the cells within `NEIGHBOURHOOD_RADIUS_CELLS` (1 → a 3×3 block, ~1.1 km across; `training.alerts.neighbourhood_active_days`). VIIRS geolocation jitters between passes, so the same physical source often lands in an adjacent cell — a single-cell check made long-active spots look new and let drifting coal-seam fires look intermittent.

What the September 2026 backfill (1–23 Sep) says about these starting values — 171 alerts after the latest re-score (504 under the first definitions, 290 with single-cell checks):

- `industrial_anomaly`: 110, unchanged (includes both September gas-flare alerts, Mangaluru 17.1× and Hazira 4.8×).
- `new_activity_at_critical_site`: 52 at 27 facilities (was 167 with the single-cell check) — OSM coal plants 26, refineries 15, gas plants 10, one GEM cement plant; 37 `industrial`, 15 `unknown`, median 214 m, e.g. Gujarat and Reliance refineries, Essar Power, Guwahati Refinery, the Bhilai 500 MW expansion. **No crop or wildfire detection can appear here in practice**: those labels require no OSM industrial-context feature within 2 km, and that context contains every OSM refinery and non-renewable power plant — only a GEM facility point could pair with one.
- `large_fire_event`: 6, all `wildfire`/`agricultural burning` (was 10 with the single-cell check, 263 before the class restriction). **Jharia coalfield: 3 events → 1** — the 12 and 14 Sep events drop out, but 3 Sep survives with 10 detections (down from 16) because the burning moved onto ground that was quiet in the previous 30 days. The two Odisha events at 22.03 N, 83.73 E also dropped. A wider neighbourhood, or excluding clusters near OSM `industrial=mine`, are the next things to try.
- `new_unmapped_source`: 3.

Re-score stored detections after changing a threshold or definition (no re-fetch): `python -m training.ingest_latest --start 2026-09-01 --end 2026-09-23 --alerts-only`. This **replaces** the history's alerts dated inside that window with the new evaluation (alerts outside it are untouched), so alerts the new definitions no longer raise leave `alerts_history.csv`. The map's alerts panel shows a type badge per alert and filters by type.

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
4. coal-mine proximity (added after the 12-month national pull; see "Coal-mine path" below): `recurrence_count >= 2` within 3000m of a GEM-sourced coal mine, **or** `recurrence_count >= 3` within 1000m of an OSM-sourced one.

A fifth path, brick-kiln proximity, was added and then removed after gold-set verification — see "Brick-kiln path (tried, removed)" below.

There is no absolute FRP threshold in this rule — a small, steady heat source (e.g. a cement kiln) and a large furnace are both legitimate industrial signatures, and gating on FRP magnitude structurally excluded the small ones. `gas flare` requires `recurrence_count >= 5` within 1000m of a flare-capable GEM facility — no polygon-only path (an earlier version also matched `recurrence_count >= 2` inside an OSM `industrial=refinery` polygon with no distance requirement; gold-set verification found that branch wrong 28/30 times — generic refinery activity, not a visible flare — so it was removed and path 3 above no longer excludes refinery polygons either). `gas flare` and `industrial` cannot both fire on the same row, by construction: `industrial`'s path 1 explicitly excludes flare-blocked rows, path 2's `recurrence_count == 1` can never satisfy `gas flare`'s `>= 5`, and paths 3-4 explicitly defer to `gas flare` when it already matches (path 4's OSM branch doesn't need the same guard — it doesn't touch flare-related distances at all).

`osm_industrial_tag` is the lowercased OSM `industrial=*` tag of the polygon a detection falls inside, or empty if it isn't inside any OSM industrial polygon. Plain `landuse=industrial` polygons with no subtype tag get `yes` — present but not a refinery, which is all the rules above need.

#### Coal-mine path

Added after the 12-month national pull showed Jharkhand and West Bengal (India's coal belt) with `unknown` rates far above the national baseline: `HEAT_INDUSTRY_FACILITY_TYPES` already lists `coal_mine`, but `gem_facilities_india.csv` has zero `coal_mine` rows, so `dist_to_heat_industry_m` never actually saw a coal mine. `dist_to_coal_mine_m` / `nearest_coal_source` (`training.spatial_features.nearest_coal_mine_distance`) are a separate signal: GEM (boundary-preferred, falling back to point-distance scaled by `area_km2` for mines GEM only has a centroid for — `effective_distance = max(0, point_distance - sqrt(area_km2 * 1e6 / pi))`) is checked first and wins whenever it has *any* data, regardless of whether OSM's `industrial=mine` match happens to be closer. Closed and mothballed mines are deliberately kept in scope (`GEM_COAL_STATUSES_EXCLUDED` only drops `proposed`/`announced`/`cancelled`/`shelved`) — seam fires outlive active mining.

**As of this writing, neither `gem_coal_mine_boundaries_india.geojson` nor any `coal_mine` row in `gem_facilities_india.csv` exists** — GEM's Global Coal Mine Tracker isn't a direct-downloadable file (gated behind a form on their site, no public URL). `nearest_coal_source` currently reads `"osm"` for every match; the GEM path is implemented and ready, but inert, until one or both files are actually supplied. Re-running `training/recompute_coal_and_relabel.py` (patches an already-built `labeled_hotspots.csv`/Parquet in place — recomputes only `dist_to_coal_mine_m`/`nearest_coal_source` and re-applies `rules.apply_rules()`, reusing every other cached column) against the 12-month national pull moved ~9,800 rows nationally from `unknown` to `industrial` (all OSM-sourced), and Jharkhand's `unknown` rate from 51.4% to 42.1% — real, but understood to be a floor: populating the GEM side is expected to move it further.

#### Brick-kiln path (tried, removed)

Same reasoning as the coal-mine path: `HEAT_INDUSTRY_FACILITY_TYPES` has no `brick_kiln` entry at all, because brick kilns aren't a GEM facility type — there's no GEM equivalent to fall back from, so a dedicated OSM-only path (`dist_to_brick_kiln_m <= 500m`, `recurrence_count >= 2`, mirroring the coal-mine path's structure) was added. `dist_to_brick_kiln_m` (`training.spatial_features.nearest_brick_kiln_distance`, sourced from `backend.ingestion.context_sources.load_osm_brick_kilns`, matching `industrial=brickyard`/`man_made=kiln`) doesn't reuse `osm_industrial_tag`'s polygon-containment logic, because `man_made=kiln` is typically mapped as an OSM node, not a polygon.

Rebuilt nationally via the targeted recompute (`training/recompute_industrial_and_relabel.py`, extended to also recompute `dist_to_brick_kiln_m`): **557 detections moved `unknown → industrial`** nationally, **zero from `agricultural burning`** — `dist_to_industrial_m` already excludes points this close to a kiln from `agricultural burning`'s `> 2000m` requirement (kilns were already inside `OSM_TARGET_TAGS`), so this path only resolved points that were already too close for `agricultural burning` but hadn't matched any `industrial` path either. Of the 1,277 detections within 500m of a mapped kiln, 619 (recurrence == 1) didn't clear the recurrence bar. West Bengal (382) and Jharkhand (145) accounted for 94.6% of the 557 — reflecting OSM `industrial=brickyard`/`man_made=kiln` mapping density in those states, not India's actual kiln geography (real kiln counts are known to be higher and more evenly spread across the Indo-Gangetic belt, including UP/Bihar/Punjab/Haryana). Monthly distribution matched the expected firing season: 95.0% (529/557) fell October-May, 5.0% June-September (monsoon).

**Removed after gold-set verification** (`training/data/gold_brick_kiln_verified.csv`, 10 of the 557, drawn across 7 states and 10 distinct ~5km kiln clusters): only 3-4 of 10 were clear hits — **3 of the 7 decided rows** verified `industrial`, 3 rows `uncertain` (a real kiln or industrial plant within ~300m of the pin — 2 kilns, 1 steel plant, 1 borderline case where the kiln was visible but 600m away, past the pin). The rest were farm or forest fires 500-800m from a mapped kiln — close enough to trip the 500m radius without the kiln being the actual heat source. That's below the pre-set 5/10 threshold, so the path was removed from `is_industrial` (rules.py) and the national data was recomputed back to its pre-brick-kiln-path state (the same 557 rows revert cleanly to `unknown`, confirmed via the targeted recompute — no other rows moved).

`dist_to_brick_kiln_m` is still computed and `industrial=brickyard`/`man_made=kiln` remain in the general OSM industrial-context match (`OSM_TARGET_TAGS`) — only the dedicated 500m proximity-triggered path was removed. Checked but **not implemented**: excluding `industrial=brickyard` from that general context (the same treatment solar/wind power plants got) would flip ~2,611 rows nationally, almost entirely `unknown → agricultural burning` (2,451) and `unknown → wildfire` (150) — `dist_to_industrial_m` currently blocks those rows from qualifying as `agricultural burning`/`wildfire` at all (`> 2000m` requirement) purely because a brickyard polygon happens to be within range, not because anything at the pin is industrial. Only 10 already-`industrial` rows would be affected by that exclusion (3 to `agricultural burning`, 7 to `unknown`) — nowhere near the null hypothesis that `industrial=brickyard` context drives most `industrial` calls. This wasn't applied; it's reported here as a candidate follow-up, not a decision.

**Future work:** the 3-4 verified hits were real kilns/plants within ~300m of the pin, well inside the 500m radius that was tested — a narrower radius (e.g. 300m) on a fresh, larger gold sample is the natural next test before deciding whether any brick-kiln path is worth re-adding.

### `agricultural burning` / `wildfire` landcover matching

Both rules key on the ESA WorldCover class in `landcover_class`, matched as **exact integer codes** (`CROPLAND_LANDCOVER_CODES = (40,)`, `NATURAL_LANDCOVER_CODES = (10, 20, 30)` in `backend/config.py`; `10`, `"10"`, and `10.0` all parse to the same code), plus the shared `dist_to_industrial_m > 2000m` requirement. They used to substring-match (`"10" in "100"`), which read class 100 (moss/lichen) as tree cover — 66 national detections were labelled `wildfire` on that basis. Fixed and recomputed via the targeted recompute: exactly those 66 moved `wildfire → unknown`, nothing else changed (`tests/test_rules.py` has the class-100 regression test). No other rule matches landcover; OSM tag rules use exact equality. The Jamnagar regional build has no class-100 pixels and is unaffected.

#### Shrubland-based wildfire labels (finding, no rule change)

Gold-set review around Jamnagar (`training/data/gold_verified_agri_wildfire.csv`) found the `wildfire` rule weak there: of 9 rows the rules called `wildfire`, only 4 were verified wildfire, and all 5 misses were verified `agricultural burning` — every one of the 9 sits on WorldCover class 20 (shrubland). In semi-arid Saurashtra, WorldCover's shrubland class often covers fallow or bare cropland, so field burning there reads as wildfire. (`agricultural burning` held up: 9/10.)

Nationally (723,223 `wildfire` detections, after the class-100 fix):

| WorldCover class | `wildfire` detections | share |
|---|---|---|
| 10 tree cover | 536,217 | 74.14% |
| 30 grassland | 134,550 | 18.60% |
| 20 shrubland | 52,456 | **7.25%** |

The national shrubland share is small, but it's concentrated: Andhra Pradesh (23,384) and Karnataka (11,766) hold 67% of all shrubland-based wildfire labels, and shrubland underpins 38.1% and 51.7% of those states' wildfire labels respectively (Gujarat 25.6%, Rajasthan 22.5%, Tamil Nadu 19.2%). If the Jamnagar pattern holds in those states, a large fraction of their `wildfire` labels may actually be crop-residue burning. **Grassland (class 30, 18.6% of national wildfire) is untested** — it may carry the same fallow-cropland confusion in semi-arid areas.

**Future work:** regional gold samples of shrubland- and grassland-based `wildfire` detections in Andhra Pradesh, Karnataka, and Rajasthan, before any change to how classes 20/30 are treated. The Jamnagar result is one semi-arid region and hasn't been validated outside Gujarat.

### `is_anomalous`

Per ~375m grid cell, per day/night (VIIRS reads FRP differently under solar illumination, so baselines are kept separate): compare a detection's FRP to that same cell's own **prior** history only — never same-day or future detections. Needs at least 5 prior active days at that cell; before that, always `False` (not "normal", just not yet judged). Baseline is the site's daily-max FRP (median + MAD, robust to outliers); flagged when the modified z-score `0.6745 * (frp - median) / MAD > 3.5`, or `frp > 3 * median` when MAD is 0. One-sided — a drop in FRP, or a missing detection, is never flagged.
