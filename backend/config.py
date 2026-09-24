"""Every cross-layer constant, shared by ingestion, classification, training, and the API.

Deliberately has zero imports from backend/ or training/ submodules, so nothing can
ever form an import cycle through this module. Constants private to one layer (FIRMS/
Overpass plumbing, etc.) stay in that layer's own module instead of being duplicated
here.
"""

from __future__ import annotations

from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent

# --- Rule thresholds (backend/classification/rules.py) ---------------------------
# Values are unchanged from the original rules.py literals -- moved here only so
# both teammates can tune them from one shared file.
GAS_FLARE_MAX_DIST_M = 1_000
GAS_FLARE_MIN_RECURRENCE = 5
INDUSTRIAL_HEAT_MAX_DIST_M = 2_000
INDUSTRIAL_MIN_RECURRENCE = 2
INDUSTRIAL_HEAT_SINGLE_DAY_MAX_DIST_M = 500
INDUSTRIAL_POLYGON_TOLERANCE_M = 1
NATURAL_FIRE_MIN_DIST_FROM_INDUSTRIAL_M = 2_000
# ESA WorldCover class codes, matched exactly (never as substrings -- "10" is a
# substring of "100", moss/lichen, which would otherwise read as tree cover).
CROPLAND_LANDCOVER_CODES = (40,)  # cropland
NATURAL_LANDCOVER_CODES = (10, 20, 30)  # tree cover, shrubland, grassland
# Coal-mine path (is_industrial): GEM (boundary-preferred, point+area-scaled fallback)
# is checked first and trusted at a generous radius -- mining leases are large, and
# seam fires outlive active mining, so GEM_COAL_STATUSES_EXCLUDED below keeps closed/
# mothballed mines in scope and only drops mines that were never actually dug. OSM's
# industrial=mine tag is a supplementary fallback used only where GEM has no coal-mine
# data at all (nationally, today: always -- see GEM_COAL_MINE_BOUNDARIES_PATH /
# GEM_COAL_MINE_CSV_PATH) -- it isn't coal-specific and carries no status, so it's
# trusted at a tighter radius and higher recurrence than a GEM match.
COAL_MINE_MAX_DIST_M = 3_000
COAL_MINE_MIN_RECURRENCE = INDUSTRIAL_MIN_RECURRENCE
COAL_MINE_OSM_MAX_DIST_M = 1_000
COAL_MINE_OSM_MIN_RECURRENCE = 3
# Brick-kiln path (is_industrial): OSM-only, no GEM equivalent (brick kilns aren't a
# GEM facility type at all). industrial=brickyard/man_made=kiln are specific tags
# (unlike industrial=mine, which is generic to any mine type), so trusted at the same
# recurrence bar as a GEM match despite being OSM-sourced -- see is_industrial's
# docstring.
BRICK_KILN_MAX_DIST_M = 500
BRICK_KILN_MIN_RECURRENCE = INDUSTRIAL_MIN_RECURRENCE
# GEM mine-status values that mean "ground was never actually disturbed" -- excluded
# even though every other status (operating, closed, mothballed, retired, ...) is
# kept in scope deliberately (seam fires outlive active mining).
GEM_COAL_STATUSES_EXCLUDED = ("proposed", "announced", "cancelled", "shelved")

# --- Hand-off CSV contract (training/build_labels.py, backend/classification/features.py) ---
CONTRACT_COLUMNS = [
    "latitude", "longitude", "acq_date", "acq_time", "satellite", "bright_ti4", "bright_ti5", "frp",
    "confidence", "daynight", "dist_to_industrial_m", "osm_industrial_tag",
    "dist_to_flare_capable_m", "nearest_flare_facility_type",
    "dist_to_heat_industry_m", "nearest_heat_facility_type",
    "dist_to_coal_mine_m", "nearest_coal_source",
    "dist_to_brick_kiln_m",
    "landcover_class", "recurrence_count", "first_seen", "last_seen",
    "is_anomalous", "label", "label_source",
]

# --- Spatial/temporal feature computation (training/spatial_features.py) ---------
GRID_SIZE = 0.0034  # roughly 375 m at this latitude
MIN_PRIOR_ACTIVE_DAYS = 5
ANOMALY_Z_THRESHOLD = 3.5
# India-wide Lambert conformal conic ("India NSF LCC") -- used for every distance
# computation in training/build_labels.py instead of EPSG:3857/Web Mercator, whose
# distance distortion varies enough across India's latitude range (~6N-37N) to bias
# dist_to_*_m differently in the north vs. the south. Covers all of India including
# Andaman/Nicobar, Lakshadweep, and Sikkim (see its EPSG area-of-use).
NATIONAL_PROJECTED_CRS = "EPSG:7755"
# Dissolved India state/UT boundary (Natural Earth admin-1, filtered to India, all 36
# states/UTs -- see training/report_by_state.py) used to clip FIRMS pulls to India's
# actual shape. INDIA_BBOX in backend/ingestion/firms_client.py is a loose rectangle
# (FIRMS' area API only accepts bbox, not a polygon) that also catches real fires in
# Myanmar, Sri Lanka, and Pakistan -- 47% of a one-month India-bbox pull, in practice.
# This boundary is what actually restricts ingested detections to India.
INDIA_BOUNDARY_PATH = ROOT_DIR / "data" / "external" / "boundaries" / "india_states.parquet"

# --- Training artifacts ------------------------------------------------------------
MODEL_ARTIFACT_PATH = ROOT_DIR / "training" / "artifacts" / "model_400k.pkl"
EVAL_SUMMARY_PATH = ROOT_DIR / "training" / "artifacts" / "eval_summary.json"
GEM_FACILITIES_PATH = ROOT_DIR / "data" / "external" / "gem_facilities_india.csv"
# Neither exists yet as of this writing -- GEM's Global Coal Mine Tracker isn't a
# direct-downloadable file (gated behind a form on their site, no public URL), and
# gem_facilities_india.csv (GEM_FACILITIES_PATH above) currently has zero
# facility_type == "coal_mine" rows. nearest_coal_mine_distance handles both being
# absent/empty gracefully -- it just means every detection's coal-mine evidence comes
# from OSM's industrial=mine tag (backend.ingestion.context_sources) until one or
# both of these are actually supplied.
GEM_COAL_MINE_BOUNDARIES_PATH = ROOT_DIR / "data" / "external" / "gem_coal_mine_boundaries_india.geojson"
GEM_COAL_MINE_CSV_PATH = GEM_FACILITIES_PATH  # facility_type == "coal_mine" rows within it
GOLD_SAMPLE_PATH = ROOT_DIR / "training" / "data" / "gold_sample.csv"
GOLD_HOLDOUT_PATH = ROOT_DIR / "training" / "data" / "gold_holdout.csv"
# Every file whose cells must be excluded from training -- gold_sample.csv drove the
# is_gas_flare rule fix (see README), so it no longer qualifies as an independent
# check; gold_holdout.csv is the independent one, and both must stay out of training.
GOLD_CSV_PATHS = (GOLD_SAMPLE_PATH, GOLD_HOLDOUT_PATH)
RANDOM_SEED = 42
SPATIAL_SPLIT_TEST_FRACTION = 0.2

# --- backend/classification/features.py: missing-value handling ------------------
MISSING_DISTANCE_SENTINEL_M = 50_000.0  # "no candidate facility of this type", not the column mean
MISSING_NUMERIC_SENTINEL = -1.0  # bright_ti4/bright_ti5/frp: an impossible physical value

# --- backend/classification/features.py: categorical vocabularies ----------------
# Used to build a deterministic one-hot column order; a value outside these lists
# at live-inference time routes into that column's trailing "_other" bucket rather
# than raising or silently zeroing the whole row.
SATELLITE_VALUES = ("N", "1", "2")
DAYNIGHT_VALUES = ("D", "N")
LANDCOVER_CLASSES = ("10", "20", "30", "40", "50", "60", "70", "80", "90", "95", "100")
CONFIDENCE_ORDINAL = {"l": 0, "low": 0, "n": 1, "nominal": 1, "h": 2, "high": 2}
CONFIDENCE_DEFAULT_ORDINAL = 1  # unseen/null confidence -> nominal, never raises

# --- training/train_model.py: XGBoost hyperparameters -----------------------------
XGB_N_ESTIMATORS = 300
XGB_MAX_DEPTH = 6
XGB_LEARNING_RATE = 0.1
XGB_SUBSAMPLE = 0.8
XGB_COLSAMPLE_BYTREE = 0.8
XGB_REG_LAMBDA = 1.0
XGB_EARLY_STOPPING_ROUNDS = 20

# --- backend/api ---------------------------------------------------------------
API_HOST = "0.0.0.0"
API_PORT = 8000
CORS_ORIGINS = ["http://localhost:3000", "http://127.0.0.1:3000"]  # placeholder for a standalone-served frontend

# --- frontend/ (served by main.py as static files -- same-origin, no CORS needed) ---
FRONTEND_DIR = ROOT_DIR / "frontend"
DEMO_HOTSPOTS_PATH = ROOT_DIR / "data" / "sample" / "demo_hotspots.csv"
# Class colours, one hue per class in both modes: light steps for the Leaflet page's
# light OSM tiles, dark steps for the national map's dark basemap. Four hues are
# on screen together in a scatter, so every pair must separate: blue/magenta/
# yellow/green is the only 4-hue set of the dataviz reference palette that passes
# the all-pairs normal-vision floor on the dark surface (CVD 6.9, in the band that
# requires secondary encoding -- the class toggles and click popup name the class).
# Red is deliberately absent: it fails against both yellow and magenta.
CLASS_COLORS = {
    "industrial": "#2a78d6",
    "gas flare": "#e87ba4",
    "agricultural burning": "#eda100",
    "wildfire": "#008300",
    "unknown": "#8f8e89",
}
CLASS_COLORS_DARK = {
    "industrial": "#3987e5",
    "gas flare": "#d55181",
    "agricultural burning": "#c98500",
    "wildfire": "#008300",
    "unknown": "#6b6a65",
}

# --- National map (training/pack_map_points.py -> backend/map_store.py) -------------
MAP_POINTS_PATH = ROOT_DIR / "data" / "map" / "national_points.bin"
MAP_POINTS_META_PATH = ROOT_DIR / "data" / "map" / "national_points.json"
NATIONAL_LABELED_PARQUETS = tuple(
    ROOT_DIR / "training" / "data" / f"labeled_hotspots_india_{period}_coalfix.parquet"
    for period in ("12mo_q1", "12mo_q2", "2026-03", "2026-04", "2026-05", "12mo_q4")
)
DUCKDB_MEMORY_LIMIT = "256MB"

# --- Live ingestion (training/ingest_latest.py) ------------------------------------
# National context caches -- the same inputs the historical national build used.
NATIONAL_INDUSTRIAL_CONTEXT_PATH = ROOT_DIR / "data" / "osm" / "india_industrial_context.parquet"
NATIONAL_FACILITIES_PATH = ROOT_DIR / "data" / "external" / "gem_facilities_india.geojson"
NATIONAL_WORLDCOVER_DIR = ROOT_DIR / "data" / "raw" / "worldcover_india"
# Live detections, one parquet per month; never mixed into NATIONAL_LABELED_PARQUETS.
LIVE_DIR = ROOT_DIR / "training" / "data" / "live"
LIVE_PARTITION_PREFIX = "labeled_hotspots_india_live_"
ALERTS_LATEST_PATH = LIVE_DIR / "alerts_latest.csv"  # this run's alerts only; alerts_history.csv beside it keeps them all
# Live recurrence_count counts a site's active days on or before each detection's
# date within this many days. The historical files counted within each period file
# (1-3 months, both directions), so an unbounded look-back would inflate recurrence
# relative to what the rules were checked against.
RECURRENCE_LOOKBACK_DAYS = 90
