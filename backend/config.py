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
CROPLAND_LANDCOVER_TERMS = ("cropland", "crop", "agriculture", "40")
NATURAL_LANDCOVER_TERMS = ("forest", "shrub", "grass", "woodland", "10", "20", "30")

# --- Hand-off CSV contract (training/build_labels.py, backend/classification/features.py) ---
CONTRACT_COLUMNS = [
    "latitude", "longitude", "acq_date", "acq_time", "satellite", "bright_ti4", "bright_ti5", "frp",
    "confidence", "daynight", "dist_to_industrial_m",
    "dist_to_flare_capable_m", "nearest_flare_facility_type",
    "dist_to_heat_industry_m", "nearest_heat_facility_type",
    "landcover_class", "recurrence_count", "first_seen", "last_seen",
    "is_anomalous", "label", "label_source",
]

# --- Spatial/temporal feature computation (training/spatial_features.py) ---------
GRID_SIZE = 0.0034  # roughly 375 m at this latitude
MIN_PRIOR_ACTIVE_DAYS = 5
ANOMALY_Z_THRESHOLD = 3.5

# --- Training artifacts ------------------------------------------------------------
MODEL_ARTIFACT_PATH = ROOT_DIR / "training" / "artifacts" / "model.pkl"
EVAL_SUMMARY_PATH = ROOT_DIR / "training" / "artifacts" / "eval_summary.json"
GEM_FACILITIES_PATH = ROOT_DIR / "data" / "external" / "gem_facilities_india.csv"
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
CLASS_COLORS = {
    "industrial": "#e6550d",
    "gas flare": "#fdae6b",
    "agricultural burning": "#8c6d31",
    "wildfire": "#d62728",
    "unknown": "#7f7f7f",
}
