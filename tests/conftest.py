from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
import pytest
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

from backend.classification.features import FEATURE_COLUMNS, hotspots_to_feature_matrix


@pytest.fixture
def sample_hotspot_dict() -> dict:
    """A well-formed, fully-populated hotspot -- the happy path for feature/model tests."""
    return {
        "latitude": 22.30, "longitude": 69.85, "acq_date": "2026-01-03", "acq_time": "0130",
        "satellite": "N", "bright_ti4": 330.0, "bright_ti5": 290.0, "frp": 7.0,
        "confidence": "n", "daynight": "D",
        "dist_to_industrial_m": 50.0,
        "dist_to_flare_capable_m": 20_000.0, "nearest_flare_facility_type": "oil_gas_field",
        "dist_to_heat_industry_m": 163.9, "nearest_heat_facility_type": "cement_plant",
        "landcover_class": "50", "recurrence_count": 3, "is_anomalous": False,
    }


@pytest.fixture
def synthetic_labeled_frame() -> pd.DataFrame:
    """A small multi-cell labeled_hotspots.csv-shaped frame spanning 2 surviving classes + unknown.

    Grid cells (GRID_SIZE=0.0034) are spaced far enough apart that grid_keys never
    merges two clusters, and there are enough cells per class for spatial_block_split
    to plausibly keep both classes represented on both sides of a small split.
    """
    rows = []
    industrial_centers = [(22.30, 69.85), (22.50, 70.05), (22.70, 70.25), (22.90, 70.45)]
    for lat, lon in industrial_centers:
        for day in range(1, 4):
            rows.append(dict(
                latitude=lat, longitude=lon, acq_date=f"2026-01-{day:02d}", acq_time="0130",
                satellite="N", bright_ti4=330.0, bright_ti5=290.0, frp=5.0 + day,
                confidence="n", daynight="D",
                dist_to_industrial_m=50.0, dist_to_flare_capable_m=20_000.0, nearest_flare_facility_type=None,
                dist_to_heat_industry_m=150.0, nearest_heat_facility_type="cement_plant",
                landcover_class="50", recurrence_count=3, first_seen="2026-01-01", last_seen="2026-01-03",
                is_anomalous=False, label="industrial", label_source="rule",
            ))
    flare_centers = [(23.10, 69.60), (23.30, 69.40), (23.50, 69.20), (23.70, 69.00)]
    for lat, lon in flare_centers:
        for day in range(1, 6):
            rows.append(dict(
                latitude=lat, longitude=lon, acq_date=f"2026-02-{day:02d}", acq_time="0200",
                satellite="N", bright_ti4=340.0, bright_ti5=300.0, frp=15.0,
                confidence="h", daynight="N",
                dist_to_industrial_m=5_000.0, dist_to_flare_capable_m=100.0, nearest_flare_facility_type="oil_gas_field",
                dist_to_heat_industry_m=20_000.0, nearest_heat_facility_type=None,
                landcover_class="60", recurrence_count=5, first_seen="2026-02-01", last_seen="2026-02-05",
                is_anomalous=False, label="gas flare", label_source="rule",
            ))
    rows.append(dict(
        latitude=25.0, longitude=68.0, acq_date="2026-03-01", acq_time="0300",
        satellite="N", bright_ti4=310.0, bright_ti5=270.0, frp=3.0,
        confidence="l", daynight="D",
        dist_to_industrial_m=8_000.0, dist_to_flare_capable_m=25_000.0, nearest_flare_facility_type=None,
        dist_to_heat_industry_m=9_000.0, nearest_heat_facility_type=None,
        landcover_class="80", recurrence_count=1, first_seen="2026-03-01", last_seen="2026-03-01",
        is_anomalous=False, label="unknown", label_source="unknown",
    ))
    return pd.DataFrame(rows)


@pytest.fixture
def tiny_fitted_artifact(tmp_path: Path, synthetic_labeled_frame: pd.DataFrame) -> Path:
    """Trains a throwaway 2-class model and joblib-dumps it in the real artifact shape."""
    frame = synthetic_labeled_frame[synthetic_labeled_frame["label"] != "unknown"].reset_index(drop=True)
    x = hotspots_to_feature_matrix(frame)
    encoder = LabelEncoder()
    y = encoder.fit_transform(frame["label"])
    model = XGBClassifier(n_estimators=5, max_depth=2, device="cpu", tree_method="hist")
    model.fit(x, y)
    artifact = {
        "model": model,
        "feature_columns": list(FEATURE_COLUMNS),
        "label_classes": list(encoder.classes_),
        "metadata": {"trained_at": "2026-01-01T00:00:00+00:00", "training_data_path": "test", "random_seed": 42},
    }
    path = tmp_path / "model.pkl"
    joblib.dump(artifact, path)
    return path
