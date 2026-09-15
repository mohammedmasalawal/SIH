"""Hotspot (CONTRACT_COLUMNS-shaped) -> numeric ML feature vector.

Shared by training/train_model.py (batch, from labeled_hotspots.csv) and
backend/classification/model_classifier.py (single live hotspot), so there is
exactly one encoding implementation -- train and serve can never silently drift
apart. latitude/longitude/acq_date/acq_time/first_seen/last_seen are deliberately
excluded from the vector: raw coordinates would let a model memorize which grid
cell was which class in training rather than generalising from distance/landcover
signal, which would defeat the point of the spatial-block train/test split.
"""

from __future__ import annotations

import pandas as pd

from backend.classification.rules import FLARE_CAPABLE_FACILITY_TYPES, HEAT_INDUSTRY_FACILITY_TYPES
from backend.config import (
    CONFIDENCE_DEFAULT_ORDINAL,
    CONFIDENCE_ORDINAL,
    DAYNIGHT_VALUES,
    LANDCOVER_CLASSES,
    MISSING_DISTANCE_SENTINEL_M,
    MISSING_NUMERIC_SENTINEL,
    SATELLITE_VALUES,
)

_DISTANCE_COLUMNS = ("dist_to_industrial_m", "dist_to_flare_capable_m", "dist_to_heat_industry_m")

# Order here fixes the categorical block order in FEATURE_COLUMNS. nearest_flare/heat
# facility type vocabularies are the exact facility-type tuples rules.py already uses,
# so a value here can never diverge from what those columns can actually contain.
CATEGORICAL_VOCABULARIES: dict[str, tuple[str, ...]] = {
    "satellite": SATELLITE_VALUES,
    "daynight": DAYNIGHT_VALUES,
    "landcover_class": LANDCOVER_CLASSES,
    "nearest_flare_facility_type": FLARE_CAPABLE_FACILITY_TYPES,
    "nearest_heat_facility_type": HEAT_INDUSTRY_FACILITY_TYPES,
}

_NUMERIC_FEATURE_COLUMNS = [
    "bright_ti4", "bright_ti4_missing",
    "bright_ti5", "bright_ti5_missing",
    "frp", "frp_missing",
    "confidence_ordinal",
    "dist_to_industrial_m", "dist_to_industrial_m_missing",
    "dist_to_flare_capable_m", "dist_to_flare_capable_m_missing",
    "dist_to_heat_industry_m", "dist_to_heat_industry_m_missing",
    "recurrence_count",
    "is_anomalous",
]


def _one_hot_columns(column: str, vocabulary: tuple[str, ...]) -> list[str]:
    return [f"{column}__{value}" for value in vocabulary] + [f"{column}__other"]


FEATURE_COLUMNS: list[str] = list(_NUMERIC_FEATURE_COLUMNS)
for _column, _vocabulary in CATEGORICAL_VOCABULARIES.items():
    FEATURE_COLUMNS.extend(_one_hot_columns(_column, _vocabulary))


def _is_null(value: object) -> bool:
    if value is None:
        return True
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def _numeric_or_sentinel(value: object, sentinel: float) -> tuple[float, float]:
    """(value_as_float, missing_flag), falling back to sentinel+1.0 for null/unparseable input."""
    if _is_null(value):
        return sentinel, 1.0
    try:
        return float(value), 0.0
    except (TypeError, ValueError):
        return sentinel, 1.0


def _confidence_ordinal(value: object) -> float:
    if _is_null(value):
        return float(CONFIDENCE_DEFAULT_ORDINAL)
    return float(CONFIDENCE_ORDINAL.get(str(value).strip().lower(), CONFIDENCE_DEFAULT_ORDINAL))


def _recurrence_feature(value: object) -> float:
    if _is_null(value):
        return 1.0
    try:
        return float(value)
    except (TypeError, ValueError):
        return 1.0


def _bool_feature(value: object) -> float:
    if _is_null(value):
        return 0.0
    if isinstance(value, str):
        return 1.0 if value.strip().lower() == "true" else 0.0
    return 1.0 if bool(value) else 0.0


def _one_hot_feature(value: object, column: str, vocabulary: tuple[str, ...]) -> dict[str, float]:
    result = dict.fromkeys(_one_hot_columns(column, vocabulary), 0.0)
    if _is_null(value):
        return result
    key = f"{column}__{value}"
    if key in result:
        result[key] = 1.0
    else:
        result[f"{column}__other"] = 1.0
    return result


def hotspot_to_feature_dict(hotspot: dict) -> dict[str, float]:
    """One hotspot (dict of CONTRACT_COLUMNS-ish fields) -> {feature_name: value}."""
    features: dict[str, float] = {}

    bright_ti4, bright_ti4_missing = _numeric_or_sentinel(hotspot.get("bright_ti4"), MISSING_NUMERIC_SENTINEL)
    features["bright_ti4"] = bright_ti4
    features["bright_ti4_missing"] = bright_ti4_missing

    bright_ti5, bright_ti5_missing = _numeric_or_sentinel(hotspot.get("bright_ti5"), MISSING_NUMERIC_SENTINEL)
    features["bright_ti5"] = bright_ti5
    features["bright_ti5_missing"] = bright_ti5_missing

    frp, frp_missing = _numeric_or_sentinel(hotspot.get("frp"), MISSING_NUMERIC_SENTINEL)
    features["frp"] = frp
    features["frp_missing"] = frp_missing

    features["confidence_ordinal"] = _confidence_ordinal(hotspot.get("confidence"))

    for column in _DISTANCE_COLUMNS:
        value, missing = _numeric_or_sentinel(hotspot.get(column), MISSING_DISTANCE_SENTINEL_M)
        features[column] = value
        features[f"{column}_missing"] = missing

    features["recurrence_count"] = _recurrence_feature(hotspot.get("recurrence_count"))
    features["is_anomalous"] = _bool_feature(hotspot.get("is_anomalous"))

    for column, vocabulary in CATEGORICAL_VOCABULARIES.items():
        features.update(_one_hot_feature(hotspot.get(column), column, vocabulary))

    return features


def hotspots_to_feature_matrix(frame: pd.DataFrame) -> pd.DataFrame:
    """Batch version of hotspot_to_feature_dict, columns fixed to FEATURE_COLUMNS order."""
    if frame.empty:
        return pd.DataFrame(columns=FEATURE_COLUMNS)
    rows = frame.apply(lambda row: pd.Series(hotspot_to_feature_dict(row.to_dict())), axis=1)
    return rows.reindex(columns=FEATURE_COLUMNS, fill_value=0.0)
