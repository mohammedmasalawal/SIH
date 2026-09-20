from __future__ import annotations

import pandas as pd
import pytest

from backend.classification.features import (
    CATEGORICAL_VOCABULARIES,
    FEATURE_COLUMNS,
    hotspot_to_feature_dict,
    hotspots_to_feature_matrix,
)
from backend.config import CONFIDENCE_DEFAULT_ORDINAL, MISSING_DISTANCE_SENTINEL_M, MISSING_NUMERIC_SENTINEL


def test_feature_columns_has_no_duplicates():
    assert len(FEATURE_COLUMNS) == len(set(FEATURE_COLUMNS))


def test_known_category_sets_correct_one_hot_bit(sample_hotspot_dict):
    features = hotspot_to_feature_dict(sample_hotspot_dict)
    assert features["daynight__D"] == 1.0
    assert features["daynight__N"] == 0.0


def test_unseen_category_routes_to_other(sample_hotspot_dict):
    hotspot = dict(sample_hotspot_dict, landcover_class="not-a-real-code")
    features = hotspot_to_feature_dict(hotspot)
    assert features["landcover_class__other"] == 1.0
    for value in CATEGORICAL_VOCABULARIES["landcover_class"]:
        assert features[f"landcover_class__{value}"] == 0.0


def test_null_categorical_zeroes_whole_block(sample_hotspot_dict):
    hotspot = dict(sample_hotspot_dict, nearest_flare_facility_type=None)
    features = hotspot_to_feature_dict(hotspot)
    block = [name for name in FEATURE_COLUMNS if name.startswith("nearest_flare_facility_type__")]
    assert block  # sanity: the block actually exists
    assert all(features[name] == 0.0 for name in block)


def test_null_numeric_distance_uses_sentinel_and_missing_flag(sample_hotspot_dict):
    hotspot = dict(sample_hotspot_dict, dist_to_flare_capable_m=None)
    features = hotspot_to_feature_dict(hotspot)
    assert features["dist_to_flare_capable_m"] == MISSING_DISTANCE_SENTINEL_M
    assert features["dist_to_flare_capable_m_missing"] == 1.0


def test_null_frp_uses_numeric_sentinel(sample_hotspot_dict):
    hotspot = dict(sample_hotspot_dict, frp=None)
    features = hotspot_to_feature_dict(hotspot)
    assert features["frp"] == MISSING_NUMERIC_SENTINEL
    assert features["frp_missing"] == 1.0


def test_confidence_ordinal_unseen_string_falls_back_to_default(sample_hotspot_dict):
    hotspot = dict(sample_hotspot_dict, confidence="totally-unknown")
    features = hotspot_to_feature_dict(hotspot)
    assert features["confidence_ordinal"] == float(CONFIDENCE_DEFAULT_ORDINAL)


def test_train_serve_parity(sample_hotspot_dict):
    single = hotspot_to_feature_dict(sample_hotspot_dict)
    batch = hotspots_to_feature_matrix(pd.DataFrame([sample_hotspot_dict]))
    for column in FEATURE_COLUMNS:
        assert single[column] == pytest.approx(batch.iloc[0][column])
