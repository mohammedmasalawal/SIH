from __future__ import annotations

import pandas as pd

from backend.classification.rules import classify, is_agricultural_burning, is_industrial, is_wildfire
from backend.config import GAS_FLARE_MAX_DIST_M, GAS_FLARE_MIN_RECURRENCE


def test_ambiguous_context_is_unknown():
    row = pd.Series({"frp": 50, "dist_to_industrial_m": 1000, "dist_to_flare_capable_m": 500, "dist_to_heat_industry_m": 500, "recurrence_count": 2, "landcover_class": "cropland"})
    assert classify(row) == "unknown"


def test_persistent_flare_is_gas_flare():
    row = pd.Series({"frp": 50, "dist_to_industrial_m": 100, "dist_to_flare_capable_m": 50, "dist_to_heat_industry_m": 50, "recurrence_count": 8})
    assert classify(row) == "gas flare"


def test_single_day_near_heat_facility_is_industrial():
    row = pd.Series({"frp": 40, "dist_to_industrial_m": 100, "dist_to_heat_industry_m": 300, "dist_to_flare_capable_m": 20_000, "recurrence_count": 1})
    assert classify(row) == "industrial"


def test_recurring_low_frp_near_heat_industry_is_industrial():
    """Membership does not depend on FRP magnitude -- a weak, steady kiln still counts."""
    row = pd.Series({"frp": 0.6, "dist_to_industrial_m": 50, "dist_to_heat_industry_m": 100, "dist_to_flare_capable_m": 20_000, "recurrence_count": 3})
    assert classify(row) == "industrial"


def test_single_day_far_from_everything_is_unknown():
    row = pd.Series({"frp": 30, "dist_to_industrial_m": 5000, "dist_to_heat_industry_m": 5000, "dist_to_flare_capable_m": 20_000, "recurrence_count": 1, "landcover_class": "50"})
    assert classify(row) == "unknown"


def test_agricultural_burning_true_case():
    row = pd.Series({"landcover_class": "cropland", "dist_to_industrial_m": 5_000})
    assert is_agricultural_burning(row) is True


def test_wildfire_true_case():
    row = pd.Series({"landcover_class": "forest", "dist_to_industrial_m": 5_000})
    assert is_wildfire(row) is True


def test_double_match_resolves_to_unknown():
    """industrial (recurrence==1 + near_heat_facility_500m) and agricultural burning
    (cropland landcover + null dist_to_industrial_m) can both fire on the same row --
    classify's len(matched) == 1 guard must resolve that ambiguity to 'unknown'
    rather than picking one arbitrarily."""
    row = pd.Series({
        "dist_to_industrial_m": None,
        "dist_to_heat_industry_m": 100,
        "dist_to_flare_capable_m": 20_000,
        "recurrence_count": 1,
        "landcover_class": "cropland",
    })
    assert is_industrial(row) is True
    assert is_agricultural_burning(row) is True
    assert classify(row) == "unknown"


def test_gas_flare_threshold_boundary():
    """Guards the config migration: rules.py must use backend.config's exact values."""
    at_threshold = pd.Series({
        "dist_to_flare_capable_m": GAS_FLARE_MAX_DIST_M,
        "recurrence_count": GAS_FLARE_MIN_RECURRENCE,
        "dist_to_industrial_m": 100, "dist_to_heat_industry_m": 20_000,
    })
    assert classify(at_threshold) == "gas flare"

    just_over = pd.Series({
        "dist_to_flare_capable_m": GAS_FLARE_MAX_DIST_M + 1,
        "recurrence_count": GAS_FLARE_MIN_RECURRENCE,
        "dist_to_industrial_m": 100, "dist_to_heat_industry_m": 20_000,
    })
    assert classify(just_over) == "unknown"
