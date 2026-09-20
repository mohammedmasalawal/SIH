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


def test_recurring_hotspot_inside_refinery_polygon_is_industrial_not_gas_flare():
    """Gold-set verification (gold_verified1.csv vs gold_key.csv) found the old
    refinery-polygon-only gas-flare path wrong 28/30 times -- generic refinery
    activity (tank farms, warehouses, pipe racks) with no visible flare stack, not
    evidence of a flare specifically. Recurrence inside a refinery polygon with no
    nearby flare-capable GEM facility must resolve to industrial now, never gas
    flare -- see is_gas_flare's and is_industrial's docstrings."""
    row = pd.Series({"frp": 20, "dist_to_industrial_m": 0, "osm_industrial_tag": "refinery", "dist_to_heat_industry_m": 5000, "dist_to_flare_capable_m": 20_000, "recurrence_count": 2})
    assert classify(row) == "industrial"


def test_recurring_hotspot_inside_other_industrial_polygon_is_industrial():
    row = pd.Series({"frp": 20, "dist_to_industrial_m": 0, "osm_industrial_tag": "yes", "dist_to_heat_industry_m": 5000, "dist_to_flare_capable_m": 20_000, "recurrence_count": 2})
    assert classify(row) == "industrial"


def test_missing_osm_industrial_tag_does_not_fake_a_polygon_match():
    """A NaN osm_industrial_tag must not be treated as a truthy tag value (regression: `NaN or ""` is NaN, not "")."""
    row = pd.Series({"frp": 20, "dist_to_industrial_m": 5000, "osm_industrial_tag": float("nan"), "dist_to_heat_industry_m": 40_000, "dist_to_flare_capable_m": 40_000, "recurrence_count": 3, "landcover_class": "20"})
    assert classify(row) == "wildfire"


def test_refinery_polygon_and_gem_heat_distance_no_longer_conflict():
    """Before the gold-set fix, a row inside an OSM refinery polygon that also
    independently qualified for is_industrial's GEM-distance path produced an
    unresolved conflict (is_gas_flare's old refinery-only branch vs is_industrial).
    is_gas_flare no longer has a polygon-only path, so is_gas_flare and
    is_industrial are mutually exclusive by construction and this now cleanly
    resolves to industrial instead of "unknown"/"conflict"."""
    row = pd.Series({
        "dist_to_industrial_m": 0, "osm_industrial_tag": "refinery",
        "dist_to_heat_industry_m": 100, "dist_to_flare_capable_m": 20_000,
        "recurrence_count": 2,
    })
    assert classify(row) == "industrial"


def test_high_recurrence_refinery_polygon_far_from_any_gem_facility_is_industrial():
    """The core of the fix: even very high recurrence inside a refinery polygon,
    with no GEM flare-capable OR heat-industry facility anywhere close, resolves
    via the polygon path to industrial -- never gas flare, regardless of how
    persistent the hotspot is."""
    row = pd.Series({
        "dist_to_industrial_m": 0, "osm_industrial_tag": "refinery",
        "dist_to_heat_industry_m": 50_000, "dist_to_flare_capable_m": 50_000,
        "recurrence_count": 80,
    })
    assert classify(row) == "industrial"
