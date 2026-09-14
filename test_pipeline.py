import pandas as pd
import pytest

from firms_client import _format_bbox, validate_firms_columns
from rules import classify


def test_bbox_validation():
    assert _format_bbox((69.5, 21.9, 70.5, 22.8)) == "69.5,21.9,70.5,22.8"
    with pytest.raises(ValueError):
        _format_bbox((70, 22, 69, 23))


def test_firms_columns_fail_loudly():
    with pytest.raises(ValueError, match="frp"):
        validate_firms_columns(pd.DataFrame({"latitude": [1]}))


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
