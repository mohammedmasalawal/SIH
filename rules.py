"""Conservative weak-label rules for FIRMS hotspot classes."""

from __future__ import annotations

import pandas as pd

CLASSES = ("industrial", "gas flare", "agricultural burning", "wildfire", "unknown")


def _number(row: pd.Series, name: str) -> float | None:
    value = row.get(name)
    if value is None or pd.isna(value):
        return None
    return float(value)


FLARE_CAPABLE_FACILITY_TYPES = (
    "oil_gas_power_plant",
    "oil_gas_field",
    "lng_terminal",
    "chemical_plant",
)

HEAT_INDUSTRY_FACILITY_TYPES = (
    "cement_plant",
    "steel_plant",
    "coal_power_plant",
    "coal_mine",
    "chemical_plant",
)


def is_gas_flare(row: pd.Series) -> bool:
    recurrence = _number(row, "recurrence_count") or 0
    distance = _number(row, "dist_to_flare_capable_m")
    return distance is not None and distance <= 1_000 and recurrence >= 5


def is_industrial(row: pd.Series) -> bool:
    """A recurring or spatially-confirmed hotspot at non-flare heavy industry.

    Membership does not depend on FRP magnitude -- a small, steady kiln and a large
    furnace are both legitimate industrial heat sources, and gating on an absolute
    FRP threshold structurally excludes the small ones (see the Sikka cluster).
    daynight is deliberately never used here: it's the independent validation check
    for whether these rules are behaving sanely, not an input to them.
    """
    heat_distance = _number(row, "dist_to_heat_industry_m")
    flare_distance = _number(row, "dist_to_flare_capable_m")
    industrial_distance = _number(row, "dist_to_industrial_m")
    recurrence = _number(row, "recurrence_count") or 0
    near_heat_industry = heat_distance is not None and heat_distance <= 2_000
    near_flare_candidate = flare_distance is not None and flare_distance <= 1_000
    inside_osm_industrial = industrial_distance is not None and industrial_distance <= 1
    near_heat_facility_500m = heat_distance is not None and heat_distance <= 500

    if near_heat_industry and not near_flare_candidate and recurrence >= 2:
        return True
    if recurrence == 1 and (inside_osm_industrial or near_heat_facility_500m):
        return True
    return False


def is_agricultural_burning(row: pd.Series) -> bool:
    landcover = str(row.get("landcover_class", "")).lower()
    distance = _number(row, "dist_to_industrial_m")
    return any(term in landcover for term in ("cropland", "crop", "agriculture", "40")) and (distance is None or distance > 2_000)


def is_wildfire(row: pd.Series) -> bool:
    landcover = str(row.get("landcover_class", "")).lower()
    distance = _number(row, "dist_to_industrial_m")
    return any(term in landcover for term in ("forest", "shrub", "grass", "woodland", "10", "20", "30")) and (distance is None or distance > 2_000)


def classify(row: pd.Series) -> str:
    matches = [
        ("gas flare", is_gas_flare),
        ("industrial", is_industrial),
        ("agricultural burning", is_agricultural_burning),
        ("wildfire", is_wildfire),
    ]
    matched = [label for label, rule in matches if rule(row)]
    return matched[0] if len(matched) == 1 else "unknown"


def apply_rules(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["label"] = result.apply(classify, axis=1)
    result["label_source"] = result["label"].map(lambda label: "rule" if label != "unknown" else "unknown")
    return result
