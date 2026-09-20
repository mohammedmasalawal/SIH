"""Conservative weak-label rules for FIRMS hotspot classes."""

from __future__ import annotations

import pandas as pd

from backend.config import (
    CROPLAND_LANDCOVER_TERMS,
    GAS_FLARE_MAX_DIST_M,
    GAS_FLARE_MIN_RECURRENCE,
    INDUSTRIAL_HEAT_MAX_DIST_M,
    INDUSTRIAL_HEAT_SINGLE_DAY_MAX_DIST_M,
    INDUSTRIAL_MIN_RECURRENCE,
    INDUSTRIAL_POLYGON_TOLERANCE_M,
    NATURAL_FIRE_MIN_DIST_FROM_INDUSTRIAL_M,
    NATURAL_LANDCOVER_TERMS,
)

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


def _osm_industrial_tag(row: pd.Series) -> str:
    value = row.get("osm_industrial_tag")
    if value is None or pd.isna(value):
        return ""
    return str(value).lower()


def is_gas_flare(row: pd.Series) -> bool:
    """Gas flare requires proximity to a flare-capable GEM facility -- no polygon-only path.

    An earlier version also matched recurrence_count >= 2 inside any OSM
    industrial=refinery polygon, with no distance requirement at all. Gold-set
    verification against gold_key.csv's silver labels (see is_industrial's
    docstring below) found that branch was wrong far more often than right: 28 of
    30 silver "gas flare" calls in the verified batch had no visible flare
    structure at the pin, just generic refinery process infrastructure (tank
    farms, warehouses, pipe racks, ETP/substation buildings) inside the same
    refinery polygon. That evidence is why the polygon-only path was removed here
    and folded into is_industrial instead (see below).
    """
    recurrence = _number(row, "recurrence_count") or 0
    distance = _number(row, "dist_to_flare_capable_m")
    return distance is not None and distance <= GAS_FLARE_MAX_DIST_M and recurrence >= GAS_FLARE_MIN_RECURRENCE


def is_industrial(row: pd.Series) -> bool:
    """A recurring or spatially-confirmed hotspot at non-flare heavy industry.

    Membership does not depend on FRP magnitude -- a small, steady kiln and a large
    furnace are both legitimate industrial heat sources, and gating on an absolute
    FRP threshold structurally excludes the small ones (see the Sikka cluster).
    daynight is deliberately never used here: it's the independent validation check
    for whether these rules are behaving sanely, not an input to them.

    Path 3 (recurrence_count >= 2 inside any OSM industrial polygon) used to
    exclude industrial=refinery polygons -- those went to is_gas_flare's
    polygon-only branch instead. Gold-set verification found that branch wrong
    28/30 times (see is_gas_flare's docstring): refinery-polygon recurrence is
    generic refinery activity, not evidence of a flare specifically, so it
    belongs here regardless of the polygon's industrial=* subtype. `not
    is_gas_flare(row)` still guards this path: a hotspot that's *also* within
    GAS_FLARE_MAX_DIST_M of a flare-capable GEM facility keeps that
    distance-confirmed gas-flare call rather than being forced into an
    artificial conflict with the (weaker) polygon-only signal.
    """
    heat_distance = _number(row, "dist_to_heat_industry_m")
    flare_distance = _number(row, "dist_to_flare_capable_m")
    industrial_distance = _number(row, "dist_to_industrial_m")
    recurrence = _number(row, "recurrence_count") or 0
    near_heat_industry = heat_distance is not None and heat_distance <= INDUSTRIAL_HEAT_MAX_DIST_M
    near_flare_candidate = flare_distance is not None and flare_distance <= GAS_FLARE_MAX_DIST_M
    inside_osm_industrial = industrial_distance is not None and industrial_distance <= INDUSTRIAL_POLYGON_TOLERANCE_M
    near_heat_facility_500m = heat_distance is not None and heat_distance <= INDUSTRIAL_HEAT_SINGLE_DAY_MAX_DIST_M

    # near_flare_candidate uses the same threshold as is_gas_flare's distance
    # rule, so this path structurally can never overlap with is_gas_flare.
    if near_heat_industry and not near_flare_candidate and recurrence >= INDUSTRIAL_MIN_RECURRENCE:
        return True
    if recurrence == 1 and (inside_osm_industrial or near_heat_facility_500m):
        return True
    if recurrence >= INDUSTRIAL_MIN_RECURRENCE and bool(_osm_industrial_tag(row)) and not is_gas_flare(row):
        return True
    return False


def is_agricultural_burning(row: pd.Series) -> bool:
    landcover = str(row.get("landcover_class", "")).lower()
    distance = _number(row, "dist_to_industrial_m")
    return any(term in landcover for term in CROPLAND_LANDCOVER_TERMS) and (
        distance is None or distance > NATURAL_FIRE_MIN_DIST_FROM_INDUSTRIAL_M
    )


def is_wildfire(row: pd.Series) -> bool:
    landcover = str(row.get("landcover_class", "")).lower()
    distance = _number(row, "dist_to_industrial_m")
    return any(term in landcover for term in NATURAL_LANDCOVER_TERMS) and (
        distance is None or distance > NATURAL_FIRE_MIN_DIST_FROM_INDUSTRIAL_M
    )


def _matched_labels(row: pd.Series) -> list[str]:
    matches = [
        ("gas flare", is_gas_flare),
        ("industrial", is_industrial),
        ("agricultural burning", is_agricultural_burning),
        ("wildfire", is_wildfire),
    ]
    return [label for label, rule in matches if rule(row)]


def classify(row: pd.Series) -> str:
    matched = _matched_labels(row)
    return matched[0] if len(matched) == 1 else "unknown"


def apply_rules(frame: pd.DataFrame) -> pd.DataFrame:
    """label_source distinguishes *why* a row is unknown: "conflict" means more than
    one rule matched; plain "unknown" means no rule matched at all. is_gas_flare and
    is_industrial are mutually exclusive by construction (see their docstrings), so
    the main remaining source of "conflict" is is_industrial's recurrence==1 path
    overlapping with is_agricultural_burning on a cropland-landcover row with no
    dist_to_industrial_m.
    """
    result = frame.copy()
    matched = result.apply(_matched_labels, axis=1)
    result["label"] = matched.map(lambda labels: labels[0] if len(labels) == 1 else "unknown")
    result["label_source"] = matched.map(
        lambda labels: "rule" if len(labels) == 1 else ("conflict" if len(labels) > 1 else "unknown")
    )
    return result
