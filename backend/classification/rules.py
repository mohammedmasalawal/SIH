"""Conservative weak-label rules for FIRMS hotspot classes."""

from __future__ import annotations

import pandas as pd

from backend.config import (
    CROPLAND_LANDCOVER_TERMS,
    GAS_FLARE_MAX_DIST_M,
    GAS_FLARE_MIN_RECURRENCE,
    GAS_FLARE_REFINERY_MIN_RECURRENCE,
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
    recurrence = _number(row, "recurrence_count") or 0
    distance = _number(row, "dist_to_flare_capable_m")
    distance_rule = distance is not None and distance <= GAS_FLARE_MAX_DIST_M and recurrence >= GAS_FLARE_MIN_RECURRENCE
    refinery_rule = recurrence >= GAS_FLARE_REFINERY_MIN_RECURRENCE and _osm_industrial_tag(row) == "refinery"
    return distance_rule or refinery_rule


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
    near_heat_industry = heat_distance is not None and heat_distance <= INDUSTRIAL_HEAT_MAX_DIST_M
    near_flare_candidate = flare_distance is not None and flare_distance <= GAS_FLARE_MAX_DIST_M
    inside_osm_industrial = industrial_distance is not None and industrial_distance <= INDUSTRIAL_POLYGON_TOLERANCE_M
    near_heat_facility_500m = heat_distance is not None and heat_distance <= INDUSTRIAL_HEAT_SINGLE_DAY_MAX_DIST_M

    # Paths 1 and 2 predate is_gas_flare's OSM-refinery branch and are deliberately
    # left unguarded against it (unlike path 3 just below). A GEM-distance hotspot
    # that's also inside an OSM refinery polygon with recurrence_count >= 2 now
    # satisfies both this path and is_gas_flare's refinery_rule, so classify()
    # reports it as an unresolved conflict (label "unknown", label_source
    # "conflict") rather than either rule silently winning. Do NOT "fix" this by
    # adding `and not is_gas_flare(row)` here: these rows are genuinely ambiguous
    # (general refinery heat vs. an actual flare) and are meant to fall through to
    # manual/gold-set review, not be guessed at.
    if near_heat_industry and not near_flare_candidate and recurrence >= INDUSTRIAL_MIN_RECURRENCE:
        return True
    if recurrence == 1 and (inside_osm_industrial or near_heat_facility_500m):
        return True
    osm_tag = _osm_industrial_tag(row)
    inside_other_industrial_polygon = bool(osm_tag) and osm_tag != "refinery"
    if recurrence >= INDUSTRIAL_MIN_RECURRENCE and inside_other_industrial_polygon and not is_gas_flare(row):
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
    one rule matched (see is_industrial's paths 1/2 vs. is_gas_flare's refinery_rule
    for the main source of these); plain "unknown" means no rule matched at all.
    """
    result = frame.copy()
    matched = result.apply(_matched_labels, axis=1)
    result["label"] = matched.map(lambda labels: labels[0] if len(labels) == 1 else "unknown")
    result["label_source"] = matched.map(
        lambda labels: "rule" if len(labels) == 1 else ("conflict" if len(labels) > 1 else "unknown")
    )
    return result
