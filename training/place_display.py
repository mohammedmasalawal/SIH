"""Display-only place names for the exported dashboard (nothing here is read by ingestion).

Two exported structures, both built from a Places lookup (backend/places.py):

  data/alert_places.json.gz   {"places": [[name, state], ...],
                               "alerts": {alert_id: [place, km, compass, level, title, osm_tag]}}
  data/facilities.json.gz     gains near_places / near_place / near_km / near_dir columns

`km` is whole kilometres, `compass` the 8-point sector 0..7 (N, NE, ...) or -1 within 0.5 km,
`level` 0 = city/town, 1 = village. alerts_history.csv itself is copied byte for byte and its
schema never changes: titles and the "near" text live only in alert_places.json.gz.

Alert titles. An alert about a facility with no name used to read as a raw OSM tag or a generic
"Industrial site · <State>". With a place lookup it reads "Industrial site near <place>"
(or "<Facility kind> near <place>" for an unnamed critical facility). The raw OSM tag ("OSM
industrial=yes") is kept as its own detail row in the alert popup.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

from backend.places import Places, format_near

# a title that still shows a raw OSM tag: "OSM industrial=yes", "OSM industrial area", "OSM power=plant"
RAW_OSM_TITLE = re.compile(r"\bOSM\s+\S*=\S*|\bOSM\s+industrial\b", re.IGNORECASE)
GENERIC_SITE = re.compile(r"^(industrial site\b|osm industrial\b)", re.IGNORECASE)


def _text(value) -> str:
    return "" if value is None or (isinstance(value, float) and np.isnan(value)) or str(value) == "nan" else str(value)


def alert_title(alert_type: str, facility_type, facility_name, place_name: str) -> str | None:
    """The display title that replaces the alert's own, or None to keep it.

    industrial_anomaly at a generic industrial site (its facility_type starts with "Industrial
    site" or "OSM industrial") -> "Industrial site near <place>".
    new_activity_at_critical_site at an unnamed facility -> "<Kind> near <place>", e.g.
    "Power plant (coal) near Kudgi". Everything else keeps its title."""
    facility = _text(facility_type).replace("_", " ").strip()
    if alert_type == "industrial_anomaly" and GENERIC_SITE.match(facility):
        return f"Industrial site near {place_name}"
    name = _text(facility_name).strip()
    if alert_type == "new_activity_at_critical_site" and name in ("", "unnamed"):
        kind = facility.split(" · ")[0].strip()
        if kind:
            return f"{kind[0].upper()}{kind[1:]} near {place_name}"
    return None


def _place_table(matches: pd.DataFrame) -> tuple[list[list], np.ndarray]:
    """De-duplicated [[name, state], ...] and each match's index into it."""
    keys = list(zip(matches["name"], matches["state"].where(matches["state"].notna(), None)))
    table, index, ids = [], {}, []
    for key in keys:
        if key not in index:
            index[key] = len(table)
            table.append([key[0], key[1]])
        ids.append(index[key])
    return table, np.asarray(ids, dtype=int)


def _km_and_dir(matches: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    km = (matches["distance_km"].to_numpy() + 0.5).astype(int)
    direction = np.where(matches["distance_km"].to_numpy() < 0.5, -1, matches["compass"].to_numpy())
    return km, direction


def build_alert_places(alerts: pd.DataFrame, places: Places, raw_tags: dict | None = None) -> dict:
    """alert_places.json for every alert in `alerts` (the alerts_history.csv frame). raw_tags maps
    alert_id -> the detection's OSM industrial tag (for the popup's detail row)."""
    if alerts.empty:
        return {"places": [], "alerts": {}}
    matches = places.nearest(alerts["latitude"].astype(float), alerts["longitude"].astype(float))
    table, ids = _place_table(matches)
    km, direction = _km_and_dir(matches)
    out = {}
    for i, row in enumerate(alerts.itertuples(index=False)):
        name = table[ids[i]][0]
        title = alert_title(row.alert_type, getattr(row, "facility_type", None), getattr(row, "facility_name", None), name)
        raw = (raw_tags or {}).get(row.alert_id) if title else None
        out[row.alert_id] = [int(ids[i]), int(km[i]), int(direction[i]), 0 if matches["level"].iat[i] == "town" else 1,
                             title, raw]
    return {"places": table, "alerts": out}


def build_facility_places(latitudes, longitudes, places: Places) -> dict:
    """Columns to merge into facilities.json (parallel to the facility rows)."""
    matches = places.nearest(latitudes, longitudes)
    table, ids = _place_table(matches)
    km, direction = _km_and_dir(matches)
    return {"near_places": table, "near_place": ids.tolist(), "near_km": km.tolist(), "near_dir": direction.tolist(),
            "near_level": (matches["level"] != "town").astype(int).tolist()}


def near_text(table: list, place: int, km: int, direction: int) -> str:
    """The string the page shows for an encoded match (must equal alerts-panel.js nearText)."""
    name, state = table[place]
    return format_near(name, state, float(km), None if direction < 0 else direction)
