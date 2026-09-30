"""Report on the display-only place names: how many alerts and known facilities got a town-level
versus a village-level name, and the farthest matches (where a bad name is most likely).

    python -m training.place_report [--top 10]
"""

from __future__ import annotations

import argparse

import pandas as pd

from backend.config import (
    ALERTS_HISTORY_PATH,
    GEM_FACILITIES_PATH,
    NATIONAL_INDUSTRIAL_CONTEXT_PATH,
    PLACES_PATH,
)
from backend.places import load_places
from training.export_site import _facility_frame


def matches(places_path=PLACES_PATH, alerts_path=ALERTS_HISTORY_PATH, facilities_path=GEM_FACILITIES_PATH,
            industrial_path=NATIONAL_INDUSTRIAL_CONTEXT_PATH) -> pd.DataFrame:
    places = load_places(places_path)
    alerts = pd.read_csv(alerts_path, dtype={"acq_time": str})
    facilities = _facility_frame(facilities_path, industrial_path)
    parts = []
    for source, lat, lon, label in (
        ("alert", alerts["latitude"], alerts["longitude"], alerts["alert_type"] + " " + alerts["alert_id"].str.split("|").str[-1]),
        ("facility", facilities["lat"], facilities["lon"], facilities["source"] + " " + facilities["kind"].astype(str) + " " + facilities["name"].fillna("(unnamed)").astype(str)),
    ):
        found = places.nearest(lat.to_numpy(), lon.to_numpy())
        found.insert(0, "what", label.to_numpy())
        found.insert(0, "source", source)
        found["lat"], found["lon"] = lat.to_numpy(), lon.to_numpy()
        parts.append(found)
    return pd.concat(parts, ignore_index=True)


def report(top: int = 10) -> str:
    found = matches()
    lines = []
    for source in ("alert", "facility"):
        part = found[found["source"] == source]
        towns = int((part["level"] == "town").sum())
        lines.append(f"{source}s: {len(part):,}  town-level {towns:,} ({towns / max(len(part), 1):.1%})  "
                     f"village-level {len(part) - towns:,}")
    lines.append(f"\n{top} farthest matches:")
    farthest = found.sort_values("distance_km", ascending=False).head(top)
    for row in farthest.itertuples():
        lines.append(f"  {row.distance_km:7.1f} km  {row.level:<7}  {row.source:<8} {row.what[:52]:<52} "
                     f"({row.lat:.4f}, {row.lon:.4f})  ->  {row.text}")
    return "\n".join(lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--top", type=int, default=10)
    print(report(parser.parse_args().top))
