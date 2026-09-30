"""Nearest-place names for a point: display-only ("Near Angul, Odisha (12 km NE)").

The lookup reads data/osm/places_india.parquet (training/extract_places.py: OSM place=city,
town and village nodes). It is used at export time only (training/export_site.py); ingestion,
labels, alerts_history.csv and the packed points never see it.

RULE (fixed, not tuned):
  1. Distances are measured in NATIONAL_PROJECTED_CRS (EPSG:7755), nearest-neighbour through a
     spatial index (shapely STRtree), place point to query point.
  2. Cities and towns are preferred: if the nearest city-or-town (the two types are pooled, no
     preference between them) is within PLACE_VILLAGE_FALLBACK_KM (15 km), that is the answer,
     level "town".
  3. Only when no city or town lies within 15 km is the nearest village used (level "village",
     at whatever distance it is). If the file holds no village at all, the nearest city or
     town is used at any distance.
  4. The bearing is the geodesic initial bearing from the place to the point, reported as the
     8-point compass sector it falls in (N, NE, E, SE, S, SW, W, NW; 45 degrees each, centred
     on the compass point, so "E" is 67.5 to 112.5 degrees).
  5. Distances are shown in whole kilometres. At under 0.5 km the point is "at" the place and
     no direction is shown.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import shapely
from pyproj import Geod, Transformer
from shapely import STRtree

from backend.config import NATIONAL_PROJECTED_CRS, PLACE_VILLAGE_FALLBACK_KM

COMPASS = ("N", "NE", "E", "SE", "S", "SW", "W", "NW")
TOWN_TYPES = ("city", "town")
_GEOD = Geod(ellps="WGS84")
_TO_PROJECTED = Transformer.from_crs("EPSG:4326", NATIONAL_PROJECTED_CRS, always_xy=True)


def compass_index(bearing_deg) -> np.ndarray:
    """0..7 (N, NE, ... NW) for bearings in degrees, 45-degree sectors centred on each point."""
    return (np.floor((np.asarray(bearing_deg, dtype="float64") % 360 + 22.5) / 45).astype(int)) % 8


def format_near(name: str, state: str | None, distance_km: float, compass: int | None) -> str:
    """"Near Angul, Odisha (12 km NE)"; "Near Angul, Odisha (0 km)" at the place itself."""
    where = f"{name}, {state}" if state else name
    km = int(distance_km + 0.5)
    if distance_km < 0.5 or compass is None or compass < 0:
        return f"Near {where} (0 km)"
    return f"Near {where} ({km} km {COMPASS[compass]})"


@dataclass
class Places:
    frame: pd.DataFrame  # name, state, place, lon, lat (+ osm_id, population)
    _towns: np.ndarray  # row positions of city/town
    _villages: np.ndarray
    _town_tree: STRtree | None
    _village_tree: STRtree | None
    _x: np.ndarray
    _y: np.ndarray

    @classmethod
    def from_frame(cls, frame: pd.DataFrame) -> "Places":
        frame = frame.reset_index(drop=True)
        x, y = _TO_PROJECTED.transform(frame["lon"].to_numpy("float64"), frame["lat"].to_numpy("float64"))
        is_town = frame["place"].isin(TOWN_TYPES).to_numpy()
        towns, villages = np.flatnonzero(is_town), np.flatnonzero(~is_town)
        tree = lambda rows: STRtree(shapely.points(x[rows], y[rows])) if len(rows) else None
        return cls(frame, towns, villages, tree(towns), tree(villages), x, y)

    def __len__(self) -> int:
        return len(self.frame)

    def nearest(self, lats, lons) -> pd.DataFrame:
        """One row per query point: place_id (row in the places file), name, state, place, level
        ("town" or "village"), distance_km, bearing (place -> point, degrees), compass (0..7), text."""
        lats, lons = np.asarray(lats, dtype="float64"), np.asarray(lons, dtype="float64")
        if not len(self):
            raise ValueError("no places loaded")
        qx, qy = _TO_PROJECTED.transform(lons, lats)
        query = shapely.points(qx, qy)
        n = len(query)
        town_id, town_dist = self._query(self._town_tree, self._towns, query)
        village_id, village_dist = self._query(self._village_tree, self._villages, query)
        use_town = town_dist <= PLACE_VILLAGE_FALLBACK_KM * 1000.0  # NaN compares False
        if self._village_tree is None:
            use_town = np.ones(n, dtype=bool)
        place_id = np.where(use_town, town_id, village_id)
        dist_m = np.where(use_town, town_dist, village_dist)
        rows = self.frame.iloc[place_id]
        bearing = _GEOD.inv(rows["lon"].to_numpy(), rows["lat"].to_numpy(), lons, lats)[0] % 360
        compass = compass_index(bearing)
        out = pd.DataFrame({
            "place_id": place_id, "name": rows["name"].to_numpy(),
            "state": rows["state"].where(rows["state"].notna(), None).to_numpy(),
            "place": rows["place"].to_numpy(), "level": np.where(use_town, "town", "village"),
            "distance_km": dist_m / 1000.0, "bearing": bearing, "compass": compass,
        })
        out["text"] = [format_near(nm, st, d, c) for nm, st, d, c in zip(out["name"], out["state"], out["distance_km"], out["compass"])]
        return out

    @staticmethod
    def _query(tree: STRtree | None, rows: np.ndarray, query) -> tuple[np.ndarray, np.ndarray]:
        if tree is None:
            return np.full(len(query), -1), np.full(len(query), np.inf)
        (q_idx, t_idx), dist = tree.query_nearest(query, return_distance=True, all_matches=False)
        ids = np.full(len(query), -1)
        dists = np.full(len(query), np.inf)
        ids[q_idx], dists[q_idx] = rows[t_idx], dist
        return ids, dists


def load_places(path: str | Path) -> Places:
    frame = pd.read_parquet(path)
    return Places.from_frame(frame)
