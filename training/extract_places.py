"""Extract OSM place=city / town / village nodes from the India .osm.pbf into a new parquet.

    python -m training.extract_places [--pbf data/osm/india-latest.osm.pbf] [--out data/osm/places_india.parquet]

Display-only input for the nearest-place names shown by the dashboard (backend/places.py). It
reads the .pbf and writes ONE new file; the existing OSM industrial-context cache is never
opened or modified.

Name: `name:en` if the node has one, else `name` if it is in Latin script, else the node is
skipped. Latin script = every letter is a Latin-script letter (digits, spaces and punctuation
are ignored), so "Angul" and "Bhubaneswar" pass and "अंगुल" or a mixed "Angul अंगुल" don't.
population is kept as an integer when the tag parses, otherwise null. `state` is the Indian
state/UT polygon (backend.config.INDIA_BOUNDARY_PATH) the place sits in, null outside India.
"""

from __future__ import annotations

import argparse
import time
import unicodedata
from pathlib import Path

import numpy as np
import osmium
import pandas as pd

from backend.config import INDIA_BOUNDARY_PATH, PLACES_PATH, ROOT_DIR

PLACE_TYPES = ("city", "town", "village")
DEFAULT_PBF = ROOT_DIR / "data" / "osm" / "india-latest.osm.pbf"
COLUMNS = ["osm_id", "place", "name", "name_source", "population", "lon", "lat", "state"]


def is_latin(text: str) -> bool:
    """True if `text` has at least one letter and every letter is Latin script."""
    letters = [c for c in text if c.isalpha()]
    return bool(letters) and all(unicodedata.name(c, "").startswith("LATIN") for c in letters)


def place_name(tags) -> tuple[str, str] | None:
    """(name, source tag) by the rule in the module docstring, or None to skip the node."""
    english = (tags.get("name:en") or "").strip()
    if english:
        return english, "name:en"
    plain = (tags.get("name") or "").strip()
    if plain and is_latin(plain):
        return plain, "name"
    return None


def parse_population(value: str | None) -> int | None:
    if not value:
        return None
    digits = value.replace(",", "").replace(" ", "").split(";")[0].split(".")[0]
    return int(digits) if digits.isdigit() else None


class _PlaceHandler(osmium.SimpleHandler):
    def __init__(self) -> None:
        super().__init__()
        self.rows: list[dict] = []
        self.seen = {t: 0 for t in PLACE_TYPES}  # tagged nodes per type, named or not

    def node(self, n) -> None:
        kind = n.tags.get("place")
        if kind not in PLACE_TYPES or not n.location.valid():
            return
        self.seen[kind] += 1
        named = place_name(n.tags)
        if named is None:
            return
        self.rows.append({
            "osm_id": n.id, "place": kind, "name": named[0], "name_source": named[1],
            "population": parse_population(n.tags.get("population")),
            "lon": n.location.lon, "lat": n.location.lat,
        })


def extract_places(pbf: str | Path, out: str | Path, boundary_path: str | Path = INDIA_BOUNDARY_PATH) -> dict:
    """Write `out` (parquet) and return {counts, skipped, seconds}. Refuses to overwrite anything
    but a previous places file (never the industrial-context cache)."""
    out = Path(out)
    if out.name != "places_india.parquet" and out.exists():
        raise FileExistsError(f"{out} exists and is not a places file")
    started = time.perf_counter()
    handler = _PlaceHandler()
    handler.apply_file(str(pbf))  # nodes only: no location index needed
    frame = pd.DataFrame(handler.rows, columns=COLUMNS)
    frame["population"] = frame["population"].astype("Int64")
    frame["state"] = _states(frame, boundary_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(out, index=False)
    kept = frame["place"].value_counts().to_dict()
    return {
        "kept": {t: int(kept.get(t, 0)) for t in PLACE_TYPES},
        "tagged": handler.seen,
        "skipped_no_latin_name": {t: handler.seen[t] - int(kept.get(t, 0)) for t in PLACE_TYPES},
        "with_population": int(frame["population"].notna().sum()),
        "name_from_name_en": int((frame["name_source"] == "name:en").sum()),
        "outside_india": int(frame["state"].isna().sum()),
        "seconds": round(time.perf_counter() - started, 1),
    }


def _states(frame: pd.DataFrame, boundary_path) -> pd.Series:
    import geopandas as gpd

    if not len(frame) or not Path(boundary_path).exists():
        return pd.Series([None] * len(frame), dtype="object")
    states = gpd.read_parquet(boundary_path)[["name", "geometry"]]
    points = gpd.GeoDataFrame(geometry=gpd.points_from_xy(frame["lon"], frame["lat"]), index=frame.index, crs="EPSG:4326")
    joined = gpd.sjoin(points, states, how="left", predicate="within")
    joined = joined[~joined.index.duplicated(keep="first")]
    return joined["name"].reindex(frame.index).astype("object")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pbf", type=Path, default=DEFAULT_PBF)
    parser.add_argument("--out", type=Path, default=PLACES_PATH)
    args = parser.parse_args()
    report = extract_places(args.pbf, args.out)
    print(f"{args.out}: {sum(report['kept'].values()):,} places in {report['seconds']} s")
    for kind in PLACE_TYPES:
        print(f"  {kind:<8}{report['kept'][kind]:>9,} kept of {report['tagged'][kind]:>9,} tagged "
              f"({report['skipped_no_latin_name'][kind]:,} skipped: no name:en and no Latin name)")
    print(f"  with population {report['with_population']:,}; name from name:en {report['name_from_name_en']:,}; "
          f"outside the Indian state polygons {report['outside_india']:,}")
