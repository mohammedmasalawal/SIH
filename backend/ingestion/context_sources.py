"""Load and cache spatial context used by the hotspot label builder.

Phase 1 national scale-up: the live Overpass query (removed) doesn't scale past a
single city/bbox -- Overpass rate-limits and times out well before a national-scale
query completes. `extract_osm_industrial_context` reads a pre-downloaded Geofabrik
.osm.pbf extract instead (e.g. https://download.geofabrik.de/asia/india-latest.osm.pbf),
once, and caches the result as GeoParquet. The bbox parameter still exists for
regional/testing runs (filter a national .pbf down to one city), it just no longer
drives a live network query.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import geopandas as gpd
import osmium
import pandas as pd
from shapely import wkb as shapely_wkb
from shapely.geometry import Polygon, box

# Tag key -> set of accepted values, or None to accept any value for that key
# (i.e. `industrial=*`). A node/way is extracted if it matches ANY one of these.
OSM_TARGET_TAGS: dict[str, set[str] | None] = {
    "landuse": {"industrial"},
    "industrial": None,
    "man_made": {"works", "kiln", "flare"},
    "power": {"plant"},
}


def _matched_tag_key(tags) -> str | None:
    """The first OSM_TARGET_TAGS key this element's tags satisfy, or None."""
    for key, allowed_values in OSM_TARGET_TAGS.items():
        value = tags.get(key)
        if value is None:
            continue
        if allowed_values is None or value in allowed_values:
            return key
    return None


class _IndustrialContextHandler(osmium.SimpleHandler):
    """Collects nodes/ways matching OSM_TARGET_TAGS into plain dict rows.

    Ways are represented as a Polygon when closed (way.is_closed, e.g. a
    landuse=industrial area) and left as a LineString otherwise (e.g. a mapped
    pipeline or a way that isn't a closed ring) -- matching the closed-ring
    heuristic the old Overpass-based extractor used, so downstream code
    (build_labels.py's polygon containment / nearest-facility joins) sees the
    same geometry shapes it always has. Relations (multipolygon-assembled
    industrial complexes spanning several ways) are not resolved in this phase --
    same limitation the Overpass `way`/`node`-only query already had.
    """

    def __init__(self, wkb_factory: "osmium.geom.WKBFactory", bbox: tuple[float, float, float, float] | None):
        super().__init__()
        self._wkb_factory = wkb_factory
        self._bbox = box(*bbox) if bbox is not None else None
        self.rows: list[dict] = []

    def _append(self, osm_type: str, osm_id: int, tags, matched_key: str, wkb_hex: str) -> None:
        geometry = shapely_wkb.loads(wkb_hex, hex=True)
        if self._bbox is not None and not geometry.intersects(self._bbox):
            return
        row = {"osm_type": osm_type, "osm_id": osm_id, "matched_tag": matched_key, "geometry": geometry}
        for tag in tags:
            row[tag.k] = tag.v
        self.rows.append(row)

    def node(self, n) -> None:
        if not n.tags or not n.location.valid():
            return
        matched_key = _matched_tag_key(n.tags)
        if matched_key is None:
            return
        self._append("node", n.id, n.tags, matched_key, self._wkb_factory.create_point(n.location))

    def way(self, w) -> None:
        if not w.tags:
            return
        matched_key = _matched_tag_key(w.tags)
        if matched_key is None:
            return
        try:
            wkb_hex = self._wkb_factory.create_linestring(w)
        except (RuntimeError, osmium.InvalidLocationError):
            # a node this way references fell outside the extract / location cache
            return
        if w.is_closed:
            geometry = shapely_wkb.loads(wkb_hex, hex=True)
            if len(geometry.coords) >= 4:
                wkb_hex = Polygon(geometry.coords).wkb_hex
        self._append("way", w.id, w.tags, matched_key, wkb_hex)


def extract_osm_industrial_context(
    pbf_path: str | Path,
    cache_path: str | Path,
    bbox: Iterable[float] | None = None,
) -> gpd.GeoDataFrame:
    """Extract industrial/power infrastructure from a local .osm.pbf, cached as GeoParquet.

    `bbox` (west, south, east, north), if given, filters to elements whose geometry
    intersects it -- the extract still parses the whole .pbf once (osmium streams
    it; there's no cheap way to skip regions of a .pbf without a spatial index
    that Geofabrik doesn't ship), so pass it mainly to produce a smaller cached
    file for regional testing, not to speed up the parse itself. Re-running with a
    different bbox against the same cache_path returns the stale cache -- delete
    the Parquet file (or use a bbox-specific cache_path) to re-extract.
    """
    destination = Path(cache_path)
    if destination.exists():
        return gpd.read_parquet(destination)

    bbox_tuple = tuple(float(v) for v in bbox) if bbox is not None else None
    wkb_factory = osmium.geom.WKBFactory()
    handler = _IndustrialContextHandler(wkb_factory, bbox_tuple)
    handler.apply_file(str(pbf_path), locations=True, idx="flex_mem")

    frame = gpd.GeoDataFrame(handler.rows, geometry="geometry", crs="EPSG:4326")
    destination.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(destination)
    return frame


def tag_count_report(frame: pd.DataFrame) -> pd.DataFrame:
    """Feature counts by exact tag=value, across all of OSM_TARGET_TAGS's keys.

    One row per (key, value) actually present in the extract, e.g. "landuse=industrial",
    "industrial=oil", "man_made=works", "power=plant" -- not just counts per matched_tag
    bucket, since `industrial=*`/`man_made=*` cover many distinct values and knowing
    which ones actually showed up (and how often) is the point of the report.
    """
    rows = []
    for key in OSM_TARGET_TAGS:
        if key not in frame.columns:
            continue
        counts = frame.loc[frame[key].notna(), key].value_counts()
        for value, count in counts.items():
            rows.append({"tag": f"{key}={value}", "count": int(count)})
    report = pd.DataFrame(rows, columns=["tag", "count"])
    return report.sort_values("count", ascending=False).reset_index(drop=True)


def load_gem_facilities(source: str | Path, cache_path: str | Path) -> gpd.GeoDataFrame:
    """Load a GEM CSV/URL once, normalising latitude/longitude to GeoJSON."""
    destination = Path(cache_path)
    if destination.exists():
        return gpd.read_file(destination)

    frame = pd.read_csv(source)
    columns = {column.lower().strip(): column for column in frame.columns}
    latitude = columns.get("latitude", columns.get("lat"))
    longitude = columns.get("longitude", columns.get("lon", columns.get("lng")))
    if not latitude or not longitude:
        raise ValueError("GEM source needs latitude/longitude columns")
    frame = frame.dropna(subset=[latitude, longitude]).copy()
    frame["facility_type"] = frame.get("facility_type", frame.get("type", "industrial"))
    facilities = gpd.GeoDataFrame(
        frame,
        geometry=gpd.points_from_xy(frame[longitude], frame[latitude]),
        crs="EPSG:4326",
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    facilities.to_file(destination, driver="GeoJSON")
    return facilities


def worldcover_class(longitude: float, latitude: float, raster_path: str | Path) -> int | None:
    """Return the ESA WorldCover class at a point from a single known tile.

    Kept for callers that already know which tile a point falls in. For batch
    lookups across a region spanning multiple tiles, use
    training.worldcover_index.WorldCoverTileIndex instead -- it picks the right
    tile per point and reads a small windowed region, rather than requiring the
    caller to load any one raster whole.
    """
    try:
        import rasterio
        from rasterio.sample import sample_gen
    except ImportError as error:
        raise RuntimeError("Install rasterio to use WorldCover lookup") from error

    with rasterio.open(raster_path) as raster:
        value = next(sample_gen(raster, [(longitude, latitude)]))[0]
        return None if raster.nodata is not None and value == raster.nodata else int(value)
