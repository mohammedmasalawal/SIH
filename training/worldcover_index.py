"""WorldCover tile index: locate and read the right 10m ESA WorldCover tile for a
point/region without loading any raster whole.

WorldCover ships as 3deg x 3deg GeoTIFF tiles named by their SW corner
(ESA_WorldCover_10m_2021_v200_{NxxExxx}_Map.tif). At national scale there are 100+
of these covering India's bbox alone; opening and reading each one in full for a
handful of lookups would be wasteful (each tile is ~100-150MB uncompressed in
memory). WorldCoverTileIndex instead builds a small in-memory index of each tile's
bounding box -- parsed straight from its filename, no raster is opened just to
build the index -- and, for each point/buffer query, opens only the ONE tile that
covers it and reads a small windowed region via rasterio, so the raster's pixel
data outside that window is never touched.
"""

from __future__ import annotations

import re
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import from_bounds
from shapely.geometry import Point, box

_TILE_NAME_RE = re.compile(r"(?P<ns>[NS])(?P<lat>\d{2})(?P<ew>[EW])(?P<lon>\d{3})")


def tile_bounds_from_name(path: str | Path) -> tuple[float, float, float, float] | None:
    """Parse a WorldCover tile's bbox from its SW-corner filename code, e.g.
    ..._N21E069_Map.tif -> (69.0, 21.0, 72.0, 24.0). Tiles are always 3deg x 3deg.
    Returns None if the filename doesn't contain a recognisable tile code.
    """
    match = _TILE_NAME_RE.search(Path(path).stem)
    if match is None:
        return None
    lat = int(match["lat"]) * (1 if match["ns"] == "N" else -1)
    lon = int(match["lon"]) * (1 if match["ew"] == "E" else -1)
    return (float(lon), float(lat), float(lon + 3), float(lat + 3))


class WorldCoverTileIndex:
    """Maps points/regions to the WorldCover tile file that covers them.

    Built once from a directory of downloaded .tif tiles -- index construction
    only reads filenames, not raster data, so it's effectively instant regardless
    of how many tiles are present.
    """

    def __init__(self, tile_dir: str | Path):
        self.tile_dir = Path(tile_dir)
        records = []
        for path in sorted(self.tile_dir.glob("*.tif")):
            bounds = tile_bounds_from_name(path)
            if bounds is None:
                continue
            records.append({"path": str(path), "geometry": box(*bounds)})
        if not records:
            raise FileNotFoundError(f"No WorldCover .tif tiles found under {self.tile_dir}")
        self.index = gpd.GeoDataFrame(records, crs="EPSG:4326")
        self.index.sindex  # build the R-tree eagerly, not on first query

    def __len__(self) -> int:
        return len(self.index)

    def tile_for_point(self, longitude: float, latitude: float) -> Path | None:
        """The single tile file covering (longitude, latitude), or None if uncovered."""
        candidates = list(self.index.sindex.query(Point(longitude, latitude), predicate="intersects"))
        return Path(self.index.iloc[candidates[0]]["path"]) if candidates else None

    def class_at_points(self, longitudes: pd.Series, latitudes: pd.Series) -> pd.Series:
        """Single-pixel WorldCover class per point, batched by covering tile so each
        tile file is opened at most once regardless of how many points fall in it,
        and only the queried pixels are read (rasterio.sample is itself a windowed
        read, not a whole-raster read).
        """
        points = gpd.GeoDataFrame(
            {"_lon": longitudes.to_numpy(), "_lat": latitudes.to_numpy()},
            geometry=gpd.points_from_xy(longitudes, latitudes),
            index=longitudes.index,
            crs="EPSG:4326",
        )
        joined = gpd.sjoin(points, self.index[["geometry", "path"]], how="left", predicate="within")
        result = pd.Series(pd.NA, index=longitudes.index, dtype="object")
        for path, group in joined.dropna(subset=["path"]).groupby("path"):
            with rasterio.open(path) as raster:
                nodata = raster.nodata
                coords = list(zip(group["_lon"], group["_lat"]))
                for idx, sampled in zip(group.index, raster.sample(coords)):
                    value = sampled[0]
                    result.loc[idx] = pd.NA if (nodata is not None and value == nodata) else int(value)
        return result

    def majority_class_in_buffer(self, longitude: float, latitude: float, buffer_m: float = 375) -> int | None:
        """Majority WorldCover class within buffer_m metres of one point.

        Opens only the covering tile and reads only the small window overlapping
        the buffer (rasterio windowed read via rasterio.windows.from_bounds) --
        never the tile's full extent, regardless of tile size.
        """
        tile_path = self.tile_for_point(longitude, latitude)
        if tile_path is None:
            return None
        with rasterio.open(tile_path) as raster:
            point_3857 = gpd.GeoSeries([Point(longitude, latitude)], crs="EPSG:4326").to_crs("EPSG:3857")
            buffered = point_3857.buffer(buffer_m).to_crs(raster.crs)
            window = from_bounds(*buffered.total_bounds, transform=raster.transform)
            data = raster.read(1, window=window, boundless=True, fill_value=raster.nodata or 0)
            nodata = raster.nodata
            values = data[data != 0]
            if nodata is not None:
                values = values[values != nodata]
            if values.size == 0:
                return None
            counts = np.bincount(values.astype(np.int64))
            return int(np.argmax(counts))
