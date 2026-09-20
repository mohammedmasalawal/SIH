from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import rasterio
from rasterio.transform import from_origin

from training.worldcover_index import WorldCoverTileIndex, tile_bounds_from_name


def _write_tile(path: Path, sw_lon: float, sw_lat: float, fill_value: int, size: int = 30) -> None:
    """A size x size, 0.1deg/pixel synthetic WorldCover-shaped tile covering a 3deg
    square starting at (sw_lon, sw_lat), filled uniformly with fill_value."""
    pixel = 3.0 / size
    transform = from_origin(sw_lon, sw_lat + 3.0, pixel, pixel)
    data = np.full((size, size), fill_value, dtype="uint8")
    with rasterio.open(
        path, "w", driver="GTiff", height=size, width=size, count=1, dtype="uint8",
        crs="EPSG:4326", transform=transform, nodata=0,
    ) as dst:
        dst.write(data, 1)


def test_tile_bounds_from_name_parses_sw_corner():
    assert tile_bounds_from_name("ESA_WorldCover_10m_2021_v200_N21E069_Map.tif") == (69.0, 21.0, 72.0, 24.0)
    assert tile_bounds_from_name("weird_filename.tif") is None


@pytest.fixture
def two_tile_dir(tmp_path: Path) -> Path:
    tile_dir = tmp_path / "tiles"
    tile_dir.mkdir()
    # Tile A: N21E069 (lon 69-72, lat 21-24), filled with class 50 (built-up)
    _write_tile(tile_dir / "ESA_WorldCover_10m_2021_v200_N21E069_Map.tif", 69, 21, fill_value=50)
    # Tile B: N21E072 (lon 72-75, lat 21-24), filled with class 40 (cropland)
    _write_tile(tile_dir / "ESA_WorldCover_10m_2021_v200_N21E072_Map.tif", 72, 21, fill_value=40)
    return tile_dir


def test_index_construction_reads_no_raster_data(two_tile_dir: Path):
    index = WorldCoverTileIndex(two_tile_dir)
    assert len(index) == 2


def test_index_construction_raises_on_empty_dir(tmp_path: Path):
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    with pytest.raises(FileNotFoundError):
        WorldCoverTileIndex(empty_dir)


def test_tile_for_point_picks_the_covering_tile(two_tile_dir: Path):
    index = WorldCoverTileIndex(two_tile_dir)
    assert "N21E069" in str(index.tile_for_point(70.0, 22.0))
    assert "N21E072" in str(index.tile_for_point(73.0, 22.0))
    assert index.tile_for_point(0.0, 0.0) is None  # nowhere near either tile


def test_class_at_points_batches_by_tile_and_returns_correct_class(two_tile_dir: Path):
    index = WorldCoverTileIndex(two_tile_dir)
    longitudes = pd.Series([70.0, 73.0, 70.5], index=[10, 20, 30])
    latitudes = pd.Series([22.0, 22.0, 22.5], index=[10, 20, 30])
    result = index.class_at_points(longitudes, latitudes)
    assert result.loc[10] == 50  # tile A
    assert result.loc[20] == 40  # tile B
    assert result.loc[30] == 50  # tile A again


def test_majority_class_in_buffer_reads_only_a_window(tmp_path: Path):
    # two_tile_dir's tiles are 0.1deg/pixel (~11km) -- far coarser than real
    # WorldCover's 10m, so a 375m buffer wouldn't cover even one pixel. This test
    # uses a dedicated, finer-resolution tile (still named as a 3deg tile for
    # routing purposes -- majority_class_in_buffer reads the raster's own
    # transform, not the filename bbox, so its actual extent/resolution can differ).
    fine_dir = tmp_path / "fine_tiles"
    fine_dir.mkdir()
    pixel = 0.0005  # ~55m/pixel near the equator -- a 375m buffer spans several pixels
    transform = from_origin(69.0, 21.05, pixel, pixel)
    data = np.full((100, 100), 50, dtype="uint8")
    path = fine_dir / "ESA_WorldCover_10m_2021_v200_N21E069_Map.tif"
    with rasterio.open(
        path, "w", driver="GTiff", height=100, width=100, count=1, dtype="uint8",
        crs="EPSG:4326", transform=transform, nodata=0,
    ) as dst:
        dst.write(data, 1)

    index = WorldCoverTileIndex(fine_dir)
    result = index.majority_class_in_buffer(69.025, 21.025, buffer_m=375)
    assert result == 50


def test_majority_class_in_buffer_returns_none_outside_any_tile(two_tile_dir: Path):
    index = WorldCoverTileIndex(two_tile_dir)
    assert index.majority_class_in_buffer(0.0, 0.0) is None
