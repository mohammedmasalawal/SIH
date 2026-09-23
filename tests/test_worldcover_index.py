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


@pytest.fixture
def two_fine_tile_dir(tmp_path: Path) -> Path:
    """Two adjacent fine-resolution (~55m/pixel) tiles, each large enough that a
    375m buffer stays well clear of both the tile edge and the raster's own edge."""
    tile_dir = tmp_path / "fine_tiles"
    tile_dir.mkdir()
    pixel = 0.0005
    for name, sw_lon, sw_lat, fill_value in [
        ("ESA_WorldCover_10m_2021_v200_N21E069_Map.tif", 69.0, 21.0, 50),
        ("ESA_WorldCover_10m_2021_v200_N21E072_Map.tif", 72.0, 21.0, 40),
    ]:
        transform = from_origin(sw_lon, sw_lat + 0.15, pixel, pixel)
        data = np.full((300, 300), fill_value, dtype="uint8")
        with rasterio.open(
            tile_dir / name, "w", driver="GTiff", height=300, width=300, count=1, dtype="uint8",
            crs="EPSG:4326", transform=transform, nodata=0,
        ) as dst:
            dst.write(data, 1)
    return tile_dir


def test_majority_class_in_buffer_batch_matches_single_point_method(two_fine_tile_dir: Path):
    index = WorldCoverTileIndex(two_fine_tile_dir)
    longitudes = pd.Series([69.05, 72.05, 69.06], index=["a", "b", "c"])
    latitudes = pd.Series([21.05, 21.05, 21.06], index=["a", "b", "c"])

    batched = index.majority_class_in_buffer_batch(longitudes, latitudes)
    assert batched.loc["a"] == index.majority_class_in_buffer(69.05, 21.05)
    assert batched.loc["b"] == index.majority_class_in_buffer(72.05, 21.05)
    assert batched.loc["c"] == index.majority_class_in_buffer(69.06, 21.06)
    assert batched.loc["a"] == 50 and batched.loc["c"] == 50  # tile A
    assert batched.loc["b"] == 40  # tile B


def test_majority_class_in_buffer_batch_opens_each_tile_at_most_once(two_fine_tile_dir: Path, monkeypatch):
    """Regression guard for the real bug this batch method fixes: an earlier
    per-row loop opened a fresh rasterio dataset for every single detection, which
    leaked enough GDAL state to crash with an out-of-memory error ~2.5 hours into a
    ~1.09M-row national run. 10 points across 2 tiles must open rasterio.open at
    most twice, not up to 10 times."""
    import rasterio as rasterio_module

    index = WorldCoverTileIndex(two_fine_tile_dir)
    open_calls = []
    real_open = rasterio_module.open

    def counting_open(path, *args, **kwargs):
        open_calls.append(str(path))
        return real_open(path, *args, **kwargs)

    monkeypatch.setattr("training.worldcover_index.rasterio.open", counting_open)

    longitudes = pd.Series([69.05 + 0.001 * i for i in range(5)] + [72.05 + 0.001 * i for i in range(5)])
    latitudes = pd.Series([21.05] * 10)
    index.majority_class_in_buffer_batch(longitudes, latitudes)

    assert len(open_calls) <= 2
    assert len(set(open_calls)) == len(open_calls)  # each tile opened once, not reopened per point
