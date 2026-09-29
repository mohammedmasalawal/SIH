from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
import rasterio
from rasterio.transform import from_origin

from training.report_cropland_share_impact import compute_impact, load_wildfire_shrub_grass, report


POINT_LAT = 22.075  # 69.03E here is just offshore (Gulf of Kutch); 69.12E is Gujarat land


@pytest.fixture
def worldcover_dir(tmp_path):
    """One fine-resolution tile: west half shrubland/cropland-adjacent (class 20),
    east half cropland (class 40), split at lon 69.075 -- same layout as
    test_worldcover_index.py's split_class_tile_dir."""
    tile_dir = tmp_path / "tiles"
    tile_dir.mkdir()
    pixel = 0.0005
    size = 300
    transform = from_origin(69.0, 22.0 + size * pixel, pixel, pixel)
    data = np.full((size, size), 20, dtype="uint8")
    data[:, size // 2:] = 40
    with rasterio.open(
        tile_dir / "ESA_WorldCover_10m_2021_v200_N22E069_Map.tif", "w", driver="GTiff",
        height=size, width=size, count=1, dtype="uint8", crs="EPSG:4326", transform=transform, nodata=0,
    ) as dst:
        dst.write(data, 1)
    return tile_dir


@pytest.fixture
def meta_path(tmp_path):
    """Two source parquets (mimics training.pack_map_points' meta.json "sources"),
    with a mix of labels/classes so filtering to wildfire+shrub/grass is exercised."""
    rows_a = pd.DataFrame([
        {"latitude": POINT_LAT, "longitude": 69.03, "label": "wildfire", "landcover_class": 20},   # deep west: 0% cropland
        {"latitude": POINT_LAT, "longitude": 69.12, "label": "wildfire", "landcover_class": 30},   # deep east: 100% cropland
        {"latitude": POINT_LAT, "longitude": 69.03, "label": "wildfire", "landcover_class": 10},   # tree cover -- excluded
        {"latitude": POINT_LAT, "longitude": 69.12, "label": "industrial", "landcover_class": 30},  # not wildfire -- excluded
    ])
    rows_b = pd.DataFrame([
        {"latitude": POINT_LAT, "longitude": 69.12, "label": "wildfire", "landcover_class": 20},   # deep east on shrubland: 100%
    ])
    path_a, path_b = tmp_path / "a.parquet", tmp_path / "b.parquet"
    rows_a.to_parquet(path_a, index=False)
    rows_b.to_parquet(path_b, index=False)
    meta = tmp_path / "meta.json"
    meta.write_text(json.dumps({"sources": [{"path": str(path_a)}, {"path": str(path_b)}]}))
    return meta


def test_load_filters_to_wildfire_shrubland_and_grassland(meta_path):
    rows = load_wildfire_shrub_grass(meta_path)
    assert len(rows) == 3  # excludes tree cover and the non-wildfire row
    assert set(rows["landcover_class"]) == {20, 30}
    assert (rows["label"] == "wildfire").all()


def test_compute_impact_flags_high_cropland_share_and_applies_nothing(meta_path, worldcover_dir):
    rows = load_wildfire_shrub_grass(meta_path)
    out = compute_impact(rows, worldcover_dir=worldcover_dir, radius_m=500, threshold=0.5)
    # deep-west point (lon 69.03): ~0% cropland share -> unaffected
    west = out[out["longitude"] == 69.03].iloc[0]
    assert west["cropland_share_1km"] == pytest.approx(0.0, abs=0.01)
    assert west["would_reclassify"] == False  # noqa: E712
    # deep-east points (lon 69.12): ~100% cropland share -> would reclassify
    east = out[out["longitude"] == 69.12]
    assert (east["cropland_share_1km"] > 0.99).all()
    assert east["would_reclassify"].all()
    # nothing written back onto the row's own label
    assert set(out["label"]) == {"wildfire"}


def test_compute_impact_empty_input_returns_empty_with_new_columns(worldcover_dir):
    empty = pd.DataFrame(columns=["latitude", "longitude", "label", "landcover_class"])
    out = compute_impact(empty, worldcover_dir=worldcover_dir)
    assert out.empty and "cropland_share_1km" in out.columns and "would_reclassify" in out.columns


def test_report_counts_by_state_and_class(meta_path, worldcover_dir):
    rows = load_wildfire_shrub_grass(meta_path)
    out = compute_impact(rows, worldcover_dir=worldcover_dir, radius_m=500, threshold=0.5)
    text = report(out)
    assert "3 wildfire detections" in text
    assert "2 of 3" in text  # the two lon=69.12 rows would reclassify
    assert "Gujarat" in text  # both changed rows sit at 69.12E, which is Gujarat land at this latitude
