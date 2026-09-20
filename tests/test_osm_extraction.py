from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import osmium
import osmium.osm.mutable as mutable
import pytest

from backend.ingestion.context_sources import (
    OSM_TARGET_TAGS,
    _matched_tag_key,
    extract_osm_industrial_context,
    tag_count_report,
)


def _write_test_pbf(path: Path) -> None:
    """A tiny synthetic .osm.pbf: 3 tagged nodes + 1 closed way (a landuse=industrial
    -tagged square) + 1 untagged node, exercising every OSM_TARGET_TAGS key."""
    writer = osmium.SimpleWriter(str(path))
    writer.add_node(mutable.Node(id=1, location=(69.85, 22.30), tags={"man_made": "flare"}, version=1))
    writer.add_node(mutable.Node(id=2, location=(69.86, 22.31), tags={"power": "plant"}, version=1))
    writer.add_node(mutable.Node(id=3, location=(69.87, 22.32), tags={"shop": "bakery"}, version=1))  # not a target tag
    writer.add_node(mutable.Node(id=4, location=(69.80, 22.30), tags={}, version=1))
    writer.add_node(mutable.Node(id=5, location=(69.81, 22.30), tags={}, version=1))
    writer.add_node(mutable.Node(id=6, location=(69.81, 22.31), tags={}, version=1))
    writer.add_node(mutable.Node(id=7, location=(69.80, 22.31), tags={}, version=1))
    writer.add_way(mutable.Way(id=10, nodes=[4, 5, 6, 7, 4], tags={"landuse": "industrial"}, version=1))
    writer.close()


@pytest.fixture
def sample_pbf(tmp_path: Path) -> Path:
    path = tmp_path / "sample.osm.pbf"
    _write_test_pbf(path)
    return path


def test_matched_tag_key_accepts_any_industrial_value():
    assert _matched_tag_key({"industrial": "oil"}) == "industrial"
    assert _matched_tag_key({"man_made": "flare"}) == "man_made"
    assert _matched_tag_key({"man_made": "bridge"}) is None  # not in the accepted set
    assert _matched_tag_key({"shop": "bakery"}) is None


def test_extract_osm_industrial_context_finds_tagged_nodes_and_closed_way(sample_pbf: Path, tmp_path: Path):
    cache_path = tmp_path / "context.parquet"
    frame = extract_osm_industrial_context(sample_pbf, cache_path)

    assert cache_path.exists()
    assert len(frame) == 3  # 2 tagged nodes + 1 tagged way; the untagged/off-target node and node 3 are excluded
    assert set(frame["matched_tag"]) == {"man_made", "power", "landuse"}

    way_row = frame[frame["osm_type"] == "way"].iloc[0]
    assert way_row.geometry.geom_type == "Polygon"  # closed ring -> Polygon, per the old Overpass heuristic

    node_rows = frame[frame["osm_type"] == "node"]
    assert (node_rows.geometry.geom_type == "Point").all()


def test_extract_osm_industrial_context_uses_cache_on_second_call(sample_pbf: Path, tmp_path: Path):
    cache_path = tmp_path / "context.parquet"
    first = extract_osm_industrial_context(sample_pbf, cache_path)
    # Even pointing at a nonexistent .pbf, the cached Parquet is returned unchanged.
    second = extract_osm_industrial_context(tmp_path / "does-not-exist.osm.pbf", cache_path)
    assert len(first) == len(second)


def test_extract_osm_industrial_context_bbox_filters_by_intersection(sample_pbf: Path, tmp_path: Path):
    # A bbox covering only the way's square (lon 69.80-69.81, lat 22.30-22.31), excluding both nodes.
    frame = extract_osm_industrial_context(
        sample_pbf, tmp_path / "bbox.parquet", bbox=(69.795, 22.295, 69.815, 22.315)
    )
    assert len(frame) == 1
    assert frame.iloc[0]["osm_type"] == "way"


def test_tag_count_report_counts_by_exact_tag_value(sample_pbf: Path, tmp_path: Path):
    frame = extract_osm_industrial_context(sample_pbf, tmp_path / "context.parquet")
    report = tag_count_report(frame)
    tags = dict(zip(report["tag"], report["count"]))
    assert tags == {"man_made=flare": 1, "power=plant": 1, "landuse=industrial": 1}


def test_osm_target_tags_cover_the_four_requested_categories():
    assert OSM_TARGET_TAGS["landuse"] == {"industrial"}
    assert OSM_TARGET_TAGS["industrial"] is None  # any value
    assert OSM_TARGET_TAGS["man_made"] == {"works", "kiln", "flare"}
    assert OSM_TARGET_TAGS["power"] == {"plant"}
