from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import box

from training.report_by_state import assign_state, label_mix_by_state


@pytest.fixture
def two_state_boundaries(tmp_path: Path) -> Path:
    """Two adjacent 1deg x 1deg synthetic "states" for a self-contained join test."""
    states = gpd.GeoDataFrame(
        {"name": ["Statesville", "Provincetown"]},
        geometry=[box(69.0, 22.0, 70.0, 23.0), box(70.0, 22.0, 71.0, 23.0)],
        crs="EPSG:4326",
    )
    path = tmp_path / "states.parquet"
    states.to_parquet(path)
    return path


def test_assign_state_matches_point_in_polygon(two_state_boundaries: Path):
    frame = pd.DataFrame({
        "latitude": [22.5, 22.5, 22.5],
        "longitude": [69.5, 70.5, 50.0],  # inside Statesville, inside Provincetown, outside both
    })
    result = assign_state(frame, two_state_boundaries)
    assert result.iloc[0] == "Statesville"
    assert result.iloc[1] == "Provincetown"
    assert pd.isna(result.iloc[2])


def test_label_mix_by_state_crosstab_has_totals(two_state_boundaries: Path):
    frame = pd.DataFrame({
        "latitude": [22.5, 22.5, 22.5, 22.5],
        "longitude": [69.5, 69.5, 70.5, 50.0],
        "label": ["industrial", "gas flare", "industrial", "wildfire"],
    })
    table = label_mix_by_state(frame, two_state_boundaries)
    assert table.loc["Statesville", "total"] == 2
    assert table.loc["Provincetown", "total"] == 1
    assert table.loc["(outside any state polygon)", "total"] == 1
    assert table["total"].sum() == len(frame)
