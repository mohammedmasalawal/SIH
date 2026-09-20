from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from backend.ingestion.context_sources import load_gem_facilities
from backend.ingestion.firms_client import _format_bbox, fetch_firms, validate_firms_columns


def test_bbox_validation():
    assert _format_bbox((69.5, 21.9, 70.5, 22.8)) == "69.5,21.9,70.5,22.8"
    with pytest.raises(ValueError):
        _format_bbox((70, 22, 69, 23))


def test_firms_columns_fail_loudly():
    with pytest.raises(ValueError, match="frp"):
        validate_firms_columns(pd.DataFrame({"latitude": [1]}))


def test_end_date_before_start_date_rejected(tmp_path: Path):
    with pytest.raises(ValueError, match="end_date"):
        fetch_firms((69, 21, 70, 22), date(2026, 1, 10), date(2026, 1, 1), tmp_path / "out.csv")


def test_missing_map_key_raises(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("FIRMS_MAP_KEY", raising=False)
    with pytest.raises(RuntimeError, match="FIRMS_MAP_KEY"):
        fetch_firms((69, 21, 70, 22), date(2026, 1, 1), date(2026, 1, 5), tmp_path / "out.csv")


def test_load_gem_facilities_requires_lat_lon_columns(tmp_path: Path):
    source_csv = tmp_path / "facilities.csv"
    source_csv.write_text("facility_type,name\ncement_plant,Foo\n")
    with pytest.raises(ValueError, match="latitude/longitude"):
        load_gem_facilities(source_csv, tmp_path / "cache.geojson")
