from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd
import pytest

import geopandas as gpd
from shapely.geometry import box

from backend.ingestion.context_sources import load_gem_facilities
from backend.ingestion.firms_client import (
    INDIA_BBOX,
    INDIA_COUNTRY_CODE,
    _clip_to_boundary,
    _env,
    _finalize,
    _format_bbox,
    fetch_firms,
    fetch_firms_country,
    validate_firms_columns,
)


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
    # _resolve_map_key also falls back to a .env file in the cwd (see dnbr_check.py's
    # SH_CLIENT_ID/SECRET convention) -- run from an empty tmp_path so that fallback
    # finds nothing either, regardless of whether the repo's own .env has a real key.
    monkeypatch.chdir(tmp_path)
    with pytest.raises(RuntimeError, match="FIRMS_MAP_KEY"):
        fetch_firms((69, 21, 70, 22), date(2026, 1, 1), date(2026, 1, 5), tmp_path / "out.csv")


def test_load_gem_facilities_requires_lat_lon_columns(tmp_path: Path):
    source_csv = tmp_path / "facilities.csv"
    source_csv.write_text("facility_type,name\ncement_plant,Foo\n")
    with pytest.raises(ValueError, match="latitude/longitude"):
        load_gem_facilities(source_csv, tmp_path / "cache.geojson")


def test_india_bbox_covers_mainland_and_island_territories():
    west, south, east, north = INDIA_BBOX
    # Kanyakumari, Kashmir, Andaman & Nicobar, Gujarat -- all must fall inside.
    for lon, lat in [(77.5, 8.1), (76.0, 34.0), (93.0, 12.0), (69.85, 22.30)]:
        assert west <= lon <= east and south <= lat <= north


def test_env_reads_process_environment_first(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    monkeypatch.setenv("FIRMS_MAP_KEY", "from-environ")
    monkeypatch.chdir(tmp_path)
    assert _env("FIRMS_MAP_KEY") == "from-environ"


def test_env_falls_back_to_dot_env_file(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    monkeypatch.delenv("FIRMS_MAP_KEY", raising=False)
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".env").write_text('FIRMS_MAP_KEY="from-dot-env"\n')
    assert _env("FIRMS_MAP_KEY") == "from-dot-env"


def test_env_missing_everywhere_returns_empty_string(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    monkeypatch.delenv("SOME_UNSET_KEY", raising=False)
    monkeypatch.chdir(tmp_path)
    assert _env("SOME_UNSET_KEY") == ""


def test_fetch_firms_country_builds_the_documented_url_shape(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """Locks in FIRMS' documented /api/country/csv/{key}/{source}/{country}/{day_range}/{date}
    shape, independent of whether the live endpoint currently accepts it (see the
    module docstring: as of writing it returns "Invalid API call" for every request)."""
    captured_urls = []

    class _FakeResponse:
        status_code = 200
        text = "Invalid API call."

        def raise_for_status(self):
            return None

    def fake_get(url, timeout):
        captured_urls.append(url)
        return _FakeResponse()

    monkeypatch.setattr("backend.ingestion.firms_client.requests.get", fake_get)

    with pytest.raises(ValueError, match="Invalid API call"):
        fetch_firms_country(
            INDIA_COUNTRY_CODE, date(2026, 1, 1), date(2026, 1, 3), tmp_path / "out.csv",
            map_key="test-key", source="VIIRS_NOAA20_NRT",
        )

    assert captured_urls == [
        "https://firms.modaps.eosdis.nasa.gov/api/country/csv/test-key/VIIRS_NOAA20_NRT/IND/3/2026-01-01"
    ]


def test_fetch_firms_country_end_date_before_start_date_rejected(tmp_path: Path):
    with pytest.raises(ValueError, match="end_date"):
        fetch_firms_country(INDIA_COUNTRY_CODE, date(2026, 1, 10), date(2026, 1, 1), tmp_path / "out.csv", map_key="k")


def test_clip_to_boundary_drops_points_outside_the_polygon(tmp_path: Path):
    boundary = gpd.GeoDataFrame({"name": ["square"]}, geometry=[box(69.0, 21.0, 70.0, 22.0)], crs="EPSG:4326")
    boundary_path = tmp_path / "boundary.parquet"
    boundary.to_parquet(boundary_path)

    frame = pd.DataFrame({
        "latitude": [21.5, 21.5, 30.0],
        "longitude": [69.5, 95.0, 69.5],  # inside, outside (east), outside (north)
        "acq_date": ["2026-01-01", "2026-01-01", "2026-01-01"],
    })
    clipped = _clip_to_boundary(frame, boundary_path)
    assert len(clipped) == 1
    assert clipped.iloc[0]["longitude"] == 69.5
    assert clipped.iloc[0]["latitude"] == 21.5


def _mixed_source_chunk(version_value: str) -> pd.DataFrame:
    return pd.DataFrame({
        "latitude": [21.5], "longitude": [69.5], "acq_date": ["2026-01-01"], "acq_time": ["0130"],
        "satellite": ["N20"], "bright_ti4": [300.0], "bright_ti5": [290.0], "frp": [1.5],
        "confidence": ["n"], "daynight": ["D"], "version": [version_value],
    })


def test_finalize_parquet_survives_mixed_sp_nrt_version_column(tmp_path: Path):
    """Regression test: concatenating an SP chunk (version like "2.0") with an NRT
    chunk (version like "2.0NRT") -- any multi-satellite pull spanning both, since
    NOAA-21 has no SP source at all -- used to raise pyarrow.lib.ArrowInvalid on
    to_parquet because pandas left `version` as a mixed-content object column."""
    frames = [_mixed_source_chunk("2.0"), _mixed_source_chunk("2.0NRT")]
    output_path = tmp_path / "out.parquet"
    result = _finalize(frames, date(2026, 1, 1), date(2026, 1, 1), output_path)
    assert len(result) == 2

    on_disk = pd.read_parquet(output_path)
    assert set(on_disk["version"]) == {"2.0", "2.0NRT"}
    assert on_disk["version"].dtype == object or str(on_disk["version"].dtype) == "str"


def test_finalize_parquet_keeps_latitude_longitude_numeric(tmp_path: Path):
    """Regression test for the fix's own first version: it cast *every*
    object-dtype column to string, including latitude/longitude/bright_ti4/etc
    whenever per-chunk dtype inference happened to leave one of them as object
    before concat (observed for real on a Mar-May 2026 national pull -- no bad
    values involved, pure pandas dtype-inference happenstance across many 5-day
    API chunks). Numeric columns must come out of the parquet round-trip as
    numeric, not string, regardless of what dtype they had in memory beforehand.
    """
    chunk_a = _mixed_source_chunk("2.0")
    chunk_b = _mixed_source_chunk("2.0NRT")
    # Simulate the real-world dtype-inference quirk: one chunk's latitude ends up
    # as a Python-object column (as if read_csv had inferred it that way) before
    # concatenation, even though every value is genuinely numeric.
    chunk_b["latitude"] = chunk_b["latitude"].astype(object)

    output_path = tmp_path / "out.parquet"
    _finalize([chunk_a, chunk_b], date(2026, 1, 1), date(2026, 1, 1), output_path)

    on_disk = pd.read_parquet(output_path)
    assert pd.api.types.is_numeric_dtype(on_disk["latitude"])
    assert pd.api.types.is_numeric_dtype(on_disk["longitude"])
    assert pd.api.types.is_numeric_dtype(on_disk["bright_ti4"])
    assert on_disk["latitude"].tolist() == [21.5, 21.5]


def test_finalize_parquet_raises_loudly_on_genuinely_bad_numeric_value(tmp_path: Path):
    """A real non-numeric value in a numeric column must fail loudly (errors="raise"
    in the pd.to_numeric coercion), not get silently stringified the way the
    blanket-cast version of this fix would have."""
    chunk = _mixed_source_chunk("2.0")
    chunk["latitude"] = ["not-a-number"]
    output_path = tmp_path / "out.parquet"
    with pytest.raises((ValueError, TypeError)):
        _finalize([chunk], date(2026, 1, 1), date(2026, 1, 1), output_path)
