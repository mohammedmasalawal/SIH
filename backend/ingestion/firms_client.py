"""NASA FIRMS VIIRS client: bbox area pulls (testing/regional) and country archive pulls (bulk).

Bulk national-scale pulls should use fetch_firms_country_multi -- it's the documented
FIRMS bulk-download shape (/api/country/csv/...) and avoids having to hand-maintain a
country bounding box. As of this writing FIRMS' country endpoint itself returns
"Invalid API call" for every request (confirmed against the live API, and the FIRMS
API docs page marks /api/country/ "* Feature currently not available."), so
fetch_firms_country_multi will raise -- that's NASA's service, not a bug here. Until
it's re-enabled upstream, fetch_firms_multi(INDIA_BBOX, ...) is the working substitute
for a national pull; the bbox path stays the default for regional/testing pulls either
way (smaller, faster, doesn't depend on the country endpoint at all).

Both bbox and country are loose shapes -- a rectangle and (once available) a whole
country record respectively -- neither clips to an actual boundary polygon. Pass
clip_boundary=backend.config.INDIA_BOUNDARY_PATH to any fetch_* function for a real
India pull: INDIA_BBOX alone also pulls in genuine fires from Myanmar, Sri Lanka, and
Pakistan (measured at 47% of one uncleared one-month pull).
"""

from __future__ import annotations

import argparse
import os
from datetime import date, timedelta
from pathlib import Path
from typing import Callable, Iterable

import geopandas as gpd
import pandas as pd
import requests

API_ROOT = "https://firms.modaps.eosdis.nasa.gov/api"
DEFAULT_SOURCE = "VIIRS_NOAA20_NRT"
REQUIRED_COLUMNS = {
    "latitude",
    "longitude",
    "acq_date",
    "acq_time",
    "satellite",
    "bright_ti4",
    "bright_ti5",
    "frp",
    "confidence",
    "daynight",
}

# Mainland + islands (Lakshadweep, Andaman & Nicobar) -- generous enough to include
# every territory FIRMS could report as India without also pulling all of Pakistan/
# China/Myanmar; the OSM/GEM/WorldCover context joins downstream naturally drop
# anything outside real coverage, so a slightly loose bbox costs nothing but a few
# extra rows.
INDIA_BBOX = (68.0, 6.0, 97.5, 37.5)
INDIA_COUNTRY_CODE = "IND"  # ISO 3166-1 alpha-3, the id FIRMS' country API expects

# Each VIIRS satellite has its own archive (SP, science-quality) vs near-real-time
# (NRT) cutover date, and it moves as NRT rolls forward and SP reprocessing catches
# up -- NOAA-21 currently has no separate SP source at all. Resolved dynamically
# against data_availability rather than hardcoded so this doesn't go stale.
SATELLITE_SOURCES: dict[str, tuple[str, ...]] = {
    "NOAA20": ("VIIRS_NOAA20_SP", "VIIRS_NOAA20_NRT"),
    "NOAA21": ("VIIRS_NOAA21_NRT",),
    "SNPP": ("VIIRS_SNPP_SP", "VIIRS_SNPP_NRT"),
}
# Near-real-time only -- for live ingestion, where every requested day is recent
# enough to be NRT and mixing in SP for a straggling day would be inconsistent.
NRT_SATELLITE_SOURCES: dict[str, tuple[str, ...]] = {
    "NOAA20": ("VIIRS_NOAA20_NRT",),
    "NOAA21": ("VIIRS_NOAA21_NRT",),
    "SNPP": ("VIIRS_SNPP_NRT",),
}


def _env(name: str) -> str:
    """os.environ first, then a plain-text .env fallback -- same convention dnbr_check.py
    uses for SH_CLIENT_ID/SH_CLIENT_SECRET, so both scripts read credentials the same way.
    """
    value = os.environ.get(name, "").strip()
    env_file = Path(".env")
    if not value and env_file.exists():
        for line in env_file.read_text().splitlines():
            if "=" not in line or line.strip().startswith("#"):
                continue
            key, _, raw_value = line.partition("=")
            if key.strip() == name:
                value = raw_value.strip().strip('"').strip("'")
                break
    return value


def _resolve_map_key(map_key: str | None) -> str:
    key = map_key or _env("FIRMS_MAP_KEY")
    if not key:
        raise RuntimeError("Set FIRMS_MAP_KEY (env var or .env) or pass map_key explicitly")
    return key


def _format_bbox(bbox: Iterable[float]) -> str:
    values = tuple(float(value) for value in bbox)
    if len(values) != 4:
        raise ValueError("bbox must contain west, south, east, north")
    west, south, east, north = values
    if not west < east or not south < north:
        raise ValueError("bbox must satisfy west < east and south < north")
    if not -180 <= west <= 180 or not -180 <= east <= 180:
        raise ValueError("longitude must be between -180 and 180")
    if not -90 <= south <= 90 or not -90 <= north <= 90:
        raise ValueError("latitude must be between -90 and 90")
    return ",".join(f"{value:g}" for value in values)


def validate_firms_columns(frame: pd.DataFrame) -> None:
    missing = sorted(REQUIRED_COLUMNS - set(frame.columns))
    if missing:
        raise ValueError(f"FIRMS response is missing required columns: {', '.join(missing)}")


def _fetch_chunk(url: str, *, timeout_seconds: int) -> str:
    response = requests.get(url, timeout=timeout_seconds)
    response.raise_for_status()
    text = response.text.strip()
    if not text:
        raise ValueError("FIRMS returned an empty response")
    if text.startswith("Invalid") or "error" in text.lower()[:20]:
        raise ValueError(f"FIRMS rejected the request: {text}")
    return text


def _fetch_range(
    url_builder: Callable[[date, int], str],
    start_date: date,
    end_date: date,
    *,
    timeout_seconds: int,
    chunk_days: int,
) -> pd.DataFrame:
    """Walk [start_date, end_date] in chunk_days-sized windows, calling url_builder(window_start, day_range) each time."""
    chunks: list[str] = []
    chunk_start = start_date
    while chunk_start <= end_date:
        chunk_end = min(end_date, chunk_start + timedelta(days=chunk_days - 1))
        day_range = (chunk_end - chunk_start).days + 1
        chunks.append(_fetch_chunk(url_builder(chunk_start, day_range), timeout_seconds=timeout_seconds))
        chunk_start = chunk_end + timedelta(days=1)
    frames = [pd.read_csv(pd.io.common.StringIO(text)) for text in chunks]
    return pd.concat(frames, ignore_index=True) if len(frames) > 1 else frames[0]


def _data_availability(key: str, timeout_seconds: int = 60) -> dict[str, tuple[date, date]]:
    url = f"{API_ROOT}/data_availability/csv/{key}/all"
    response = requests.get(url, timeout=timeout_seconds)
    response.raise_for_status()
    frame = pd.read_csv(pd.io.common.StringIO(response.text))
    return {
        row["data_id"]: (date.fromisoformat(row["min_date"]), date.fromisoformat(row["max_date"]))
        for _, row in frame.iterrows()
    }


def _resolve_segments(
    source_names: tuple[str, ...],
    start_date: date,
    end_date: date,
    availability: dict[str, tuple[date, date]],
) -> list[tuple[str, date, date]]:
    """Split [start_date, end_date] into (source, seg_start, seg_end) pieces, each covered by one source.

    Sources are tried in the given order for each day. A day outside every listed
    source's coverage is dropped as a real gap (e.g. a satellite that didn't exist
    yet) rather than raising, so it doesn't block fetching the rest of the range.
    """
    segments: list[tuple[str, date, date]] = []
    current_source: str | None = None
    segment_start: date | None = None
    day = start_date
    while day <= end_date:
        covering_source = next(
            (
                name
                for name in source_names
                if name in availability and availability[name][0] <= day <= availability[name][1]
            ),
            None,
        )
        if covering_source != current_source:
            if current_source is not None:
                segments.append((current_source, segment_start, day - timedelta(days=1)))
            current_source = covering_source
            segment_start = day
        day += timedelta(days=1)
    if current_source is not None:
        segments.append((current_source, segment_start, end_date))
    return segments


def _clip_to_boundary(frame: pd.DataFrame, boundary_path: str | Path) -> pd.DataFrame:
    """Keep only rows whose (longitude, latitude) falls inside the dissolved
    polygon(s) at boundary_path (a GeoParquet, unioned here if it has more than one
    row -- see data/external/boundaries/india_states.parquet, 36 India state/UT
    polygons). A bbox pull is necessarily a loose rectangle (FIRMS' area API takes
    no other shape) -- for INDIA_BBOX specifically this also catches real fires in
    Myanmar, Sri Lanka, and Pakistan (47% of one uncleared one-month pull); this is
    what actually restricts detections to the country's real shape, after the fact.
    """
    boundary = gpd.read_parquet(boundary_path)
    boundary_union = boundary.geometry.union_all()
    points = gpd.GeoDataFrame(
        frame, geometry=gpd.points_from_xy(frame["longitude"], frame["latitude"]), crs="EPSG:4326"
    )
    inside = points.geometry.within(boundary_union).to_numpy()
    return frame[inside].reset_index(drop=True)


def _finalize(
    frames: list[pd.DataFrame],
    start_date: date,
    end_date: date,
    output_path: str | Path,
    *,
    clip_boundary: str | Path | None = None,
) -> pd.DataFrame:
    combined = pd.concat(frames, ignore_index=True) if len(frames) > 1 else frames[0]
    validate_firms_columns(combined)
    combined = combined[
        (combined["acq_date"] >= start_date.isoformat()) & (combined["acq_date"] <= end_date.isoformat())
    ]
    sort_cols = [c for c in ("acq_date", "acq_time", "satellite") if c in combined.columns]
    combined = combined.drop_duplicates().sort_values(sort_cols).reset_index(drop=True)
    if clip_boundary is not None:
        combined = _clip_to_boundary(combined, clip_boundary)
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.suffix == ".parquet":
        # FIRMS' `version` column (and potentially others) is formatted differently
        # per source -- SP archive rows read as e.g. "2.0", NRT rows as "2.0NRT" --
        # so a pull spanning both (any multi-satellite range predating the SP/NRT
        # cutover, since NOAA-21 has no SP at all) concatenates into an object
        # column pandas/pyarrow can't agree on a single Arrow type for, and
        # to_parquet raises ArrowInvalid. An earlier version of this fix cast
        # *every* object-dtype column to string, which also silently stringified
        # latitude/longitude/bright_ti4/bright_ti5/frp/scan/track/type whenever
        # per-chunk dtype inference happened to leave one of them as object before
        # concat (observed on two real pulls, no bad values involved -- pure
        # pandas dtype-inference happenstance across ~18 five-day chunks per
        # satellite segment) -- every downstream numeric op on those columns broke
        # silently-until-used. Numeric columns are now coerced back to numeric
        # explicitly (errors="raise": a real non-numeric value here should fail
        # loudly, not get quietly stringified); only the genuinely textual leftover
        # object columns (version, confidence, satellite, ...) are cast to string.
        # CSV output is unaffected (already untyped text either way).
        to_write = combined.copy()
        numeric_columns = [
            c for c in ("latitude", "longitude", "bright_ti4", "bright_ti5", "frp", "scan", "track", "type")
            if c in to_write.columns
        ]
        for column in numeric_columns:
            to_write[column] = pd.to_numeric(to_write[column], errors="raise")
        object_columns = [c for c in to_write.select_dtypes(include="object").columns if c not in numeric_columns]
        to_write[object_columns] = to_write[object_columns].astype(str)
        to_write.to_parquet(destination, index=False)
    else:
        combined.to_csv(destination, index=False)
    return combined


def fetch_firms(
    bbox: Iterable[float],
    start_date: date,
    end_date: date,
    output_path: str | Path,
    *,
    map_key: str | None = None,
    source: str = DEFAULT_SOURCE,
    timeout_seconds: int = 60,
    chunk_days: int = 5,
    clip_boundary: str | Path | None = None,
) -> pd.DataFrame:
    """Fetch FIRMS CSV data for one bbox from a single named source (area API).

    For routine regional/testing pulls prefer fetch_firms_multi, which merges all
    three VIIRS satellites with the correct SP/NRT source per sub-range. Use this
    directly only when you deliberately want one named source (e.g. debugging a
    single satellite). Writes CSV if output_path ends in .csv, Parquet if .parquet.

    clip_boundary, if given, drops rows outside that polygon file after fetching
    (see _clip_to_boundary) -- bbox is necessarily a rectangle, so pass
    backend.config.INDIA_BOUNDARY_PATH here for a real-shape India pull.
    """
    if end_date < start_date:
        raise ValueError("end_date must be on or after start_date")
    key = _resolve_map_key(map_key)
    bbox_str = _format_bbox(bbox)

    def url_builder(window_start: date, day_range: int) -> str:
        return f"{API_ROOT}/area/csv/{key}/{source}/{bbox_str}/{day_range}/{window_start.isoformat()}"

    frame = _fetch_range(url_builder, start_date, end_date, timeout_seconds=timeout_seconds, chunk_days=chunk_days)
    return _finalize([frame], start_date, end_date, output_path, clip_boundary=clip_boundary)


def fetch_firms_multi(
    bbox: Iterable[float],
    start_date: date,
    end_date: date,
    output_path: str | Path,
    *,
    satellites: tuple[str, ...] = ("NOAA20", "NOAA21", "SNPP"),
    map_key: str | None = None,
    timeout_seconds: int = 60,
    chunk_days: int = 5,
    clip_boundary: str | Path | None = None,
    satellite_sources: dict[str, tuple[str, ...]] = SATELLITE_SOURCES,
) -> pd.DataFrame:
    """Fetch and merge all three VIIRS satellites over [start_date, end_date] for one bbox (area API).

    Recurrence features assume detections from all three satellites are present;
    fetching a single satellite (as fetch_firms does when given one --source)
    silently undercounts recurrence and truncates the date range at that one
    satellite's SP/NRT cutover. This is the working bulk-pull path today (see
    module docstring re: fetch_firms_country_multi) -- pass bbox=INDIA_BBOX for a
    national pull, and clip_boundary=backend.config.INDIA_BOUNDARY_PATH with it:
    INDIA_BBOX is a loose rectangle that also catches real fires in Myanmar, Sri
    Lanka, and Pakistan (47% of an unclipped one-month pull, in practice) --
    clip_boundary is what actually restricts detections to India's real shape.
    satellite_sources picks which FIRMS sources may serve each satellite, in
    preference order (default SP-then-NRT; NRT_SATELLITE_SOURCES for live pulls).
    """
    if end_date < start_date:
        raise ValueError("end_date must be on or after start_date")
    key = _resolve_map_key(map_key)
    bbox_str = _format_bbox(bbox)
    availability = _data_availability(key, timeout_seconds=timeout_seconds)
    total_days = (end_date - start_date).days + 1

    frames: list[pd.DataFrame] = []
    gaps: list[str] = []
    for satellite in satellites:
        source_names = satellite_sources[satellite]
        segments = _resolve_segments(source_names, start_date, end_date, availability)
        covered_days = sum((seg_end - seg_start).days + 1 for _, seg_start, seg_end in segments)
        if covered_days < total_days:
            gaps.append(f"{satellite}: only {covered_days} of {total_days} days covered by {source_names}")
        for source, seg_start, seg_end in segments:

            def url_builder(window_start: date, day_range: int, _source=source) -> str:
                return f"{API_ROOT}/area/csv/{key}/{_source}/{bbox_str}/{day_range}/{window_start.isoformat()}"

            frames.append(
                _fetch_range(url_builder, seg_start, seg_end, timeout_seconds=timeout_seconds, chunk_days=chunk_days)
            )

    if not frames:
        raise RuntimeError("No FIRMS source covers any part of the requested date range")

    combined = _finalize(frames, start_date, end_date, output_path, clip_boundary=clip_boundary)
    if gaps:
        print("Coverage gaps (satellite unavailable for part of the requested range):")
        for gap in gaps:
            print(f"  {gap}")
    return combined


def fetch_firms_country(
    country_code: str,
    start_date: date,
    end_date: date,
    output_path: str | Path,
    *,
    map_key: str | None = None,
    source: str = DEFAULT_SOURCE,
    timeout_seconds: int = 60,
    chunk_days: int = 5,
    clip_boundary: str | Path | None = None,
) -> pd.DataFrame:
    """Fetch FIRMS CSV data for one country from a single named source (country API).

    Implements FIRMS' documented /api/country/csv/{MAP_KEY}/{SOURCE}/{COUNTRY}/{DAY_RANGE}/{DATE}
    shape. As of this writing the live endpoint returns "Invalid API call" for
    every request regardless of parameters -- verified directly, and FIRMS' own
    /api/country/ docs page marks it "* Feature currently not available." This
    will raise ValueError("FIRMS rejected the request: Invalid API call.") until
    NASA re-enables it; that's expected, not a bug in this function. Use
    fetch_firms_multi(INDIA_BBOX, ...) / fetch_firms_country_multi (below) for a
    working national pull today. clip_boundary works the same as fetch_firms'.
    """
    if end_date < start_date:
        raise ValueError("end_date must be on or after start_date")
    key = _resolve_map_key(map_key)

    def url_builder(window_start: date, day_range: int) -> str:
        return f"{API_ROOT}/country/csv/{key}/{source}/{country_code}/{day_range}/{window_start.isoformat()}"

    frame = _fetch_range(url_builder, start_date, end_date, timeout_seconds=timeout_seconds, chunk_days=chunk_days)
    return _finalize([frame], start_date, end_date, output_path, clip_boundary=clip_boundary)


def fetch_firms_country_multi(
    country_code: str,
    start_date: date,
    end_date: date,
    output_path: str | Path,
    *,
    satellites: tuple[str, ...] = ("NOAA20", "NOAA21", "SNPP"),
    map_key: str | None = None,
    timeout_seconds: int = 60,
    chunk_days: int = 5,
    clip_boundary: str | Path | None = None,
) -> pd.DataFrame:
    """fetch_firms_multi's country-API equivalent -- see fetch_firms_country's docstring:
    currently unavailable upstream (NASA returns "Invalid API call"), included so the
    call is ready to use the moment FIRMS re-enables /api/country/, without another
    round of client changes.
    """
    if end_date < start_date:
        raise ValueError("end_date must be on or after start_date")
    key = _resolve_map_key(map_key)
    availability = _data_availability(key, timeout_seconds=timeout_seconds)
    total_days = (end_date - start_date).days + 1

    frames: list[pd.DataFrame] = []
    gaps: list[str] = []
    for satellite in satellites:
        source_names = SATELLITE_SOURCES[satellite]
        segments = _resolve_segments(source_names, start_date, end_date, availability)
        covered_days = sum((seg_end - seg_start).days + 1 for _, seg_start, seg_end in segments)
        if covered_days < total_days:
            gaps.append(f"{satellite}: only {covered_days} of {total_days} days covered by {source_names}")
        for source, seg_start, seg_end in segments:

            def url_builder(window_start: date, day_range: int, _source=source) -> str:
                return f"{API_ROOT}/country/csv/{key}/{_source}/{country_code}/{day_range}/{window_start.isoformat()}"

            frames.append(
                _fetch_range(url_builder, seg_start, seg_end, timeout_seconds=timeout_seconds, chunk_days=chunk_days)
            )

    if not frames:
        raise RuntimeError("No FIRMS source covers any part of the requested date range")
    combined = _finalize(frames, start_date, end_date, output_path, clip_boundary=clip_boundary)
    if gaps:
        print("Coverage gaps (satellite unavailable for part of the requested range):")
        for gap in gaps:
            print(f"  {gap}")
    return combined


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    location = parser.add_mutually_exclusive_group(required=True)
    location.add_argument("--bbox", nargs=4, type=float, metavar=("WEST", "SOUTH", "EAST", "NORTH"))
    location.add_argument("--country", help="ISO 3166-1 alpha-3 code, e.g. IND (currently unavailable upstream -- see module docstring)")
    parser.add_argument("--start", type=date.fromisoformat, required=True)
    parser.add_argument("--end", type=date.fromisoformat)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--source", default=None,
        help="Fetch one named FIRMS source only (e.g. VIIRS_NOAA20_NRT), skipping the multi-satellite merge",
    )
    parser.add_argument(
        "--satellites", default="NOAA20,NOAA21,SNPP",
        help="Comma-separated VIIRS satellites to merge when --source is not given",
    )
    parser.add_argument(
        "--clip-boundary", type=Path, default=None,
        help="GeoParquet polygon(s) to clip results to after fetching (e.g. backend/config.py's INDIA_BOUNDARY_PATH) -- bbox/country are both loose shapes otherwise",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    end_date = args.end or args.start + timedelta(days=30)
    satellites = tuple(s.strip() for s in args.satellites.split(",") if s.strip())
    if args.country:
        if args.source:
            result = fetch_firms_country(args.country, args.start, end_date, args.output, source=args.source, clip_boundary=args.clip_boundary)
        else:
            result = fetch_firms_country_multi(args.country, args.start, end_date, args.output, satellites=satellites, clip_boundary=args.clip_boundary)
    else:
        if args.source:
            result = fetch_firms(args.bbox, args.start, end_date, args.output, source=args.source, clip_boundary=args.clip_boundary)
        else:
            result = fetch_firms_multi(args.bbox, args.start, end_date, args.output, satellites=satellites, clip_boundary=args.clip_boundary)
    print(f"Saved {len(result):,} detections to {args.output}")
