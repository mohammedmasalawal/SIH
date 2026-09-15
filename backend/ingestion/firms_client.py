"""NASA FIRMS VIIRS client with explicit bbox/date controls."""

from __future__ import annotations

import argparse
import os
from datetime import date, timedelta
from pathlib import Path
from typing import Iterable

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

# Each VIIRS satellite has its own archive (SP, science-quality) vs near-real-time
# (NRT) cutover date, and it moves as NRT rolls forward and SP reprocessing catches
# up -- NOAA-21 currently has no separate SP source at all. Resolved dynamically
# against data_availability rather than hardcoded so this doesn't go stale.
SATELLITE_SOURCES: dict[str, tuple[str, ...]] = {
    "NOAA20": ("VIIRS_NOAA20_SP", "VIIRS_NOAA20_NRT"),
    "NOAA21": ("VIIRS_NOAA21_NRT",),
    "SNPP": ("VIIRS_SNPP_SP", "VIIRS_SNPP_NRT"),
}


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


def _fetch_chunk(
    bbox_str: str,
    window_start: date,
    day_range: int,
    *,
    key: str,
    source: str,
    timeout_seconds: int,
) -> str:
    url = f"{API_ROOT}/area/csv/{key}/{source}/{bbox_str}/{day_range}/{window_start.isoformat()}"
    response = requests.get(url, timeout=timeout_seconds)
    response.raise_for_status()
    text = response.text.strip()
    if not text:
        raise ValueError("FIRMS returned an empty response")
    if text.startswith("Invalid") or "error" in text.lower()[:20]:
        raise ValueError(f"FIRMS rejected the request: {text}")
    return text


def _fetch_range(
    bbox_str: str,
    start_date: date,
    end_date: date,
    *,
    key: str,
    source: str,
    timeout_seconds: int,
    chunk_days: int,
) -> pd.DataFrame:
    """Fetch [start_date, end_date] from a single named source, walked forward in chunks."""
    chunks: list[str] = []
    chunk_start = start_date
    while chunk_start <= end_date:
        chunk_end = min(end_date, chunk_start + timedelta(days=chunk_days - 1))
        day_range = (chunk_end - chunk_start).days + 1
        chunks.append(
            _fetch_chunk(
                bbox_str, chunk_start, day_range, key=key, source=source, timeout_seconds=timeout_seconds
            )
        )
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
) -> pd.DataFrame:
    """Fetch FIRMS CSV data from a single named source and persist the raw response.

    For routine pulls prefer fetch_firms_multi, which merges all three VIIRS
    satellites with the correct SP/NRT source per sub-range. Use this directly only
    when you deliberately want one named source (e.g. debugging a single satellite).
    """
    if end_date < start_date:
        raise ValueError("end_date must be on or after start_date")
    key = map_key or os.getenv("FIRMS_MAP_KEY")
    if not key:
        raise RuntimeError("Set FIRMS_MAP_KEY or pass map_key explicitly")
    bbox_str = _format_bbox(bbox)

    frame = _fetch_range(
        bbox_str, start_date, end_date, key=key, source=source, timeout_seconds=timeout_seconds, chunk_days=chunk_days
    )
    validate_firms_columns(frame)
    frame = frame[(frame["acq_date"] >= start_date.isoformat()) & (frame["acq_date"] <= end_date.isoformat())]
    frame = frame.drop_duplicates().sort_values(["acq_date", "acq_time"]).reset_index(drop=True)
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(destination, index=False)
    return frame


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
) -> pd.DataFrame:
    """Fetch and merge all three VIIRS satellites over [start_date, end_date].

    Recurrence features assume detections from all three satellites are present;
    fetching a single satellite (as fetch_firms does when given one --source)
    silently undercounts recurrence and truncates the date range at that one
    satellite's SP/NRT cutover.
    """
    if end_date < start_date:
        raise ValueError("end_date must be on or after start_date")
    key = map_key or os.getenv("FIRMS_MAP_KEY")
    if not key:
        raise RuntimeError("Set FIRMS_MAP_KEY or pass map_key explicitly")
    bbox_str = _format_bbox(bbox)
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
            frames.append(
                _fetch_range(
                    bbox_str, seg_start, seg_end, key=key, source=source,
                    timeout_seconds=timeout_seconds, chunk_days=chunk_days,
                )
            )

    if not frames:
        raise RuntimeError("No FIRMS source covers any part of the requested date range")

    combined = pd.concat(frames, ignore_index=True)
    validate_firms_columns(combined)
    combined = combined[
        (combined["acq_date"] >= start_date.isoformat()) & (combined["acq_date"] <= end_date.isoformat())
    ]
    combined = combined.drop_duplicates().sort_values(["acq_date", "acq_time", "satellite"]).reset_index(drop=True)
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(destination, index=False)
    if gaps:
        print("Coverage gaps (satellite unavailable for part of the requested range):")
        for gap in gaps:
            print(f"  {gap}")
    return combined


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bbox", nargs=4, type=float, required=True, metavar=("WEST", "SOUTH", "EAST", "NORTH"))
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
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    end_date = args.end or args.start + timedelta(days=30)
    if args.source:
        result = fetch_firms(args.bbox, args.start, end_date, args.output, source=args.source)
    else:
        satellites = tuple(s.strip() for s in args.satellites.split(",") if s.strip())
        result = fetch_firms_multi(args.bbox, args.start, end_date, args.output, satellites=satellites)
    print(f"Saved {len(result):,} detections to {args.output}")
