"""Feature engineering for FIRMS hotspots: recurrence, anomaly, and spatial-context joins.

These functions compute the per-detection features documented in the README
("Contract column changes" / "recurrence_count, first_seen, last_seen, and
is_anomalous" / "industrial class membership") that build_labels.py joins onto
the raw FIRMS CSV before rules.apply_rules assigns a label.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.mask import mask as rasterio_mask
from shapely.geometry import mapping

from backend.config import ANOMALY_Z_THRESHOLD, GRID_SIZE, MIN_PRIOR_ACTIVE_DAYS


def grid_keys(frame: pd.DataFrame, grid_size: float = GRID_SIZE) -> tuple[pd.Series, pd.Series]:
    """Snap each detection's lat/lon onto a roughly 375 m grid for site-level grouping.

    Used instead of exact-coordinate grouping since VIIRS re-detections of the same
    physical site rarely land on identical lat/lon.
    """
    grid_lat = (frame["latitude"] / grid_size).round().astype("int64")
    grid_lon = (frame["longitude"] / grid_size).round().astype("int64")
    return grid_lat, grid_lon


def add_recurrence_features(frame: pd.DataFrame, grid_lat: pd.Series, grid_lon: pd.Series) -> pd.DataFrame:
    """Add recurrence_count, first_seen, last_seen: distinct active days per grid cell.

    Computed over the full input frame, including whatever date range it covers --
    correct for labelling, but callers who need pre-split (training-only) values
    must recompute these from a pre-split frame first (see README).
    """
    result = frame.copy()
    result["_grid_lat"] = grid_lat
    result["_grid_lon"] = grid_lon
    keys = ["_grid_lat", "_grid_lon"]
    result["recurrence_count"] = result.groupby(keys)["acq_date"].transform("nunique")
    result["first_seen"] = result.groupby(keys)["acq_date"].transform("min")
    result["last_seen"] = result.groupby(keys)["acq_date"].transform("max")
    return result.drop(columns=keys)


def compute_is_anomalous(
    frame: pd.DataFrame,
    grid_lat: pd.Series,
    grid_lon: pd.Series,
    *,
    min_prior_active_days: int = MIN_PRIOR_ACTIVE_DAYS,
    z_threshold: float = ANOMALY_Z_THRESHOLD,
) -> pd.Series:
    """Flag detections whose FRP is a robust outlier against that same site's own prior history.

    Baseline is the site's daily-max FRP, computed separately for day and night
    detections (VIIRS reads FRP differently under solar illumination), using only
    days strictly before the detection's own date -- so nothing here can leak
    forward. A site needs >=min_prior_active_days of history before it can be
    judged at all; until then every detection at that site is False, never True.
    Missing FRP is never flagged, and only increases over baseline count -- the
    modified z-score is one-sided (>z_threshold), so a drop in FRP is never
    "anomalous" by construction.
    """
    work = pd.DataFrame(
        {"_glat": grid_lat, "_glon": grid_lon, "daynight": frame["daynight"], "acq_date": frame["acq_date"], "frp": frame["frp"]},
        index=frame.index,
    )
    flags = pd.Series(False, index=frame.index)

    for _, group in work.groupby(["_glat", "_glon", "daynight"]):
        daily_max = group.groupby("acq_date")["frp"].max().sort_index()
        dates = daily_max.index.to_numpy()
        values = daily_max.to_numpy(dtype="float64")
        for idx, row in group.iterrows():
            frp = row["frp"]
            if pd.isna(frp):
                continue
            cutoff = int(np.searchsorted(dates, row["acq_date"], side="left"))
            prior = values[:cutoff]
            if prior.size < min_prior_active_days:
                continue
            median = float(np.median(prior))
            mad = float(np.median(np.abs(prior - median)))
            if mad == 0:
                anomalous = frp > 3 * median
            else:
                modified_z = 0.6745 * (frp - median) / mad
                anomalous = modified_z > z_threshold
            flags.loc[idx] = bool(anomalous)
    return flags


def nearest_facility_distance(
    detections: gpd.GeoDataFrame, facilities: gpd.GeoDataFrame, facility_types: tuple[str, ...]
) -> tuple[pd.Series, pd.Series]:
    """Distance (m) and facility_type of the nearest facility among the given types, per detection.

    Point-to-point nearest-neighbour against GEM's single lat/lon per facility, so
    it can pick the wrong point for a large multi-site complex (see README) --
    treat the result as "nearest tagged facility of this type," not "distance to
    the site boundary."
    """
    subset = facilities[facilities["facility_type"].isin(facility_types)]
    if subset.empty:
        empty = pd.Series(None, index=detections.index)
        return empty.astype("float64"), empty.astype("object")
    joined = gpd.sjoin_nearest(detections[["geometry"]], subset[["geometry", "facility_type"]], distance_col="_dist")
    joined = joined.reset_index(names="_detection_idx").sort_values("_dist")
    nearest = joined.drop_duplicates(subset="_detection_idx", keep="first").set_index("_detection_idx")
    return nearest["_dist"].reindex(detections.index), nearest["facility_type"].reindex(detections.index)


def landcover_majority(points_3857: gpd.GeoSeries, raster_path: str | Path, buffer_m: float = 375) -> pd.Series:
    """Majority ESA WorldCover class code within buffer_m metres of each point.

    Left null (never fabricated) when the raster has no data at a point -- callers
    who omit --worldcover entirely get the same null, which silently disables the
    agricultural burning and wildfire rules per the README.
    """
    with rasterio.open(raster_path) as raster:
        nodata = raster.nodata
        buffered = gpd.GeoSeries(points_3857.buffer(buffer_m), crs="EPSG:3857").to_crs(raster.crs)
        classes: list[str | None] = []
        for geom in buffered:
            try:
                image, _ = rasterio_mask(raster, [mapping(geom)], crop=True)
            except ValueError:
                classes.append(None)
                continue
            values = image[0]
            values = values[values != 0]
            if nodata is not None:
                values = values[values != nodata]
            if values.size == 0:
                classes.append(None)
                continue
            classes.append(str(Counter(values.tolist()).most_common(1)[0][0]))
    return pd.Series(classes, index=points_3857.index)
