"""Join FIRMS detections to context and create the teammate-2 hand-off CSV."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.mask import mask as rasterio_mask
from shapely.geometry import mapping

from rules import FLARE_CAPABLE_FACILITY_TYPES, HEAT_INDUSTRY_FACILITY_TYPES, apply_rules

CONTRACT_COLUMNS = [
    "latitude", "longitude", "acq_date", "acq_time", "satellite", "bright_ti4", "bright_ti5", "frp",
    "confidence", "daynight", "dist_to_industrial_m",
    "dist_to_flare_capable_m", "nearest_flare_facility_type",
    "dist_to_heat_industry_m", "nearest_heat_facility_type",
    "landcover_class", "recurrence_count", "first_seen", "last_seen",
    "is_anomalous", "label", "label_source",
]

GRID_SIZE = 0.0034  # roughly 375 m at this latitude


def _grid_keys(frame: pd.DataFrame, grid_size: float = GRID_SIZE) -> tuple[pd.Series, pd.Series]:
    grid_lat = (frame["latitude"] / grid_size).round().astype("int64")
    grid_lon = (frame["longitude"] / grid_size).round().astype("int64")
    return grid_lat, grid_lon


def _recurrence(frame: pd.DataFrame, grid_lat: pd.Series, grid_lon: pd.Series) -> pd.DataFrame:
    """Count distinct active days in a roughly 375 m lat/lon grid, avoiding exact-coordinate grouping."""
    result = frame.copy()
    result["_grid_lat"] = grid_lat
    result["_grid_lon"] = grid_lon
    keys = ["_grid_lat", "_grid_lon"]
    result["recurrence_count"] = result.groupby(keys)["acq_date"].transform("nunique")
    result["first_seen"] = result.groupby(keys)["acq_date"].transform("min")
    result["last_seen"] = result.groupby(keys)["acq_date"].transform("max")
    return result.drop(columns=keys)


def _is_anomalous(
    frame: pd.DataFrame,
    grid_lat: pd.Series,
    grid_lon: pd.Series,
    *,
    min_prior_active_days: int = 5,
    z_threshold: float = 3.5,
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


def _nearest_facility(
    detections: gpd.GeoDataFrame, facilities: gpd.GeoDataFrame, facility_types: tuple[str, ...]
) -> tuple[pd.Series, pd.Series]:
    """Distance and facility_type of the nearest facility among the given types, per detection."""
    subset = facilities[facilities["facility_type"].isin(facility_types)]
    if subset.empty:
        empty = pd.Series(None, index=detections.index)
        return empty.astype("float64"), empty.astype("object")
    joined = gpd.sjoin_nearest(detections[["geometry"]], subset[["geometry", "facility_type"]], distance_col="_dist")
    joined = joined.reset_index(names="_detection_idx").sort_values("_dist")
    nearest = joined.drop_duplicates(subset="_detection_idx", keep="first").set_index("_detection_idx")
    return nearest["_dist"].reindex(detections.index), nearest["facility_type"].reindex(detections.index)


def _landcover_majority(points_3857: gpd.GeoSeries, raster_path: str | Path, buffer_m: float = 375) -> pd.Series:
    """Majority ESA WorldCover class code within buffer_m metres of each point."""
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


def build_labels(
    firms_csv: str | Path,
    industrial_geojson: str | Path,
    facilities_geojson: str | Path,
    output_csv: str | Path,
    *,
    worldcover_raster: str | Path | None = None,
) -> pd.DataFrame:
    firms = pd.read_csv(firms_csv)
    industrial = gpd.read_file(industrial_geojson).to_crs("EPSG:3857")
    facilities = gpd.read_file(facilities_geojson).to_crs("EPSG:3857")
    detections = gpd.GeoDataFrame(
        firms,
        geometry=gpd.points_from_xy(firms["longitude"], firms["latitude"]),
        crs="EPSG:4326",
    ).to_crs("EPSG:3857")

    industrial_union = industrial.geometry.union_all() if not industrial.empty else None
    result = firms.copy()
    result["dist_to_industrial_m"] = (
        detections.geometry.distance(industrial_union).values if industrial_union is not None else None
    )
    flare_dist, flare_type = _nearest_facility(detections, facilities, FLARE_CAPABLE_FACILITY_TYPES)
    heat_dist, heat_type = _nearest_facility(detections, facilities, HEAT_INDUSTRY_FACILITY_TYPES)
    result["dist_to_flare_capable_m"] = flare_dist.values
    result["nearest_flare_facility_type"] = flare_type.values
    result["dist_to_heat_industry_m"] = heat_dist.values
    result["nearest_heat_facility_type"] = heat_type.values

    grid_lat, grid_lon = _grid_keys(result)
    result = _recurrence(result, grid_lat, grid_lon)
    result["is_anomalous"] = _is_anomalous(result, grid_lat, grid_lon).values

    if worldcover_raster is not None:
        result["landcover_class"] = _landcover_majority(detections.geometry, worldcover_raster).values
    else:
        result["landcover_class"] = result.get("landcover_class", pd.Series(index=result.index, dtype="object"))

    result["label"] = "unknown"
    result["label_source"] = "unclassified"
    result = apply_rules(result)
    for column in CONTRACT_COLUMNS:
        if column not in result:
            result[column] = None
    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result[CONTRACT_COLUMNS].to_csv(output_path, index=False)
    return result[CONTRACT_COLUMNS]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--firms", type=Path, required=True)
    parser.add_argument("--industrial", type=Path, required=True)
    parser.add_argument("--facilities", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--worldcover", type=Path, default=None, help="ESA WorldCover GeoTIFF tile")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    build_labels(args.firms, args.industrial, args.facilities, args.output, worldcover_raster=args.worldcover)
