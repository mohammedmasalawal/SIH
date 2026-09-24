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
    return anomaly_baselines(
        frame, grid_lat, grid_lon, min_prior_active_days=min_prior_active_days, z_threshold=z_threshold
    )["is_anomalous"]


def anomaly_baselines(
    frame: pd.DataFrame,
    grid_lat: pd.Series,
    grid_lon: pd.Series,
    *,
    targets: pd.Series | None = None,
    min_prior_active_days: int = MIN_PRIOR_ACTIVE_DAYS,
    z_threshold: float = ANOMALY_Z_THRESHOLD,
) -> pd.DataFrame:
    """compute_is_anomalous's per-row detail: is_anomalous plus the baseline it was
    judged against (baseline_frp = median prior daily-max FRP for that site and
    day/night, prior_active_days = how many prior days that median covers; both NaN
    when the site has too little history). `targets` (bool mask over frame) limits
    evaluation to those rows -- the rest still serve as history but are returned
    False/NaN -- so a small batch can be judged against a large history cheaply.
    """
    work = pd.DataFrame(
        {"_glat": grid_lat, "_glon": grid_lon, "daynight": frame["daynight"], "acq_date": frame["acq_date"], "frp": frame["frp"]},
        index=frame.index,
    )
    if targets is None:
        targets = pd.Series(True, index=frame.index)
    out = pd.DataFrame(
        {"is_anomalous": False, "baseline_frp": np.nan, "prior_active_days": np.nan}, index=frame.index
    )
    work["_target"] = targets.to_numpy()
    site = ["_glat", "_glon", "daynight"]
    target_sites = pd.MultiIndex.from_frame(work.loc[work["_target"], site].drop_duplicates())
    work = work[pd.MultiIndex.from_frame(work[site]).isin(target_sites)]  # only sites with a target row

    for _, group in work.groupby(site):
        daily_max = group.groupby("acq_date")["frp"].max().sort_index()
        dates = daily_max.index.to_numpy()
        values = daily_max.to_numpy(dtype="float64")
        for idx, row in group.iterrows():
            if not row["_target"]:
                continue
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
            out.loc[idx] = (bool(anomalous), median, prior.size)
    out["is_anomalous"] = out["is_anomalous"].astype(bool)
    return out


def add_causal_recurrence_features(
    frame: pd.DataFrame,
    grid_lat: pd.Series,
    grid_lon: pd.Series,
    targets: pd.Series,
    *,
    lookback_days: int | None = None,
) -> pd.DataFrame:
    """recurrence_count / first_seen / last_seen for the `targets` rows, counting only
    that site's active days on or before each row's own date (earlier detections
    only -- never later ones, unlike add_recurrence_features, which counts the
    whole frame). The row's own day counts, so a first-ever detection gets 1, same
    as the batch definition. lookback_days, if given, also drops days more than that
    many days before the row's date. Non-target rows are returned unchanged.
    """
    result = frame.copy()
    days = pd.DataFrame({"_glat": grid_lat, "_glon": grid_lon, "acq_date": pd.to_datetime(frame["acq_date"])})
    target_rows = days[targets.to_numpy()]
    cells = target_rows[["_glat", "_glon"]].drop_duplicates()
    active = days.merge(cells, on=["_glat", "_glon"]).drop_duplicates().sort_values("acq_date")
    by_cell = {key: grp["acq_date"].to_numpy() for key, grp in active.groupby(["_glat", "_glon"])}

    counts, firsts = [], []
    for glat, glon, when in zip(target_rows["_glat"], target_rows["_glon"], target_rows["acq_date"]):
        cell_days = by_cell[(glat, glon)]
        end = int(np.searchsorted(cell_days, np.datetime64(when), side="right"))
        start = 0
        if lookback_days is not None:
            start = int(np.searchsorted(cell_days, np.datetime64(when - pd.Timedelta(days=lookback_days)), side="left"))
        counts.append(end - start)
        firsts.append(pd.Timestamp(cell_days[start]).strftime("%Y-%m-%d"))

    for column in ("recurrence_count", "first_seen", "last_seen"):
        if column not in result:
            result[column] = pd.Series(pd.NA, index=result.index, dtype="object")
    mask = targets.to_numpy()
    result.loc[mask, "recurrence_count"] = counts
    result.loc[mask, "first_seen"] = firsts
    result.loc[mask, "last_seen"] = target_rows["acq_date"].dt.strftime("%Y-%m-%d").to_numpy()
    return result


def osm_industrial_tag(detections: gpd.GeoDataFrame, industrial: gpd.GeoDataFrame) -> pd.Series:
    """The OSM `industrial=*` tag of the polygon each detection falls inside, lowercased.

    Plain `landuse=industrial` polygons (no `industrial` subtype tag) fall back to
    "yes" -- present but not the refinery subtype, which is all the label rules need.
    When a point falls inside overlapping polygons, a refinery match wins the tie.
    """
    polygons = industrial[industrial.geometry.geom_type.isin(["Polygon", "MultiPolygon"])].copy()
    result = pd.Series(None, index=detections.index, dtype="object")
    if polygons.empty:
        return result
    polygons["_tag"] = (
        polygons["industrial"].fillna("yes").astype(str).str.lower() if "industrial" in polygons.columns else "yes"
    )
    joined = gpd.sjoin(detections[["geometry"]], polygons[["geometry", "_tag"]], predicate="within", how="inner")
    joined = joined.sort_values("_tag", key=lambda tags: tags != "refinery")
    joined = joined[~joined.index.duplicated(keep="first")]
    result.loc[joined.index] = joined["_tag"]
    return result


def nearest_industrial_distance(detections: gpd.GeoDataFrame, industrial: gpd.GeoDataFrame) -> pd.Series:
    """Distance (m) to the nearest OSM industrial-context feature, per detection.

    Uses gpd.sjoin_nearest against the individual industrial features (spatial-
    indexed), the same approach nearest_facility_distance uses for GEM facilities --
    not a single detections.geometry.distance(industrial.union_all()) call against
    one unioned mega-geometry, which isn't index-accelerated. On a national-scale
    benchmark (13,120 detections x 37,908 OSM features) the union+distance approach
    took 35.9s of a 46.5s total join time; this replacement is the direct fix (see
    training/benchmark_joins.py, re-run after this change: 0.115s of a 0.622s total
    on the same context re-timed against a clipped one-month pull). Distance to a
    polygon the point is inside is 0 either way -- the two approaches are
    mathematically equivalent, only the second is spatial-indexed.
    """
    if industrial.empty:
        return pd.Series(None, index=detections.index, dtype="float64")
    industrial.sindex  # force spatial-index construction before the timed join
    joined = gpd.sjoin_nearest(detections[["geometry"]], industrial[["geometry"]], distance_col="_dist")
    joined = joined.reset_index(names="_detection_idx").sort_values("_dist")
    nearest = joined.drop_duplicates(subset="_detection_idx", keep="first").set_index("_detection_idx")
    return nearest["_dist"].reindex(detections.index)


def _nearest_feature_distance(detections: gpd.GeoDataFrame, features: gpd.GeoDataFrame) -> pd.Series:
    """Shared sjoin_nearest helper: distance (m) to the nearest row in `features`,
    per detection, or all-null if `features` is empty. Same spatial-indexed
    approach as nearest_industrial_distance/nearest_facility_distance, factored out
    since nearest_coal_mine_distance needs it against three different feature sets.
    """
    if features.empty:
        return pd.Series(None, index=detections.index, dtype="float64")
    features.sindex
    joined = gpd.sjoin_nearest(detections[["geometry"]], features[["geometry"]], distance_col="_dist")
    joined = joined.reset_index(names="_detection_idx").sort_values("_dist")
    nearest = joined.drop_duplicates(subset="_detection_idx", keep="first").set_index("_detection_idx")
    return nearest["_dist"].reindex(detections.index)


def nearest_brick_kiln_distance(detections: gpd.GeoDataFrame, osm_brick_kilns: gpd.GeoDataFrame) -> pd.Series:
    """Distance (m) to the nearest OSM industrial=brickyard/man_made=kiln feature,
    per detection -- OSM-only, no GEM equivalent (see load_osm_brick_kilns)."""
    return _nearest_feature_distance(detections, osm_brick_kilns)


def nearest_coal_mine_distance(
    detections: gpd.GeoDataFrame,
    gem_coal_boundaries: gpd.GeoDataFrame,
    gem_coal_points: gpd.GeoDataFrame,
    osm_mines: gpd.GeoDataFrame,
) -> tuple[pd.Series, pd.Series]:
    """Distance (m) to the nearest coal-mine evidence, per detection, plus which
    source supplied it ("gem", "osm", or NaN if neither has any data at all).

    GEM (gem_coal_boundaries + gem_coal_points -- callers should pass coal_mine
    facilities of every status except backend.config.GEM_COAL_STATUSES_EXCLUDED;
    closed and mothballed mines are deliberately kept in scope, since seam fires
    outlive active mining) is checked first and wins whenever it has ANY data at
    all, regardless of whether OSM's match happens to be closer for a given
    detection -- OSM's industrial=mine tag isn't coal-specific and carries no
    status, so it's weaker evidence, used only as a fallback for detections GEM's
    coal-mine data can't speak to (as of this writing, that's every detection: see
    GEM_COAL_MINE_BOUNDARIES_PATH / GEM_COAL_MINE_CSV_PATH in backend/config.py --
    neither file has real coal-mine content yet, so nearest_coal_source will read
    "osm" everywhere until one or both are actually supplied).

    gem_coal_points' distance is adjusted by each mine's area_km2, treating it as a
    circle of that area centred on the point and measuring to the circle's edge
    rather than its centre (effective_distance = max(0, point_distance - radius)) --
    an approximation of "distance to the boundary" for mines GEM only has a point
    for. gem_coal_boundaries, where available, is used directly (real boundary
    geometry, no area adjustment needed). The smaller of the two per detection is
    used as the GEM distance, which in practice prefers the boundary-based figure
    whenever a mine has one (a real boundary is almost always tighter than a circle
    approximation of the same mine).
    """
    gem_boundary_dist = _nearest_feature_distance(detections, gem_coal_boundaries)

    if gem_coal_points.empty or "area_km2" not in gem_coal_points.columns:
        gem_point_dist = pd.Series(None, index=detections.index, dtype="float64")
    else:
        area_km2 = pd.to_numeric(gem_coal_points["area_km2"], errors="coerce").fillna(0.0)
        radius_m = np.sqrt(area_km2.to_numpy() * 1_000_000 / np.pi)  # circle-of-area_km2's radius, in metres
        points_with_radius = gem_coal_points.assign(_radius_m=radius_m)
        points_with_radius.sindex
        joined = gpd.sjoin_nearest(
            detections[["geometry"]], points_with_radius[["geometry", "_radius_m"]], distance_col="_dist"
        )
        joined = joined.reset_index(names="_detection_idx").sort_values("_dist")
        nearest = joined.drop_duplicates(subset="_detection_idx", keep="first").set_index("_detection_idx")
        point_dist = nearest["_dist"].reindex(detections.index)
        radius = nearest["_radius_m"].reindex(detections.index)
        gem_point_dist = (point_dist - radius).clip(lower=0)

    gem_dist = pd.concat([gem_boundary_dist, gem_point_dist], axis=1).min(axis=1, skipna=True)
    osm_dist = _nearest_feature_distance(detections, osm_mines)

    dist = gem_dist.where(gem_dist.notna(), osm_dist)
    source = pd.Series(pd.NA, index=detections.index, dtype="object")
    source.loc[gem_dist.notna()] = "gem"
    source.loc[gem_dist.isna() & osm_dist.notna()] = "osm"
    return dist, source


def nearest_facility_distance(
    detections: gpd.GeoDataFrame, facilities: gpd.GeoDataFrame, facility_types: tuple[str, ...]
) -> tuple[pd.Series, pd.Series]:
    """Distance (m) and facility_type of the nearest facility among the given types, per detection.

    Point-to-point nearest-neighbour against GEM's single lat/lon per facility, so
    it can pick the wrong point for a large multi-site complex (see README) --
    treat the result as "nearest tagged facility of this type," not "distance to
    the site boundary." Distances are only meaningful in a projected (metric) CRS --
    callers must reproject both detections and facilities before calling this (see
    training/build_labels.py, which uses EPSG:7755 for India-wide accuracy).

    subset.sindex is built explicitly before the join, not left to be built lazily
    on gpd.sjoin_nearest's first internal query -- geopandas' sjoin_nearest always
    uses a spatial index either way, but building it up front here means a caller
    timing this call (see training/benchmark_joins.py) measures the join itself,
    not index construction folded into the first call.
    """
    subset = facilities[facilities["facility_type"].isin(facility_types)]
    if subset.empty:
        empty = pd.Series(None, index=detections.index)
        return empty.astype("float64"), empty.astype("object")
    subset.sindex  # force spatial-index construction before the timed join
    joined = gpd.sjoin_nearest(detections[["geometry"]], subset[["geometry", "facility_type"]], distance_col="_dist")
    joined = joined.reset_index(names="_detection_idx").sort_values("_dist")
    nearest = joined.drop_duplicates(subset="_detection_idx", keep="first").set_index("_detection_idx")
    return nearest["_dist"].reindex(detections.index), nearest["facility_type"].reindex(detections.index)


def landcover_majority(points_projected: gpd.GeoSeries, raster_path: str | Path, buffer_m: float = 375) -> pd.Series:
    """Majority ESA WorldCover class code within buffer_m metres of each point.

    points_projected must already be in a projected (metric) CRS -- buffer_m is
    interpreted in that CRS's units. Reprojects the buffered geometries from
    points_projected's own CRS (points_projected.crs), not a hardcoded one, so this
    works the same whether the caller used EPSG:3857 or an India-wide CRS like
    EPSG:7755. (An earlier version hardcoded crs="EPSG:3857" here regardless of the
    input's actual CRS -- silently wrong the moment a caller passed anything else.)

    Left null (never fabricated) when the raster has no data at a point -- callers
    who omit --worldcover entirely get the same null, which silently disables the
    agricultural burning and wildfire rules per the README. For a national-scale
    run spanning many WorldCover tiles, prefer
    training.worldcover_index.WorldCoverTileIndex.majority_class_in_buffer instead
    of this function, which expects one single raster covering every point.
    """
    with rasterio.open(raster_path) as raster:
        nodata = raster.nodata
        buffered = gpd.GeoSeries(points_projected.buffer(buffer_m), crs=points_projected.crs).to_crs(raster.crs)
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
    return pd.Series(classes, index=points_projected.index)
