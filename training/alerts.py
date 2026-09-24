"""Alert types for live ingestion. Alerts flag detections worth a look; they never
change a detection's label. Thresholds live in backend/config.py and are untested
starting values (see README "Live alerts").

    industrial_anomaly        is_anomalous detection labelled industrial / gas flare
    fire_near_infrastructure  crop/natural fire within INFRA_FIRE_MAX_DIST_M of energy infrastructure
    new_unmapped_source       cell with no known facility nearby that started burning repeatedly
    large_fire_event          same-day cluster of LARGE_FIRE_MIN_DETECTIONS+ within LARGE_FIRE_RADIUS_M

Every alert has an alert_id that stays the same across runs, so re-evaluating a
window never raises the same alert twice.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN

from backend.config import (
    GAS_FLARE_MAX_DIST_M,
    INDUSTRIAL_HEAT_MAX_DIST_M,
    INFRA_FIRE_LABELS,
    INFRA_FIRE_MAX_DIST_M,
    INFRA_GEM_FACILITY_TYPES,
    INFRA_INCLUDE_OSM_POWER_PLANTS,
    INFRA_OSM_INDUSTRIAL_TAGS,
    LARGE_FIRE_MIN_DETECTIONS,
    LARGE_FIRE_RADIUS_M,
    NATIONAL_PROJECTED_CRS,
    UNMAPPED_MAX_PRIOR_ACTIVE_DAYS,
    UNMAPPED_MIN_ACTIVE_DAYS,
    UNMAPPED_NO_FACILITY_WITHIN_M,
    UNMAPPED_WINDOW_DAYS,
)
from training.spatial_features import grid_keys

ALERT_TYPES = ("industrial_anomaly", "fire_near_infrastructure", "new_unmapped_source", "large_fire_event")
ALERT_COLUMNS = [
    "alert_id", "alert_type",
    "latitude", "longitude", "acq_date", "acq_time", "satellite", "daynight", "label", "frp",
    "site_normal_frp", "frp_vs_normal", "prior_active_days",          # industrial_anomaly
    "facility_type", "facility_name", "facility_distance_m",          # industrial_anomaly, fire_near_infrastructure
    "active_days_30d", "recurrence_count", "first_seen",              # new_unmapped_source
    "event_count", "dominant_class",                                  # large_fire_event
    "maps_link",
]
ANOMALY_LABELS = ("industrial", "gas flare")
_FACILITY_DISTANCES = (
    "dist_to_industrial_m", "dist_to_heat_industry_m", "dist_to_flare_capable_m",
    "dist_to_coal_mine_m", "dist_to_brick_kiln_m",
)
_GEM_KIND = {
    "oil_gas_power_plant": "power plant (oil/gas)", "coal_power_plant": "power plant (coal)",
    "oil_gas_field": "oil/gas field", "lng_terminal": "LNG terminal",
}


def detection_keys(frame: pd.DataFrame) -> pd.Series:
    """One string per detection over (lat, lon, acq_date, acq_time, satellite), normalised
    across stored dtypes (acq_time int vs zero-padded str; FIRMS' 5-decimal coordinates)."""
    if frame.empty:
        return pd.Series([], index=frame.index, dtype="object")
    acq_time = frame["acq_time"].astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(4)
    return (
        frame["latitude"].astype(float).round(5).map("{:.5f}".format)
        + "|" + frame["longitude"].astype(float).round(5).map("{:.5f}".format)
        + "|" + frame["acq_date"].astype(str).str[:10]
        + "|" + acq_time
        + "|" + frame["satellite"].astype(str)
    )


def _text(value) -> str | None:
    return value if isinstance(value, str) and value and value.lower() != "nan" else None


def _finish(frame: pd.DataFrame, alert_type: str, ids: pd.Series) -> pd.DataFrame:
    out = frame.copy()
    out["alert_type"] = alert_type
    out["alert_id"] = alert_type + "|" + ids.astype(str).to_numpy()
    out["maps_link"] = [f"https://www.google.com/maps?q={lat},{lon}&t=k" for lat, lon in zip(out["latitude"], out["longitude"])]
    return out.reindex(columns=ALERT_COLUMNS)


def _empty() -> pd.DataFrame:
    return pd.DataFrame(columns=ALERT_COLUMNS)


# --- industrial_anomaly (unchanged selection and fields) -----------------------------

def _site_evidence(row: pd.Series) -> tuple[str, float | None]:
    """The facility that makes this an industrial/flare site, and its distance."""
    flare_type = _text(row.get("nearest_flare_facility_type")) or "flare-capable facility"
    if row["label"] == "gas flare":
        return flare_type, row.get("dist_to_flare_capable_m")
    heat = row.get("dist_to_heat_industry_m")
    if pd.notna(heat) and heat <= INDUSTRIAL_HEAT_MAX_DIST_M:
        return _text(row.get("nearest_heat_facility_type")) or "heat industry", heat
    flare = row.get("dist_to_flare_capable_m")
    if pd.notna(flare) and flare <= GAS_FLARE_MAX_DIST_M:
        return flare_type, flare
    if tag := _text(row.get("osm_industrial_tag")):
        return f"OSM industrial={tag}", row.get("dist_to_industrial_m")
    if source := _text(row.get("nearest_coal_source")):
        return f"coal mine ({source})", row.get("dist_to_coal_mine_m")
    return "OSM industrial area", row.get("dist_to_industrial_m")


def industrial_anomaly_alerts(batch: pd.DataFrame) -> pd.DataFrame:
    """Anomalous detections at industrial / gas-flare sites (needs batch baseline_frp,
    prior_active_days from anomaly_baselines)."""
    if batch.empty:
        return _empty()
    hits = batch[batch["is_anomalous"].astype(bool) & batch["label"].isin(ANOMALY_LABELS)].copy()
    if hits.empty:
        return _empty()
    evidence = hits.apply(_site_evidence, axis=1, result_type="expand")
    hits["facility_type"] = evidence[0]
    hits["facility_distance_m"] = pd.to_numeric(evidence[1], errors="coerce").round(0)
    hits["site_normal_frp"] = hits["baseline_frp"].round(2)
    hits["frp_vs_normal"] = (hits["frp"] / hits["baseline_frp"]).round(2)
    hits["prior_active_days"] = hits["prior_active_days"].astype("Int64")
    return _finish(hits, "industrial_anomaly", detection_keys(hits))


# --- fire_near_infrastructure ---------------------------------------------------------

def _column(frame: pd.DataFrame, name: str) -> pd.Series:
    """frame[name], or all-missing if the extract has no such tag at all."""
    return frame[name] if name in frame else pd.Series(None, index=frame.index, dtype="object")


def load_infrastructure_sites(industrial_path: Path, facilities_path: Path) -> gpd.GeoDataFrame:
    """Refineries, power plants, LNG terminals and oil/gas fields from GEM and OSM, with
    a display name and kind, in NATIONAL_PROJECTED_CRS. Reads the raw OSM extract (not
    the solar/wind-excluded industrial context), so solar and wind plants count."""
    gem = gpd.read_file(facilities_path) if Path(facilities_path).suffix != ".parquet" else gpd.read_parquet(facilities_path)
    gem = gem[gem["facility_type"].isin(INFRA_GEM_FACILITY_TYPES)]
    gem_sites = gpd.GeoDataFrame({
        "facility_name": _column(gem, "facility_name").fillna("unnamed").to_numpy(),
        "facility_type": gem["facility_type"].map(_GEM_KIND).to_numpy(),
        "source": "GEM",
    }, geometry=gem.geometry.to_numpy(), crs=gem.crs)

    osm = gpd.read_parquet(industrial_path) if Path(industrial_path).suffix == ".parquet" else gpd.read_file(industrial_path)
    tags = _column(osm, "industrial")
    power = (_column(osm, "power") == "plant") & INFRA_INCLUDE_OSM_POWER_PLANTS
    keep = tags.isin(INFRA_OSM_INDUSTRIAL_TAGS) | power
    osm, tags, power = osm[keep], tags[keep], power[keep]
    kind = np.where(
        power,
        "power plant (" + _column(osm, "plant:source").fillna("unknown source").astype(str) + ")",
        np.where(tags == "refinery", "refinery", "oil/gas (" + tags.astype(str) + ")"),
    )
    osm_sites = gpd.GeoDataFrame({
        "facility_name": _column(osm, "name").fillna("unnamed").to_numpy(),
        "facility_type": kind,
        "source": "OSM",
    }, geometry=osm.geometry.to_numpy(), crs=osm.crs)

    sites = pd.concat([gem_sites.to_crs(NATIONAL_PROJECTED_CRS), osm_sites.to_crs(NATIONAL_PROJECTED_CRS)], ignore_index=True)
    return gpd.GeoDataFrame(sites, geometry="geometry", crs=NATIONAL_PROJECTED_CRS)


def fire_near_infrastructure_alerts(batch: pd.DataFrame, sites: gpd.GeoDataFrame) -> pd.DataFrame:
    if batch.empty or sites.empty:
        return _empty()
    fires = batch[batch["label"].isin(INFRA_FIRE_LABELS)]
    if fires.empty:
        return _empty()
    points = gpd.GeoDataFrame(
        fires, geometry=gpd.points_from_xy(fires["longitude"], fires["latitude"]), crs="EPSG:4326"
    ).to_crs(NATIONAL_PROJECTED_CRS)
    joined = gpd.sjoin_nearest(
        points[["geometry"]], sites, max_distance=INFRA_FIRE_MAX_DIST_M, distance_col="facility_distance_m"
    )
    joined = joined.sort_values("facility_distance_m")
    joined = joined[~joined.index.duplicated(keep="first")]
    hits = fires.loc[joined.index].copy()
    hits["facility_name"] = joined["facility_name"]
    hits["facility_type"] = joined["facility_type"] + " · " + joined["source"]
    hits["facility_distance_m"] = joined["facility_distance_m"].round(0)
    return _finish(hits, "fire_near_infrastructure", detection_keys(hits))


# --- new_unmapped_source ------------------------------------------------------------------

def new_unmapped_source_alerts(batch: pd.DataFrame, store: pd.DataFrame) -> pd.DataFrame:
    """One alert per ~375 m cell, raised at the first batch detection where the cell has
    UNMAPPED_MIN_ACTIVE_DAYS+ active days in the trailing UNMAPPED_WINDOW_DAYS, at most
    UNMAPPED_MAX_PRIOR_ACTIVE_DAYS before that window, and no known facility (OSM
    industrial, GEM heat/flare, coal mine, brick kiln) within UNMAPPED_NO_FACILITY_WITHIN_M.
    `store` = every stored detection plus the batch (latitude, longitude, acq_date, label, frp)."""
    if batch.empty:
        return _empty()
    distances = batch.reindex(columns=list(_FACILITY_DISTANCES)).apply(pd.to_numeric, errors="coerce")
    unmapped = batch[distances.fillna(np.inf).min(axis=1) > UNMAPPED_NO_FACILITY_WITHIN_M].copy()
    if unmapped.empty:
        return _empty()
    unmapped["_glat"], unmapped["_glon"] = grid_keys(unmapped)
    cells = unmapped[["_glat", "_glon"]].drop_duplicates()

    glat, glon = grid_keys(store)
    in_cells = pd.MultiIndex.from_arrays([glat, glon]).isin(pd.MultiIndex.from_frame(cells))
    cell_rows = store[in_cells].assign(_glat=glat[in_cells].to_numpy(), _glon=glon[in_cells].to_numpy())
    cell_rows["_day"] = pd.to_datetime(cell_rows["acq_date"])

    rows = []
    for (cell_lat, cell_lon), group in unmapped.sort_values("acq_date").groupby(["_glat", "_glon"]):
        history = cell_rows[(cell_rows["_glat"] == cell_lat) & (cell_rows["_glon"] == cell_lon)]
        days = np.sort(history["_day"].unique())
        for _, detection in group.iterrows():
            when = pd.Timestamp(detection["acq_date"])
            window_start = when - pd.Timedelta(days=UNMAPPED_WINDOW_DAYS - 1)
            in_window = int(((days >= window_start) & (days <= when)).sum())
            before = int((days < window_start).sum())
            if in_window >= UNMAPPED_MIN_ACTIVE_DAYS and before <= UNMAPPED_MAX_PRIOR_ACTIVE_DAYS:
                window = history[(history["_day"] >= window_start) & (history["_day"] <= when)]
                rows.append({
                    "_id": f"{cell_lat}|{cell_lon}",  # once per cell, ever
                    "latitude": round(float(window["latitude"].mean()), 5),
                    "longitude": round(float(window["longitude"].mean()), 5),
                    "acq_date": detection["acq_date"], "acq_time": detection["acq_time"],
                    "satellite": detection["satellite"], "daynight": detection["daynight"],
                    "label": window["label"].mode().iat[0] if window["label"].notna().any() else detection["label"],
                    "frp": round(float(window["frp"].max()), 2),
                    "active_days_30d": in_window,
                    "recurrence_count": int((days <= when).sum()),
                    "first_seen": pd.Timestamp(days[0]).strftime("%Y-%m-%d"),
                })
                break
    if not rows:
        return _empty()
    hits = pd.DataFrame(rows)
    return _finish(hits, "new_unmapped_source", hits.pop("_id"))


# --- large_fire_event ---------------------------------------------------------------------

def large_fire_event_alerts(store: pd.DataFrame, dates: list[str]) -> pd.DataFrame:
    """Same-day clusters of LARGE_FIRE_MIN_DETECTIONS+ detections within LARGE_FIRE_RADIUS_M
    (DBSCAN), for each date in `dates` -- callers pass only complete days, since a day's
    later satellite passes can still grow a cluster."""
    rows = []
    for day in dates:
        same_day = store[store["acq_date"] == day]
        if len(same_day) < LARGE_FIRE_MIN_DETECTIONS:
            continue
        projected = gpd.GeoSeries(
            gpd.points_from_xy(same_day["longitude"], same_day["latitude"]), crs="EPSG:4326"
        ).to_crs(NATIONAL_PROJECTED_CRS)
        xy = np.column_stack([projected.x, projected.y])
        labels = DBSCAN(eps=LARGE_FIRE_RADIUS_M, min_samples=LARGE_FIRE_MIN_DETECTIONS).fit_predict(xy)
        for cluster in sorted(set(labels) - {-1}):
            members = same_day[labels == cluster]
            centre_lat, centre_lon = float(members["latitude"].mean()), float(members["longitude"].mean())
            anchor_lat, anchor_lon = grid_keys(pd.DataFrame({"latitude": [centre_lat], "longitude": [centre_lon]}))
            rows.append({
                "_id": f"{day}|{anchor_lat.iat[0]}|{anchor_lon.iat[0]}",
                "latitude": round(centre_lat, 5), "longitude": round(centre_lon, 5),
                "acq_date": day, "label": members["label"].mode().iat[0],
                "frp": round(float(members["frp"].sum()), 1),  # total FRP of the event
                "event_count": len(members),
                "dominant_class": members["label"].mode().iat[0],
            })
    if not rows:
        return _empty()
    hits = pd.DataFrame(rows)
    return _finish(hits, "large_fire_event", hits.pop("_id"))


def complete_days(start: date, end: date, today: date) -> list[str]:
    """Dates in [start, end] strictly before today (UTC) -- days no satellite pass can still add to."""
    last = min(end, today - pd.Timedelta(days=1).to_pytimedelta())
    return [d.strftime("%Y-%m-%d") for d in pd.date_range(start, last)] if last >= start else []


# --- history --------------------------------------------------------------------------

def read_alert_history(path: Path) -> pd.DataFrame:
    """alerts_history.csv, upgrading rows written before alert types existed (all of
    which were industrial_anomaly) to the current columns."""
    if not Path(path).exists():
        return _empty()
    history = pd.read_csv(path, dtype={"acq_time": str})
    if "alert_type" not in history:
        history["alert_type"] = "industrial_anomaly"
    if "alert_id" not in history:
        history["alert_id"] = "industrial_anomaly|" + detection_keys(history)
    return history.reindex(columns=ALERT_COLUMNS)


def drop_known(alerts: pd.DataFrame, history: pd.DataFrame) -> pd.DataFrame:
    """Alerts not already in history: same alert_id, or (large_fire_event) an event
    already recorded on the same date within LARGE_FIRE_RADIUS_M -- a cluster's centre
    drifts as later passes add detections, so its id alone isn't stable."""
    if alerts.empty:
        return alerts
    fresh = alerts[~alerts["alert_id"].isin(set(history["alert_id"]))]
    events = history[history["alert_type"] == "large_fire_event"]
    if fresh.empty or events.empty:
        return fresh

    def _seen(row) -> bool:
        if row["alert_type"] != "large_fire_event":
            return False
        same_day = events[events["acq_date"] == row["acq_date"]]
        if same_day.empty:
            return False
        dlat = np.radians(same_day["latitude"].astype(float) - row["latitude"])
        dlon = np.radians(same_day["longitude"].astype(float) - row["longitude"])
        a = np.sin(dlat / 2) ** 2 + np.cos(np.radians(row["latitude"])) * np.cos(np.radians(same_day["latitude"].astype(float))) * np.sin(dlon / 2) ** 2
        return bool((2 * 6_371_000 * np.arcsin(np.sqrt(a)) <= LARGE_FIRE_RADIUS_M).any())

    return fresh[~fresh.apply(_seen, axis=1)]


def build_alerts(
    batch: pd.DataFrame,
    store: pd.DataFrame,
    sites: gpd.GeoDataFrame,
    event_dates: list[str],
) -> pd.DataFrame:
    """Every alert type for one run, newest first. `batch` = the detections being
    evaluated (labeled, with anomaly baselines); `store` = all detections (historical +
    live + batch) with latitude, longitude, acq_date, label, frp."""
    frames = [
        industrial_anomaly_alerts(batch),
        fire_near_infrastructure_alerts(batch, sites),
        new_unmapped_source_alerts(batch, store),
        large_fire_event_alerts(store, event_dates),
    ]
    frames = [f for f in frames if not f.empty]
    if not frames:
        return _empty()
    alerts = pd.concat(frames, ignore_index=True)
    alerts["_sort"] = alerts["acq_date"].astype(str) + alerts["acq_time"].fillna("").astype(str)
    return alerts.sort_values("_sort", ascending=False).drop(columns="_sort").reset_index(drop=True)
