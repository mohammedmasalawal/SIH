"""Industrial sub-types: display-only metadata for detections already labelled `industrial`.

    python -m training.industrial_subtype        # national split, read-only

A SECOND LAYER, NOT A NEW LABEL. The class stays "industrial"; the sub-type says what
kind of mapped facility is nearest. Three rules keep it honest:

1. No label changes. This module takes lat/lon/label in and returns new columns; it never
   returns or writes `label`, and nothing in backend/classification or the distance columns
   reads it. tests/test_industrial_subtype.py checks the label column is byte-identical.
2. The sub-type describes the nearest typed facility, not the fire. A detection 300 m from a
   cement plant is not necessarily the kiln. The radius is fixed at
   INDUSTRIAL_SUBTYPE_RADIUS_M (1 km) and was not tuned; the facility name and distance are
   stored so the popup can show them.
3. The multi-sector GEM sites are NOT fed to the rules. This is a separate join over a
   separate file (GEM_MULTISECTOR_FACILITIES_PATH); the rules and dist_to_* columns still use
   only the 98-site table.

Sub-type mapping, fixed before any run:

    Refinery / oil & gas     GEM oil_gas_field, lng_terminal; OSM industrial = refinery, oil,
                             oil_storage, petroleum_terminal, gas, gas_plant, natural_gas,
                             gas_storage
    Thermal power plant      GEM coal_power_plant, oil_gas_power_plant; OSM power=plant with
                             plant:source containing coal, gas, oil, diesel or nuclear
                             (solar/wind/hydro/biomass/waste never count)
    Mine / coal-seam fire    GEM coal_mine; OSM industrial = mine (OSM's tag is not
                             coal-specific, hence the wider name)
    Steel                    GEM steel_plant; OSM industrial = steel, steel_mill, steelmaking
    Cement                   GEM cement_plant; OSM industrial = cement, cement_plant
    Chemical / petrochemical GEM chemical_plant; OSM industrial = chemical, petrochemical
    Industrial, type not     everything else -- including generic OSM landuse=industrial,
    identified               which has no type

GEM sites are used at every status except GEM_COAL_STATUSES_EXCLUDED (proposed, announced,
cancelled, shelved): a closed or mothballed coal mine can still have a seam fire. Coal mines
are GEM points, so a large mine's far side can be beyond 1 km and read "not identified".
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

from backend.config import (
    GEM_COAL_STATUSES_EXCLUDED,
    GEM_MULTISECTOR_FACILITIES_PATH,
    INDUSTRIAL_SUBTYPE_RADIUS_M,
    NATIONAL_INDUSTRIAL_CONTEXT_PATH,
    NATIONAL_PROJECTED_CRS,
)
from backend.ingestion.context_sources import exclude_renewable_power_plants

SUBTYPES = (
    "Refinery / oil & gas",
    "Thermal power plant",
    "Mine / coal-seam fire",
    "Steel",
    "Cement",
    "Chemical / petrochemical",
)
NOT_IDENTIFIED = "Industrial, type not identified"
ALL_SUBTYPES = SUBTYPES + (NOT_IDENTIFIED,)
SUBTYPE_CODE = {name: i + 1 for i, name in enumerate(ALL_SUBTYPES)}  # 0 = not an industrial detection
NOT_APPLICABLE_CODE = 0

GEM_TYPE_TO_SUBTYPE = {
    "oil_gas_field": "Refinery / oil & gas",
    "lng_terminal": "Refinery / oil & gas",
    "coal_power_plant": "Thermal power plant",
    "oil_gas_power_plant": "Thermal power plant",
    "coal_mine": "Mine / coal-seam fire",
    "steel_plant": "Steel",
    "cement_plant": "Cement",
    "chemical_plant": "Chemical / petrochemical",
}
OSM_INDUSTRIAL_TO_SUBTYPE = {
    **dict.fromkeys(("refinery", "oil", "oil_storage", "petroleum_terminal", "gas", "gas_plant", "natural_gas", "gas_storage"),
                    "Refinery / oil & gas"),
    "mine": "Mine / coal-seam fire",
    **dict.fromkeys(("steel", "steel_mill", "steelmaking"), "Steel"),
    **dict.fromkeys(("cement", "cement_plant"), "Cement"),
    **dict.fromkeys(("chemical", "petrochemical"), "Chemical / petrochemical"),
}
OSM_THERMAL_FUELS = ("coal", "gas", "oil", "diesel", "nuclear")

OUTPUT_COLUMNS = ["industrial_subtype", "subtype_facility", "subtype_source", "subtype_kind", "subtype_distance_m"]
_TYPED_COLUMNS = ["name", "source", "kind", "subtype"]


def _clean(text) -> str:
    return str(text).strip().lower() if isinstance(text, str) else ""


def _gem_typed(path: Path) -> gpd.GeoDataFrame:
    gem = pd.read_csv(path)
    first_status = gem["status"].astype(str).str.split(";").str[0].str.strip().str.lower()
    gem = gem[~first_status.isin(GEM_COAL_STATUSES_EXCLUDED)]
    gem = gem[gem["facility_type"].isin(GEM_TYPE_TO_SUBTYPE)].dropna(subset=["latitude", "longitude"])
    return gpd.GeoDataFrame(
        {
            "name": gem["facility_name"].astype(str).to_numpy(),
            "source": "GEM",
            "kind": gem["facility_type"].str.replace("_", " ").to_numpy(),
            "subtype": gem["facility_type"].map(GEM_TYPE_TO_SUBTYPE).to_numpy(),
        },
        geometry=gpd.points_from_xy(gem["longitude"], gem["latitude"]),
        crs="EPSG:4326",
    )


def _osm_typed(path: Path) -> gpd.GeoDataFrame:
    osm = exclude_renewable_power_plants(gpd.read_parquet(path))
    column = lambda name: osm[name] if name in osm else pd.Series(None, index=osm.index, dtype="object")
    industrial = column("industrial").map(_clean)
    by_tag = industrial.map(OSM_INDUSTRIAL_TO_SUBTYPE)
    fuels = column("plant:source").map(_clean).str.split(";")
    thermal = (column("power") == "plant") & fuels.map(lambda parts: any(p.strip() in OSM_THERMAL_FUELS for p in parts))
    subtype = by_tag.where(~thermal, "Thermal power plant")
    keep = subtype.notna()
    kind = np.where(thermal, "power plant (" + column("plant:source").astype(str) + ")", "industrial=" + industrial)
    names = column("name")
    return gpd.GeoDataFrame(
        {
            "name": names[keep].where(names[keep].notna(), None).to_numpy(),
            "source": "OSM",
            "kind": kind[keep.to_numpy()],
            "subtype": subtype[keep].to_numpy(),
        },
        geometry=osm.geometry[keep].to_numpy(),
        crs=osm.crs,
    )


def load_typed_facilities(
    gem_path: Path = GEM_MULTISECTOR_FACILITIES_PATH,
    osm_path: Path = NATIONAL_INDUSTRIAL_CONTEXT_PATH,
) -> gpd.GeoDataFrame:
    """Every typed facility (GEM first, then OSM), in NATIONAL_PROJECTED_CRS. A missing
    file contributes nothing, so its detections fall through to 'type not identified'."""
    parts = []
    if Path(gem_path).exists():
        parts.append(_gem_typed(Path(gem_path)))
    if Path(osm_path).exists():
        parts.append(_osm_typed(Path(osm_path)))
    if not parts:
        empty = gpd.GeoDataFrame({c: [] for c in _TYPED_COLUMNS}, geometry=[], crs=NATIONAL_PROJECTED_CRS)
        return empty
    typed = pd.concat([p.to_crs(NATIONAL_PROJECTED_CRS) for p in parts], ignore_index=True)
    return gpd.GeoDataFrame(typed, geometry="geometry", crs=NATIONAL_PROJECTED_CRS)


def compute_industrial_subtypes(
    frame: pd.DataFrame, typed: gpd.GeoDataFrame, radius_m: float = INDUSTRIAL_SUBTYPE_RADIUS_M
) -> pd.DataFrame:
    """OUTPUT_COLUMNS for every row of `frame`, indexed like it. Only rows labelled
    `industrial` get a sub-type (nearest typed facility within radius_m, else NOT_IDENTIFIED);
    every other row is all-missing. Reads latitude, longitude and label; never modifies
    `frame` and never returns a label column."""
    out = pd.DataFrame({
        "industrial_subtype": pd.Series(None, index=frame.index, dtype="object"),
        "subtype_facility": pd.Series(None, index=frame.index, dtype="object"),
        "subtype_source": pd.Series(None, index=frame.index, dtype="object"),
        "subtype_kind": pd.Series(None, index=frame.index, dtype="object"),
        "subtype_distance_m": pd.Series(np.nan, index=frame.index, dtype="float64"),
    })
    is_industrial = (frame["label"] == "industrial").to_numpy()
    if not is_industrial.any():
        return out
    positions = np.flatnonzero(is_industrial)
    out.iloc[positions, out.columns.get_loc("industrial_subtype")] = NOT_IDENTIFIED
    if typed.empty:
        return out

    points = gpd.GeoDataFrame(
        {"pos": positions},
        geometry=gpd.points_from_xy(frame["longitude"].to_numpy()[positions], frame["latitude"].to_numpy()[positions]),
        crs="EPSG:4326",
    ).to_crs(NATIONAL_PROJECTED_CRS)
    joined = gpd.sjoin_nearest(points, typed, max_distance=radius_m, distance_col="_dist", how="inner")
    # nearest wins; on an exact tie the earlier typed row (GEM before OSM) wins
    joined = joined.reset_index().sort_values(["pos", "_dist", "index_right"]).drop_duplicates("pos")
    rows = joined["pos"].to_numpy()
    for column, values in (
        ("industrial_subtype", joined["subtype"]),
        ("subtype_facility", joined["name"]),
        ("subtype_source", joined["source"]),
        ("subtype_kind", joined["kind"]),
        ("subtype_distance_m", joined["_dist"].round(0)),
    ):
        out.iloc[rows, out.columns.get_loc(column)] = values.to_numpy()
    return out


def label_fingerprint(labels: pd.Series) -> str:
    """SHA-256 of the label column's exact bytes, for the byte-identical-before/after check."""
    return hashlib.sha256("\n".join(labels.astype(str)).encode("utf-8")).hexdigest()


def split_table(subtypes: pd.DataFrame) -> pd.DataFrame:
    counts = subtypes["industrial_subtype"].value_counts().reindex(ALL_SUBTYPES, fill_value=0)
    table = counts.rename("detections").to_frame()
    table["share_pct"] = (table["detections"] / max(int(counts.sum()), 1) * 100).round(1)
    return table


if __name__ == "__main__":
    from backend.config import MAP_POINTS_META_PATH, ROOT_DIR
    import pyarrow.parquet as pq

    sources = [Path(s["path"]) for s in json.loads(Path(MAP_POINTS_META_PATH).read_text())["sources"]]
    frames, before = [], []
    for path in sources:
        path = path if path.is_absolute() else ROOT_DIR / path
        frame = pd.read_parquet(path, columns=["latitude", "longitude", "label"])
        frames.append(frame)
        before.append(label_fingerprint(frame["label"]))
    rows = pd.concat(frames, ignore_index=True)
    result = compute_industrial_subtypes(rows, load_typed_facilities())
    table = split_table(result)
    print(f"{int((rows['label'] == 'industrial').sum()):,} industrial detections of {len(rows):,}")
    print(table.to_string())
    # the parquets were only read; confirm the label column on disk is unchanged
    after = [label_fingerprint(pd.read_parquet(p if p.is_absolute() else ROOT_DIR / p, columns=["label"])["label"]) for p in sources]
    print(f"label column identical before/after in all {len(sources)} source files: {before == after}")
