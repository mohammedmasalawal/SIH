"""Pack the national labeled parquets into one compact binary for the GPU map.

Struct-of-arrays, little-endian, every array aligned for its type so the browser
can view each one as a typed array without copying:

    positions   float32[2N]  lon, lat interleaved (deck.gl getPosition, size 2)
    row_id      uint32[N]    the point's row in the concatenated parquets (sources order);
                             /api/detection/{row_id} resolves it back to a
                             (file, file_row_number) pair via the sidecar's sources.
                             Points themselves are stored in draw order (rare classes
                             last), so point index != row_id.
    day         uint16[N]    days since meta["base_date"]
    class_id    uint8[N]     index into meta["classes"] (rules.CLASSES order)
    is_anomalous uint8[N]    0/1

16 bytes per detection. The JSON sidecar records each array's byte offset, the
class order, the date base, and the source files with row counts. Regenerate
whenever the labeled parquets change -- row ids are only valid against the exact
files and order recorded in the sidecar.
"""

from __future__ import annotations

import argparse
import gzip
import json
import shutil
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from backend.classification.rules import CLASSES
from backend.config import MAP_POINTS_META_PATH, MAP_POINTS_PATH, NATIONAL_LABELED_PARQUETS, ROOT_DIR

_COLUMNS = ["longitude", "latitude", "acq_date", "label", "is_anomalous"]
_DRAW_PRIORITY = ("unknown", "agricultural burning", "wildfire", "industrial", "gas flare")  # first drawn first
_LAYOUT = (  # name, dtype, values per row -- order keeps every offset aligned
    ("positions", "float32", 2),
    ("row_id", "uint32", 1),
    ("day", "uint16", 1),
    ("class_id", "uint8", 1),
    ("is_anomalous", "uint8", 1),
)


def _relative(path: Path) -> str:
    path = Path(path).resolve()
    try:
        return path.relative_to(ROOT_DIR).as_posix()
    except ValueError:
        return path.as_posix()


def pack(parquets: list[Path], out_bin: Path, out_meta: Path) -> dict:
    frames, sources = [], []
    for path in parquets:
        frame = pd.read_parquet(path, columns=_COLUMNS)
        frames.append(frame)
        sources.append({"path": _relative(path), "rows": len(frame)})
    data = pd.concat(frames, ignore_index=True)
    count = len(data)

    unknown_labels = set(data["label"].unique()) - set(CLASSES)
    if unknown_labels:
        raise ValueError(f"labels not in rules.CLASSES: {sorted(unknown_labels)}")

    dates = pd.to_datetime(data["acq_date"]).dt.normalize()
    base = dates.min()
    day = (dates - base).dt.days.to_numpy()
    if day.max() > np.iinfo(np.uint16).max:
        raise ValueError("date span exceeds uint16 day index")

    class_id = data["label"].map({label: i for i, label in enumerate(CLASSES)}).to_numpy(dtype="u1")
    # GPU draw order = file order, so later points paint over earlier ones. Sort so the
    # rare classes land last and stay visible over the ~1.5M agri/wildfire points;
    # row_id keeps each point's original row for /api/detection lookups.
    priority = np.array([_DRAW_PRIORITY.index(label) for label in CLASSES])[class_id]
    order = np.argsort(priority, kind="stable")
    positions = np.column_stack([data["longitude"], data["latitude"]]).astype("<f4")
    arrays = {
        "positions": positions[order].ravel(),
        "row_id": order.astype("<u4"),
        "day": day.astype("<u2")[order],
        "class_id": class_id[order],
        "is_anomalous": data["is_anomalous"].fillna(False).astype(bool).to_numpy(dtype="u1")[order],
    }

    layout, offset = {}, 0
    out_bin.parent.mkdir(parents=True, exist_ok=True)
    with open(out_bin, "wb") as handle:
        for name, dtype, per_row in _LAYOUT:
            array = arrays[name]
            assert array.size == count * per_row
            layout[name] = {"offset": offset, "dtype": dtype, "length": int(array.size)}
            handle.write(array.tobytes())
            offset += array.nbytes
    # Pre-compressed copy (~50% of raw) so the API can serve Content-Encoding: gzip
    # without compressing 30 MB on every request.
    gz_path = out_bin.with_name(out_bin.name + ".gz")
    with open(out_bin, "rb") as src, gzip.open(gz_path, "wb", compresslevel=6) as dst:
        shutil.copyfileobj(src, dst)

    class_counts = np.bincount(arrays["class_id"], minlength=len(CLASSES))
    max_day = int(day.max())
    base_date = base.date()
    meta = {
        "count": count,
        "bytes": offset,
        "base_date": base_date.isoformat(),
        "max_day": max_day,
        "max_date": (base_date + timedelta(days=max_day)).isoformat(),
        "classes": list(CLASSES),
        "class_counts": {label: int(n) for label, n in zip(CLASSES, class_counts)},
        "layout": layout,
        "sources": sources,
    }
    out_meta.write_text(json.dumps(meta, indent=2))
    return meta


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--parquets", type=Path, nargs="+", default=list(NATIONAL_LABELED_PARQUETS))
    parser.add_argument("--out", type=Path, default=MAP_POINTS_PATH)
    parser.add_argument("--meta", type=Path, default=MAP_POINTS_META_PATH)
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    meta = pack(args.parquets, args.out, args.meta)
    print(f"Packed {meta['count']:,} detections -> {args.out} ({meta['bytes'] / 1e6:.1f} MB), "
          f"{meta['base_date']}..{meta['max_date']}")
