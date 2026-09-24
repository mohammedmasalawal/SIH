"""National map data: the packed point binary (training/pack_map_points.py) and
per-detection lookups back into the labeled parquets via DuckDB.

The browser holds all 1.89M points; the server never loads them. A click sends
the point's row_id, which the sidecar's source list maps to (parquet file, row
within file), and DuckDB reads that one row -- so server memory stays flat
regardless of dataset size (DuckDB is capped at DUCKDB_MEMORY_LIMIT).
"""

from __future__ import annotations

import bisect
import datetime
import json
import math
import threading
from pathlib import Path

import duckdb

from backend.config import DUCKDB_MEMORY_LIMIT, MAP_POINTS_META_PATH, MAP_POINTS_PATH, ROOT_DIR


class MapDataUnavailable(RuntimeError):
    """The packed map files haven't been built (run training/pack_map_points.py)."""


class MapStore:
    def __init__(self, points_path: Path = MAP_POINTS_PATH, meta_path: Path = MAP_POINTS_META_PATH) -> None:
        self.points_path = Path(points_path)
        self.gz_path = self.points_path.with_name(self.points_path.name + ".gz")
        self.meta_path = Path(meta_path)
        self._meta: dict | None = None
        self._starts: list[int] = []
        self._con: duckdb.DuckDBPyConnection | None = None
        self._lock = threading.Lock()

    def meta(self) -> dict:
        if self._meta is None:
            if not self.meta_path.exists() or not self.points_path.exists():
                raise MapDataUnavailable("map data not built -- run: python -m training.pack_map_points")
            meta = json.loads(self.meta_path.read_text())
            starts, total = [], 0
            for source in meta["sources"]:
                starts.append(total)
                total += source["rows"]
            self._starts, self._meta = starts, meta
        return self._meta

    def _cursor(self) -> duckdb.DuckDBPyConnection:
        with self._lock:
            if self._con is None:
                self._con = duckdb.connect()
                self._con.execute(f"SET memory_limit='{DUCKDB_MEMORY_LIMIT}'; SET threads=2")
            return self._con.cursor()  # DuckDB cursors are safe to use one per thread

    def detection(self, row_id: int) -> dict | None:
        """The full labeled record for a packed row id, or None if out of range."""
        meta = self.meta()
        if not 0 <= row_id < meta["count"]:
            return None
        source_index = bisect.bisect_right(self._starts, row_id) - 1
        source = meta["sources"][source_index]
        path = Path(source["path"])
        if not path.is_absolute():
            path = ROOT_DIR / path
        file_row = row_id - self._starts[source_index]
        cursor = self._cursor()
        try:
            cursor.execute(
                "SELECT * EXCLUDE (file_row_number) FROM read_parquet(?, file_row_number = true) "
                "WHERE file_row_number = ?",
                [path.as_posix(), file_row],
            )
            columns = [d[0] for d in cursor.description]
            row = cursor.fetchone()
        finally:
            cursor.close()
        if row is None:
            return None
        record = {name: _plain(value) for name, value in zip(columns, row)}
        if record.get("acq_time") is not None:
            # stored as int in some period files (626) and zero-padded str in others ("0626")
            record["acq_time"] = str(record["acq_time"]).zfill(4)
        record["id"] = row_id
        return record


def _plain(value):
    if isinstance(value, float) and math.isnan(value):
        return None
    if isinstance(value, (datetime.date, datetime.datetime)):
        return value.isoformat()[:10]
    return value


map_store = MapStore()
