"""Phase 1 national scale-up checkpoint: build labeled_hotspots for one month of
all-India FIRMS detections, time it, and report the label mix by state.

This is the explicit "one month before 12 months" gate -- run this, look at the
counts/timing/state breakdown, and only then decide whether to repeat it for a full
12-month pull. Not part of the routine per-region build_labels.py path.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd

from training.build_labels import build_labels
from training.report_by_state import label_mix_by_state


def run_pilot(
    firms_path: str | Path,
    industrial_path: str | Path,
    facilities_path: str | Path,
    worldcover_dir: str | Path,
    output_path: str | Path,
) -> None:
    t0 = time.perf_counter()
    result = build_labels(firms_path, industrial_path, facilities_path, output_path, worldcover_raster=worldcover_dir)
    elapsed = time.perf_counter() - t0

    print(f"Total detections: {len(result):,}")
    print(f"build_labels runtime: {elapsed:.1f}s ({elapsed / max(len(result), 1) * 1000:.2f}ms/detection)")
    print(f"Wrote {output_path}")
    print()
    print("Label mix:")
    print(result["label"].value_counts().to_string())
    print()
    print("label_source mix:")
    print(result["label_source"].value_counts().to_string())
    print()
    print("Label mix by state:")
    print(label_mix_by_state(result).to_string())


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--firms", type=Path, required=True)
    parser.add_argument("--industrial", type=Path, required=True)
    parser.add_argument("--facilities", type=Path, required=True)
    parser.add_argument("--worldcover-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_pilot(args.firms, args.industrial, args.facilities, args.worldcover_dir, args.output)
