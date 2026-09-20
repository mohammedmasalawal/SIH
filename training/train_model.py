"""STEP 2: fit a classifier on labeled_hotspots.csv -> training/artifacts/model.pkl.

Excludes label == "unknown" (never a supervised training target -- see README) and
excludes every gold-verification cell (see _load_gold_cell_ids) -- a gold cell inside
a training block would let manually-verified ground truth leak into the model's own
training signal, defeating the point of holding it out for blind review.

Two split strategies are available (see SPLIT_STRATEGIES):

- "detection" (the default train() uses): every detection dated on/after a
  row-count-based cutoff goes to test, regardless of which grid cell it's in.
  Persistent sites straddle train/test.
- "spatial_block": whole ~375m grid cells go to one side only (no physical site
  straddles train/test), ordered by each cell's first-seen date so test is drawn
  from later activity than train.

spatial_block_split was the original design -- it guarantees no site's rows are
ever split across train/test, which is the cleaner leakage story in principle.
In practice, on this dataset it pushed gas flare and industrial down to 1 and 17
test rows respectively (out of ~570 and ~384 total): both classes are dominated by
a handful of long-lived sites that were first detected early in the pull and kept
recurring for months, so ordering *cells* by first-seen date puts nearly all of
their rows on the train side no matter the target test_fraction. detection_level_
temporal_split fixes the per-class test-set size by cutting on detection date
directly (28 and 27 test rows for gas flare/industrial on the same data) at the
cost of letting a single site's earlier detections train the model that is then
evaluated on that same site's later ones. Both are computed and reported by
training/evaluate.py --split=both; nothing here silently prefers one.

recurrence_count/first_seen/last_seen/is_anomalous, as loaded from the labeled CSV,
were computed by training/build_labels.py over its entire input FIRMS pull --
including whatever is on the test side of either split. Before either split's rows
reach the feature matrix, recompute_pre_split_features recomputes all four using
only detections dated before the split's cutoff date, so no model input (train or
test) can reflect information from on or after the train/test boundary.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import xgboost
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

from backend.classification.features import FEATURE_COLUMNS, hotspots_to_feature_matrix
from backend.config import (
    GOLD_SAMPLE_PATH,
    MODEL_ARTIFACT_PATH,
    RANDOM_SEED,
    SPATIAL_SPLIT_TEST_FRACTION,
    XGB_COLSAMPLE_BYTREE,
    XGB_EARLY_STOPPING_ROUNDS,
    XGB_LEARNING_RATE,
    XGB_MAX_DEPTH,
    XGB_N_ESTIMATORS,
    XGB_REG_LAMBDA,
    XGB_SUBSAMPLE,
)
from training.spatial_features import add_recurrence_features, compute_is_anomalous, grid_keys


def _cell_ids(frame: pd.DataFrame) -> pd.Series:
    grid_lat, grid_lon = grid_keys(frame)
    return grid_lat.astype(str) + "_" + grid_lon.astype(str)


def _load_gold_cell_ids(gold_csv: str | Path = GOLD_SAMPLE_PATH) -> set[str]:
    """Grid-cell ids covered by the manual-review gold set (training/data/gold_sample.csv).

    Matched purely by cell id, not by exact lat/lon or sample_id -- so anything else
    that happens to fall in the same ~375m block as a gold sample is caught too,
    per the README's "exclude every gold-set cell, and the whole spatial block it
    falls in" requirement.
    """
    gold = pd.read_csv(gold_csv)
    return set(_cell_ids(gold))


def exclude_gold_cells(frame: pd.DataFrame, gold_csv: str | Path = GOLD_SAMPLE_PATH) -> pd.DataFrame:
    """Drop every row whose spatial block contains a gold-verification sample.

    Must run before spatial_block_split, not after: a gold cell that slipped into
    either side would let hand-verified ground truth leak into that side's own
    training signal (directly, if it lands in train; or into the eval score's
    apparent difficulty, if it lands in test).
    """
    gold_cells = _load_gold_cell_ids(gold_csv)
    cell_id = _cell_ids(frame)
    return frame[~cell_id.isin(gold_cells)].reset_index(drop=True)


def spatial_block_split(
    frame: pd.DataFrame,
    test_fraction: float = SPATIAL_SPLIT_TEST_FRACTION,
    seed: int = RANDOM_SEED,
) -> tuple[np.ndarray, np.ndarray]:
    """Split frame.index into (train, test) by whole grid cell, ordered by first-seen date.

    Every row of a given ~375m grid cell goes entirely to one side (no physical site
    straddles train/test), and cells are ordered by their own first-seen date before
    the last test_fraction of them are cut off into test -- so test is drawn from the
    latest activity in the frame, train from everything earlier, not a random sample
    of cells. `seed` only breaks ties between cells sharing an identical first-seen
    date; with no ties the split is fully determined by (frame, test_fraction).

    Reproducible by re-running this function with the same frame + seed -- no second
    held-out-rows file is needed (training/evaluate.py relies on exactly this).
    """
    cell_id = _cell_ids(frame)
    cell_first_seen = frame.groupby(cell_id)["acq_date"].min()
    rng = np.random.default_rng(seed)
    tie_break = pd.Series(rng.permutation(len(cell_first_seen)), index=cell_first_seen.index)
    ordered_cells = (
        pd.DataFrame({"first_seen": cell_first_seen, "tie_break": tie_break})
        .sort_values(["first_seen", "tie_break"])
        .index.to_numpy()
    )
    n_test_cells = max(1, round(len(ordered_cells) * test_fraction)) if len(ordered_cells) > 1 else 0
    test_cells = set(ordered_cells[len(ordered_cells) - n_test_cells :]) if n_test_cells else set()
    is_test = cell_id.isin(test_cells)
    test_idx = frame.index[is_test].to_numpy()
    train_idx = frame.index[~is_test].to_numpy()
    return train_idx, test_idx


def detection_level_temporal_split(
    frame: pd.DataFrame,
    test_fraction: float = SPATIAL_SPLIT_TEST_FRACTION,
) -> tuple[np.ndarray, np.ndarray, str]:
    """Split frame.index into (train, test) by detection date alone: every row dated
    on/after the cutoff goes to test, regardless of which grid cell it belongs to.

    Unlike spatial_block_split, a persistent site's rows can straddle train/test here
    -- that's the deliberate trade this makes. spatial_block_split's per-cell
    first-seen ordering means a site active across most of the pull's date range
    (the norm for gas flare/industrial: a handful of facilities recurring for
    months) contributes almost none of its rows to test no matter test_fraction,
    because the *cell* "started" early even though it kept detecting long after the
    cutoff. Cutting on each detection's own date instead guarantees test_fraction of
    *rows* -- and therefore a proportional share of every class's rows -- ends up in
    test, at the cost of letting some sites' earlier detections train the model that
    is then evaluated on their own later ones.

    cutoff_date is chosen so that the last test_fraction of rows by acq_date (ties
    included) fall on/after it. Returns (train_idx, test_idx, cutoff_date).
    """
    dates_sorted = np.sort(frame["acq_date"].to_numpy())
    n_test = max(1, round(len(dates_sorted) * test_fraction)) if len(dates_sorted) > 1 else 0
    cutoff_date = dates_sorted[len(dates_sorted) - n_test] if n_test else None
    if cutoff_date is None:
        return frame.index.to_numpy(), np.array([], dtype=frame.index.dtype), str(dates_sorted[-1])
    is_test = frame["acq_date"] >= cutoff_date
    test_idx = frame.index[is_test].to_numpy()
    train_idx = frame.index[~is_test].to_numpy()
    return train_idx, test_idx, str(cutoff_date)


SPLIT_STRATEGIES = ("detection", "spatial_block")


def recompute_pre_split_features(frame: pd.DataFrame, cutoff_date: str) -> pd.DataFrame:
    """Overwrite recurrence_count/first_seen/last_seen/is_anomalous using only
    detections dated strictly before cutoff_date as the grouping/history universe.

    Applied to the whole frame (train side and test side alike) so that no row's
    model-input features can reflect information from on or after the train/test
    boundary -- not just "don't use test rows to compute train features," but "don't
    use anything dated at-or-after cutoff at all," which also catches a train-side
    cell whose own recurrence continues past the boundary. A cell with zero
    detections before cutoff_date (i.e. its activity starts only at/after the
    boundary -- always true of test cells, by construction of spatial_block_split)
    gets recurrence_count=0, first_seen/last_seen=NaT, is_anomalous=False: as far as
    pre-split data is concerned, that site had no observed history yet.
    """
    result = frame.copy()
    grid_lat, grid_lon = grid_keys(result)
    cell_id = grid_lat.astype(str) + "_" + grid_lon.astype(str)

    pre = result[result["acq_date"] < cutoff_date]
    pre_grid_lat, pre_grid_lon = grid_keys(pre)
    pre_cell_id = pre_grid_lat.astype(str) + "_" + pre_grid_lon.astype(str)

    pre_features = add_recurrence_features(pre, pre_grid_lat, pre_grid_lon)
    cell_stats = (
        pre_features.assign(_cell=pre_cell_id)
        .groupby("_cell")[["recurrence_count", "first_seen", "last_seen"]]
        .first()
    )
    result["recurrence_count"] = cell_id.map(cell_stats["recurrence_count"]).fillna(0).astype("int64")
    result["first_seen"] = cell_id.map(cell_stats["first_seen"])
    result["last_seen"] = cell_id.map(cell_stats["last_seen"])

    pre_anomalous = compute_is_anomalous(pre, pre_grid_lat, pre_grid_lon)
    result["is_anomalous"] = False
    result.loc[pre.index, "is_anomalous"] = pre_anomalous.to_numpy()
    return result


def _load_and_filter_frame(labeled_csv: str | Path, gold_csv: str | Path) -> pd.DataFrame:
    """Read labeled_csv, drop label == "unknown", and exclude gold cells.

    Shared starting point for both split strategies, so "detection" and
    "spatial_block" runs on the same labeled_csv are evaluated against the exact
    same candidate rows before splitting.
    """
    frame = pd.read_csv(labeled_csv)
    frame = frame[frame["label"] != "unknown"].reset_index(drop=True)
    frame = exclude_gold_cells(frame, gold_csv)
    if frame.empty:
        raise ValueError(f"No non-'unknown', non-gold-cell rows found in {labeled_csv}")
    return frame


def prepare_training_frame(
    labeled_csv: str | Path,
    gold_csv: str | Path = GOLD_SAMPLE_PATH,
    test_fraction: float = SPATIAL_SPLIT_TEST_FRACTION,
    seed: int = RANDOM_SEED,
    split: str = "detection",
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, str]:
    """Load labeled_csv, exclude unknown/gold rows, split, and recompute pre-split-safe
    features. Shared by train() and evaluate() so both operate on identically-prepared
    data -- evaluate() calling this with the model's own recorded seed is what makes
    its reproduced test set match the one the model was actually evaluated against.

    `split` selects the strategy -- "detection" (default; see
    detection_level_temporal_split) or "spatial_block" (see spatial_block_split) --
    see the module docstring for why "detection" is the default train() uses.

    Returns (safe_frame, train_idx, test_idx, cutoff_date). Index the same rows out of
    safe_frame for both splits/scoring -- it, not the raw CSV, carries the leak-safe
    recurrence_count/first_seen/last_seen/is_anomalous.
    """
    if split not in SPLIT_STRATEGIES:
        raise ValueError(f"split must be one of {SPLIT_STRATEGIES}, got {split!r}")
    frame = _load_and_filter_frame(labeled_csv, gold_csv)

    if split == "detection":
        train_idx, test_idx, cutoff_date = detection_level_temporal_split(frame, test_fraction=test_fraction)
    else:
        train_idx, test_idx = spatial_block_split(frame, test_fraction=test_fraction, seed=seed)
        cutoff_date = frame.loc[test_idx, "acq_date"].min() if len(test_idx) else None

    if len(train_idx) == 0 or len(test_idx) == 0:
        raise ValueError(
            f"{split} split produced an empty train or test set -- "
            "not enough distinct grid cells/dates in this dataset to hold one out."
        )

    safe_frame = recompute_pre_split_features(frame, cutoff_date)
    return safe_frame, train_idx, test_idx, cutoff_date


def select_device() -> str:
    """Fit a throwaway 1-estimator model with device="cuda" to probe real GPU usability.

    A real capability probe against the installed xgboost build, not a guess --
    correctly handles both "no GPU/driver present" and "this xgboost wheel wasn't
    built with GPU support at all," falling back to "cpu" on any failure.
    """
    try:
        dummy_x = np.zeros((4, 2), dtype="float32")
        dummy_y = np.array([0, 1, 0, 1])
        probe = XGBClassifier(n_estimators=1, max_depth=1, device="cuda", tree_method="hist")
        probe.fit(dummy_x, dummy_y)
        return "cuda"
    except Exception:
        return "cpu"


def train(
    labeled_csv: str | Path,
    output_artifact: str | Path = MODEL_ARTIFACT_PATH,
    gold_csv: str | Path = GOLD_SAMPLE_PATH,
    split: str = "detection",
) -> dict:
    labeled_csv = Path(labeled_csv)
    safe_frame, train_idx, test_idx, cutoff_date = prepare_training_frame(labeled_csv, gold_csv=gold_csv, split=split)

    train_frame = safe_frame.loc[train_idx]
    test_frame = safe_frame.loc[test_idx]
    x_train = hotspots_to_feature_matrix(train_frame)
    x_test = hotspots_to_feature_matrix(test_frame)

    # xgboost>=2's sklearn wrapper requires y already encoded as 0..n_classes-1 --
    # raw string labels raise ValueError at fit time. label_classes preserves the
    # string<->index mapping (sorted by LabelEncoder) so predictions can be decoded
    # back to label strings downstream (model_classifier.py, evaluate.py).
    encoder = LabelEncoder()
    y_train = encoder.fit_transform(train_frame["label"])
    y_test = encoder.transform(test_frame["label"])

    device = select_device()
    model = XGBClassifier(
        n_estimators=XGB_N_ESTIMATORS,
        max_depth=XGB_MAX_DEPTH,
        learning_rate=XGB_LEARNING_RATE,
        subsample=XGB_SUBSAMPLE,
        colsample_bytree=XGB_COLSAMPLE_BYTREE,
        reg_lambda=XGB_REG_LAMBDA,
        # objective/eval_metric/num_class deliberately left to XGBClassifier's own
        # auto-selection from the encoded y (binary:logistic+logloss vs
        # multi:softprob+mlogloss+num_class) -- how many label classes actually
        # survive excluding "unknown" varies per dataset (could be as few as 2), and
        # hand-setting either a multiclass-only objective or eval_metric breaks the
        # binary case; explicit multi:softprob also changes predict()'s return shape
        # on this xgboost build.
        early_stopping_rounds=XGB_EARLY_STOPPING_ROUNDS,
        device=device,
        tree_method="hist",
        random_state=RANDOM_SEED,
    )
    model.fit(x_train, y_train, eval_set=[(x_test, y_test)], verbose=False)

    artifact = {
        "model": model,
        "feature_columns": list(FEATURE_COLUMNS),
        "label_classes": list(encoder.classes_),
        "metadata": {
            "trained_at": datetime.now(timezone.utc).isoformat(),
            "training_data_path": str(labeled_csv),
            "training_data_row_count": int(len(safe_frame)),
            "train_row_count": int(len(train_frame)),
            "test_row_count": int(len(test_frame)),
            "n_grid_cells_train": int(_cell_ids(train_frame).nunique()),
            "n_grid_cells_test": int(_cell_ids(test_frame).nunique()),
            "gold_cells_excluded": len(_load_gold_cell_ids(gold_csv)),
            "split_strategy": split,
            "split_cutoff_date": str(cutoff_date),
            "xgboost_version": xgboost.__version__,
            "device": device,
            "random_seed": RANDOM_SEED,
        },
    }
    output_artifact = Path(output_artifact)
    output_artifact.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, output_artifact)
    return artifact


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labeled-csv", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=MODEL_ARTIFACT_PATH)
    parser.add_argument("--split", choices=SPLIT_STRATEGIES, default="detection")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    artifact = train(args.labeled_csv, output_artifact=args.output, split=args.split)
    print(f"Trained on device={artifact['metadata']['device']}, wrote {args.output}")
