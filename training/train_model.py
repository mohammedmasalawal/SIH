"""STEP 2: fit a classifier on labeled_hotspots.csv -> training/artifacts/model.pkl.

Excludes label == "unknown" (never a supervised training target -- see README) and
splits by whole ~375m grid cell, never by row, so the same physical site can never
appear in both train and test (a random row split would let recurrence_count/
is_anomalous, which are computed per-site, leak the answer).
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
from training.spatial_features import grid_keys


def _cell_ids(frame: pd.DataFrame) -> pd.Series:
    grid_lat, grid_lon = grid_keys(frame)
    return grid_lat.astype(str) + "_" + grid_lon.astype(str)


def spatial_block_split(
    frame: pd.DataFrame,
    test_fraction: float = SPATIAL_SPLIT_TEST_FRACTION,
    seed: int = RANDOM_SEED,
) -> tuple[np.ndarray, np.ndarray]:
    """Split frame.index into (train, test) by whole grid cell, deterministic given (frame, seed).

    Every row of a given ~375m grid cell goes entirely to one side, so the same
    physical site never straddles train/test. Reproducible by re-running this
    function with the same frame + seed -- no second held-out-rows file is needed
    (training/evaluate.py relies on exactly this).
    """
    cell_id = _cell_ids(frame)
    unique_cells = np.sort(cell_id.unique())
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(unique_cells)
    n_test_cells = max(1, round(len(shuffled) * test_fraction)) if len(shuffled) > 1 else 0
    test_cells = set(shuffled[:n_test_cells])
    is_test = cell_id.isin(test_cells)
    test_idx = frame.index[is_test].to_numpy()
    train_idx = frame.index[~is_test].to_numpy()
    return train_idx, test_idx


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


def train(labeled_csv: str | Path, output_artifact: str | Path = MODEL_ARTIFACT_PATH) -> dict:
    labeled_csv = Path(labeled_csv)
    frame = pd.read_csv(labeled_csv)
    frame = frame[frame["label"] != "unknown"].reset_index(drop=True)
    if frame.empty:
        raise ValueError(f"No non-'unknown' labeled rows found in {labeled_csv}")

    train_idx, test_idx = spatial_block_split(frame)
    if len(train_idx) == 0 or len(test_idx) == 0:
        raise ValueError(
            "Spatial-block split produced an empty train or test set -- "
            "not enough distinct grid cells in this dataset to hold one out."
        )

    train_frame = frame.loc[train_idx]
    test_frame = frame.loc[test_idx]
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
            "training_data_row_count": int(len(frame)),
            "train_row_count": int(len(train_frame)),
            "test_row_count": int(len(test_frame)),
            "n_grid_cells_train": int(_cell_ids(train_frame).nunique()),
            "n_grid_cells_test": int(_cell_ids(test_frame).nunique()),
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
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    artifact = train(args.labeled_csv, output_artifact=args.output)
    print(f"Trained on device={artifact['metadata']['device']}, wrote {args.output}")
