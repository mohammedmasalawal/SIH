from __future__ import annotations

from unittest.mock import patch

import pandas as pd

from training.spatial_features import grid_keys
from training.train_model import select_device, spatial_block_split, train


def _compute_cell_ids(frame: pd.DataFrame) -> pd.Series:
    grid_lat, grid_lon = grid_keys(frame)
    return grid_lat.astype(str) + "_" + grid_lon.astype(str)


def test_spatial_block_split_never_splits_a_cell(synthetic_labeled_frame):
    frame = synthetic_labeled_frame[synthetic_labeled_frame["label"] != "unknown"].reset_index(drop=True)
    train_idx, test_idx = spatial_block_split(frame)
    cell_id = _compute_cell_ids(frame)
    assert set(cell_id.loc[train_idx]).isdisjoint(set(cell_id.loc[test_idx]))


def test_spatial_block_split_is_deterministic(synthetic_labeled_frame):
    frame = synthetic_labeled_frame[synthetic_labeled_frame["label"] != "unknown"].reset_index(drop=True)
    first = spatial_block_split(frame, seed=7)
    second = spatial_block_split(frame, seed=7)
    assert list(first[0]) == list(second[0])
    assert list(first[1]) == list(second[1])


def test_spatial_block_split_respects_test_fraction_in_cell_terms(synthetic_labeled_frame):
    frame = synthetic_labeled_frame[synthetic_labeled_frame["label"] != "unknown"].reset_index(drop=True)
    cell_id = _compute_cell_ids(frame)
    n_cells = cell_id.nunique()
    train_idx, test_idx = spatial_block_split(frame, test_fraction=0.25)
    n_test_cells = cell_id.loc[test_idx].nunique()
    assert n_test_cells == max(1, round(n_cells * 0.25))


def test_select_device_falls_back_to_cpu_when_cuda_probe_fails():
    with patch("training.train_model.XGBClassifier") as mock_cls:
        mock_cls.return_value.fit.side_effect = RuntimeError("no cuda")
        assert select_device() == "cpu"


def test_train_end_to_end_produces_well_formed_artifact(synthetic_labeled_frame, tmp_path):
    labeled_csv = tmp_path / "labeled.csv"
    synthetic_labeled_frame.to_csv(labeled_csv, index=False)
    output_path = tmp_path / "model.pkl"

    artifact = train(labeled_csv, output_artifact=output_path)

    assert {"model", "feature_columns", "label_classes", "metadata"}.issubset(artifact.keys())
    assert "unknown" not in artifact["label_classes"]
    assert artifact["metadata"]["training_data_row_count"] < len(synthetic_labeled_frame)
    assert output_path.exists()
