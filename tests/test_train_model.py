from __future__ import annotations

from unittest.mock import patch

import pandas as pd

from training.spatial_features import grid_keys
from training.train_model import (
    detection_level_temporal_split,
    exclude_gold_cells,
    select_device,
    spatial_block_split,
    train,
)


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


def test_detection_level_temporal_split_is_deterministic(synthetic_labeled_frame):
    frame = synthetic_labeled_frame[synthetic_labeled_frame["label"] != "unknown"].reset_index(drop=True)
    first = detection_level_temporal_split(frame)
    second = detection_level_temporal_split(frame)
    assert list(first[0]) == list(second[0])
    assert list(first[1]) == list(second[1])
    assert first[2] == second[2]


def test_detection_level_temporal_split_respects_test_fraction_in_row_terms(synthetic_labeled_frame):
    frame = synthetic_labeled_frame[synthetic_labeled_frame["label"] != "unknown"].reset_index(drop=True)
    train_idx, test_idx, _ = detection_level_temporal_split(frame, test_fraction=0.25)
    assert len(test_idx) == max(1, round(len(frame) * 0.25))
    assert len(train_idx) + len(test_idx) == len(frame)


def test_detection_level_temporal_split_test_rows_are_all_on_or_after_cutoff(synthetic_labeled_frame):
    """Unlike spatial_block_split, a single grid cell's rows CAN straddle train/test
    here -- what must hold is that every test row's own date is >= cutoff."""
    frame = synthetic_labeled_frame[synthetic_labeled_frame["label"] != "unknown"].reset_index(drop=True)
    train_idx, test_idx, cutoff_date = detection_level_temporal_split(frame)
    assert (frame.loc[test_idx, "acq_date"] >= cutoff_date).all()
    assert (frame.loc[train_idx, "acq_date"] < cutoff_date).all()


def test_detection_level_split_gives_minority_classes_far_more_test_rows_than_spatial_block(synthetic_labeled_frame):
    """The whole reason detection-level exists: spatial_block_split orders whole
    cells by first-seen date, so a class dominated by a handful of long-lived sites
    (started early, kept recurring) contributes almost none of its rows to test no
    matter test_fraction. Splitting by each detection's own date instead guarantees
    a proportional share of every class's rows lands in test."""
    frame = synthetic_labeled_frame[synthetic_labeled_frame["label"] != "unknown"].reset_index(drop=True)
    _, spatial_test_idx = spatial_block_split(frame, test_fraction=0.3)
    _, detection_test_idx, _ = detection_level_temporal_split(frame, test_fraction=0.3)

    spatial_gas_flare_test = (frame.loc[spatial_test_idx, "label"] == "gas flare").sum()
    detection_gas_flare_test = (frame.loc[detection_test_idx, "label"] == "gas flare").sum()
    assert detection_gas_flare_test > spatial_gas_flare_test


def test_exclude_gold_cells_unions_multiple_gold_files(synthetic_labeled_frame, tmp_path):
    """gold_sample.csv drove the is_gas_flare rule fix, so gold_holdout.csv is the
    independent check now -- both must be excluded from training, not just the
    first one. A row is dropped if its cell matches EITHER file, and a single path
    (the old call shape) still works unchanged."""
    frame = synthetic_labeled_frame[synthetic_labeled_frame["label"] != "unknown"].reset_index(drop=True)
    cell_id = _compute_cell_ids(frame)
    industrial_cell = cell_id[frame["label"] == "industrial"].iloc[0]
    flare_cell = cell_id[frame["label"] == "gas flare"].iloc[0]
    industrial_row = frame[cell_id == industrial_cell].iloc[0]
    flare_row = frame[cell_id == flare_cell].iloc[0]

    gold_a = tmp_path / "gold_a.csv"
    gold_b = tmp_path / "gold_b.csv"
    pd.DataFrame([{"sample_id": "A1", "latitude": industrial_row["latitude"], "longitude": industrial_row["longitude"]}]).to_csv(gold_a, index=False)
    pd.DataFrame([{"sample_id": "B1", "latitude": flare_row["latitude"], "longitude": flare_row["longitude"]}]).to_csv(gold_b, index=False)

    # single path (backward-compatible call shape): only the industrial cell is gone
    only_a = exclude_gold_cells(frame, gold_csv=gold_a)
    assert industrial_cell not in set(_compute_cell_ids(only_a))
    assert flare_cell in set(_compute_cell_ids(only_a))  # gold_b's cell wasn't excluded
    assert len(only_a) == len(frame) - (cell_id == industrial_cell).sum()

    # both paths together: both cells are gone
    both = exclude_gold_cells(frame, gold_csv=(gold_a, gold_b))
    remaining_cells = set(_compute_cell_ids(both))
    assert industrial_cell not in remaining_cells
    assert flare_cell not in remaining_cells
    assert len(both) < len(only_a)


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
