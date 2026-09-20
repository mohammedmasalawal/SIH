from __future__ import annotations

import json

from training.evaluate import evaluate, evaluate_both_splits
from training.train_model import detection_level_temporal_split, spatial_block_split, train


def test_evaluate_reproduces_same_test_indices_as_train(synthetic_labeled_frame, tmp_path):
    """train()'s default split is "detection" -- evaluate() with no explicit split
    must reproduce that, not spatial_block_split (a different split entirely)."""
    labeled_csv = tmp_path / "labeled.csv"
    synthetic_labeled_frame.to_csv(labeled_csv, index=False)
    model_path = tmp_path / "model.pkl"
    artifact = train(labeled_csv, output_artifact=model_path)
    assert artifact["metadata"]["split_strategy"] == "detection"

    frame = synthetic_labeled_frame[synthetic_labeled_frame["label"] != "unknown"].reset_index(drop=True)
    _, expected_test_idx, _ = detection_level_temporal_split(frame)

    summary_path = tmp_path / "eval_summary.json"
    summary = evaluate(labeled_csv, model_path=model_path, output_summary=summary_path)

    assert summary["split_strategy"] == "detection"
    assert summary["test_row_count"] == len(expected_test_idx)
    assert summary_path.exists()
    on_disk = json.loads(summary_path.read_text())
    assert on_disk["accuracy"] == summary["accuracy"]
    assert {"accuracy", "classification_report", "confusion_matrix", "labels", "per_class"}.issubset(summary.keys())


def test_evaluate_can_score_spatial_block_as_a_secondary_split(synthetic_labeled_frame, tmp_path):
    """A model trained on the "detection" split can still be scored against
    "spatial_block" explicitly -- the secondary, reported-not-trained-on evaluation."""
    labeled_csv = tmp_path / "labeled.csv"
    synthetic_labeled_frame.to_csv(labeled_csv, index=False)
    model_path = tmp_path / "model.pkl"
    artifact = train(labeled_csv, output_artifact=model_path)

    frame = synthetic_labeled_frame[synthetic_labeled_frame["label"] != "unknown"].reset_index(drop=True)
    _, expected_test_idx = spatial_block_split(frame, seed=artifact["metadata"]["random_seed"])

    summary_path = tmp_path / "eval_summary_spatial.json"
    summary = evaluate(labeled_csv, model_path=model_path, output_summary=summary_path, split="spatial_block")

    assert summary["split_strategy"] == "spatial_block"
    assert summary["test_row_count"] == len(expected_test_idx)


def test_evaluate_both_splits_reports_both_without_preferring_one(synthetic_labeled_frame, tmp_path):
    labeled_csv = tmp_path / "labeled.csv"
    synthetic_labeled_frame.to_csv(labeled_csv, index=False)
    model_path = tmp_path / "model.pkl"
    train(labeled_csv, output_artifact=model_path)

    results = evaluate_both_splits(labeled_csv, model_path=model_path, output_dir=tmp_path)

    assert set(results.keys()) == {"detection", "spatial_block"}
    assert results["detection"]["split_strategy"] == "detection"
    assert results["spatial_block"]["split_strategy"] == "spatial_block"
    assert (tmp_path / "eval_summary_detection.json").exists()
    assert (tmp_path / "eval_summary_spatial_block.json").exists()
