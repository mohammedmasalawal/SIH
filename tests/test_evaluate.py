from __future__ import annotations

import json

from training.evaluate import evaluate
from training.train_model import spatial_block_split, train


def test_evaluate_reproduces_same_test_indices_as_train(synthetic_labeled_frame, tmp_path):
    labeled_csv = tmp_path / "labeled.csv"
    synthetic_labeled_frame.to_csv(labeled_csv, index=False)
    model_path = tmp_path / "model.pkl"
    artifact = train(labeled_csv, output_artifact=model_path)

    frame = synthetic_labeled_frame[synthetic_labeled_frame["label"] != "unknown"].reset_index(drop=True)
    _, expected_test_idx = spatial_block_split(frame, seed=artifact["metadata"]["random_seed"])

    summary_path = tmp_path / "eval_summary.json"
    summary = evaluate(labeled_csv, model_path=model_path, output_summary=summary_path)

    assert summary["test_row_count"] == len(expected_test_idx)
    assert summary_path.exists()
    on_disk = json.loads(summary_path.read_text())
    assert on_disk["accuracy"] == summary["accuracy"]
    assert {"accuracy", "classification_report", "confusion_matrix", "labels"}.issubset(summary.keys())
