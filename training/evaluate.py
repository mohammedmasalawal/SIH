"""STEP 3: score training/artifacts/model.pkl -> training/artifacts/eval_summary.json.

Reproduces the model's own held-out spatial-block set deterministically (same CSV +
same random_seed recorded in the artifact's metadata), rather than persisting a
second held-out-rows file -- since a grid cell never straddles train/test, nothing
about that reproduction can leak train rows into the score.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from backend.classification.features import hotspots_to_feature_matrix
from backend.config import EVAL_SUMMARY_PATH, MODEL_ARTIFACT_PATH, RANDOM_SEED
from training.train_model import spatial_block_split


def evaluate(
    labeled_csv: str | Path,
    model_path: str | Path = MODEL_ARTIFACT_PATH,
    output_summary: str | Path = EVAL_SUMMARY_PATH,
) -> dict:
    model_path = Path(model_path)
    artifact = joblib.load(model_path)
    model = artifact["model"]
    metadata = artifact.get("metadata", {})

    labeled_csv = Path(labeled_csv)
    trained_on = metadata.get("training_data_path")
    if trained_on is not None and str(labeled_csv) != trained_on:
        print(
            f"Warning: evaluating against {labeled_csv}, but this model was trained on "
            f"{trained_on} -- the reproduced split may not match the model's own held-out set."
        )

    frame = pd.read_csv(labeled_csv)
    frame = frame[frame["label"] != "unknown"].reset_index(drop=True)
    seed = metadata.get("random_seed", RANDOM_SEED)
    _, test_idx = spatial_block_split(frame, seed=seed)
    if len(test_idx) == 0:
        raise ValueError("Reproduced spatial-block split has an empty test set for this CSV/seed.")

    test_frame = frame.loc[test_idx]
    x_test = hotspots_to_feature_matrix(test_frame)
    y_test = test_frame["label"]
    labels = list(artifact.get("label_classes", sorted(y_test.unique())))
    # model.predict returns integer class indices (0..n_classes-1) -- decode back to
    # label strings via the same label_classes ordering train_model.py encoded with.
    y_pred = [labels[int(index)] for index in model.predict(x_test)]

    summary = {
        "model_path": str(model_path),
        "labeled_csv": str(labeled_csv),
        "test_row_count": int(len(test_frame)),
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "classification_report": classification_report(y_test, y_pred, labels=labels, output_dict=True, zero_division=0),
        "confusion_matrix": confusion_matrix(y_test, y_pred, labels=labels).tolist(),
        "labels": labels,
    }
    output_summary = Path(output_summary)
    output_summary.parent.mkdir(parents=True, exist_ok=True)
    output_summary.write_text(json.dumps(summary, indent=2))
    return summary


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labeled-csv", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=MODEL_ARTIFACT_PATH)
    parser.add_argument("--output", type=Path, default=EVAL_SUMMARY_PATH)
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    summary = evaluate(args.labeled_csv, model_path=args.model, output_summary=args.output)
    print(f"Accuracy: {summary['accuracy']:.3f}")
    print(f"Wrote {args.output}")
