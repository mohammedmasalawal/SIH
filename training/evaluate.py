"""STEP 3: score training/artifacts/model.pkl -> training/artifacts/eval_summary.json.

Reproduces the model's own held-out set deterministically (same CSV + same
random_seed recorded in the artifact's metadata, via the same
prepare_training_frame() train_model.py used) -- since gold cells are excluded the
same way in both places, nothing about that reproduction can leak train or gold rows
into the score.

Two split strategies exist (see training/train_model.py's module docstring for the
full rationale): "detection" (train() uses this by default -- every detection
on/after a row-count cutoff goes to test, regardless of cell) and "spatial_block"
(no physical site straddles train/test, but on this data starves gas flare/
industrial down to single-digit test rows). evaluate() reports whichever `split` is
asked for; evaluate_both_splits() runs both against the same model and returns them
side by side, so a reviewer never has to take just one split's word for it.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from backend.classification.features import hotspots_to_feature_matrix
from backend.config import EVAL_SUMMARY_PATH, GOLD_SAMPLE_PATH, MODEL_ARTIFACT_PATH, RANDOM_SEED
from training.train_model import SPLIT_STRATEGIES, prepare_training_frame


def evaluate(
    labeled_csv: str | Path,
    model_path: str | Path = MODEL_ARTIFACT_PATH,
    output_summary: str | Path = EVAL_SUMMARY_PATH,
    gold_csv: str | Path = GOLD_SAMPLE_PATH,
    split: str | None = None,
) -> dict:
    """Score model_path against one split. `split` defaults to the model's own
    recorded split_strategy (metadata["split_strategy"], "detection" if the artifact
    predates that field) -- pass "spatial_block" explicitly to score the same model
    against the other split as a secondary, reported-not-trained-on evaluation.
    """
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

    if split is None:
        split = metadata.get("split_strategy", "detection")
    seed = metadata.get("random_seed", RANDOM_SEED)
    safe_frame, train_idx, test_idx, cutoff_date = prepare_training_frame(
        labeled_csv, gold_csv=gold_csv, seed=seed, split=split
    )
    if len(test_idx) == 0:
        raise ValueError(f"Reproduced {split} split has an empty test set for this CSV/seed.")

    train_frame = safe_frame.loc[train_idx]
    test_frame = safe_frame.loc[test_idx]
    x_test = hotspots_to_feature_matrix(test_frame)
    y_test = test_frame["label"]
    labels = list(artifact.get("label_classes", sorted(y_test.unique())))
    # model.predict returns integer class indices (0..n_classes-1) -- decode back to
    # label strings via the same label_classes ordering train_model.py encoded with.
    y_pred = [labels[int(index)] for index in model.predict(x_test)]

    train_label_counts = train_frame["label"].value_counts()
    test_label_counts = test_frame["label"].value_counts()
    per_class = {
        label: {
            "train_n": int(train_label_counts.get(label, 0)),
            "train_pct": round(100 * train_label_counts.get(label, 0) / max(len(train_frame), 1), 1),
            "test_n": int(test_label_counts.get(label, 0)),
            "test_pct": round(100 * test_label_counts.get(label, 0) / max(len(test_frame), 1), 1),
        }
        for label in labels
    }

    summary = {
        "model_path": str(model_path),
        "labeled_csv": str(labeled_csv),
        "split_strategy": split,
        "split_cutoff_date": str(cutoff_date),
        "train_row_count": int(len(train_frame)),
        "test_row_count": int(len(test_frame)),
        "per_class": per_class,
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "classification_report": classification_report(y_test, y_pred, labels=labels, output_dict=True, zero_division=0),
        "confusion_matrix": confusion_matrix(y_test, y_pred, labels=labels).tolist(),
        "labels": labels,
    }
    output_summary = Path(output_summary)
    output_summary.parent.mkdir(parents=True, exist_ok=True)
    output_summary.write_text(json.dumps(summary, indent=2))
    return summary


def evaluate_both_splits(
    labeled_csv: str | Path,
    model_path: str | Path = MODEL_ARTIFACT_PATH,
    output_dir: str | Path = None,
    gold_csv: str | Path = GOLD_SAMPLE_PATH,
) -> dict[str, dict]:
    """Score the same fitted model against both split strategies and return both
    summaries -- so a reviewer sees the detection-level numbers the model was
    actually trained/selected against, and the spatial-block numbers as an explicit
    secondary check, side by side rather than one being silently preferred.
    """
    output_dir = Path(output_dir) if output_dir is not None else Path(EVAL_SUMMARY_PATH).parent
    results = {}
    for split in SPLIT_STRATEGIES:
        results[split] = evaluate(
            labeled_csv,
            model_path=model_path,
            output_summary=output_dir / f"eval_summary_{split}.json",
            gold_csv=gold_csv,
            split=split,
        )
    return results


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labeled-csv", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=MODEL_ARTIFACT_PATH)
    parser.add_argument("--output", type=Path, default=EVAL_SUMMARY_PATH)
    parser.add_argument("--split", choices=(*SPLIT_STRATEGIES, "both"), default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    if args.split == "both":
        both = evaluate_both_splits(args.labeled_csv, model_path=args.model, output_dir=args.output.parent)
        for split_name, summary in both.items():
            print(f"[{split_name}] accuracy={summary['accuracy']:.3f} test_row_count={summary['test_row_count']}")
            for label, counts in summary["per_class"].items():
                print(f"  {label:<22} train={counts['train_n']:>5} ({counts['train_pct']:>5.1f}%)  test={counts['test_n']:>4} ({counts['test_pct']:>5.1f}%)")
    else:
        summary = evaluate(args.labeled_csv, model_path=args.model, output_summary=args.output, split=args.split)
        print(f"[{summary['split_strategy']}] Accuracy: {summary['accuracy']:.3f}")
        print(f"Wrote {args.output}")
