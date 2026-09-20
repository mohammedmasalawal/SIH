"""Picks the trained model if one is loaded and usable, else falls back to rules.py.

classify_hotspot always returns a ClassifyResult -- it never raises to callers
(backend/api/routes.py in particular). The model is tried first only when
ModelClassifier reports it loaded successfully at process start; even then, any
runtime failure during predict() (not just a load-time failure) falls back to the
rule engine rather than propagating.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from backend.classification import rules
from backend.classification.model_classifier import ModelClassifier

# Loaded once at process start, not per-request -- re-reading and re-validating
# model.pkl on every classify call would be wasted work for an artifact that only
# changes when someone re-runs training/train_model.py and restarts the process.
_model_classifier = ModelClassifier()


@dataclass
class ClassifyResult:
    label: str
    # "model" is specific to this live-serving path: it means _model_classifier
    # produced the label, not a rule. It is never written by training/build_labels.py's
    # batch rules.apply_rules(), whose label_source contract is rule/conflict/unknown --
    # classify_hotspot's rule fallback below calls rules.classify() (single-row, no
    # "conflict" concept) rather than apply_rules(), so "conflict" cannot appear here.
    label_source: str  # "model" | "rule" | "unknown"
    probabilities: dict[str, float] | None = None
    model_version: str | None = None


def classify_hotspot(hotspot: dict) -> ClassifyResult:
    if _model_classifier.is_available:
        try:
            label, probabilities = _model_classifier.predict(hotspot)
            metadata = _model_classifier.metadata or {}
            trained_at = metadata.get("trained_at")
            return ClassifyResult(
                label=label,
                label_source="model",
                probabilities=probabilities,
                model_version=str(trained_at) if trained_at else None,
            )
        except Exception:
            pass  # any predict-time failure falls through to the rule engine below

    label = rules.classify(pd.Series(hotspot))
    label_source = "rule" if label != "unknown" else "unknown"
    return ClassifyResult(label=label, label_source=label_source)


def model_is_available() -> bool:
    return _model_classifier.is_available
