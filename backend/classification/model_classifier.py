"""Loads training/artifacts/model.pkl (if present and valid) and predicts live.

Every failure mode here -- missing file, corrupt file, wrong shape -- degrades to
"no model available" rather than raising, so classifier.py can fall back to
rules.py cleanly in every case. Only ModelClassifier.predict() raises, and only
when called on an unavailable model, which is a programming error in the caller
(classifier.py checks is_available first).
"""

from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd

from backend.classification.features import FEATURE_COLUMNS, hotspot_to_feature_dict
from backend.config import MODEL_ARTIFACT_PATH

_REQUIRED_ARTIFACT_KEYS = {"model", "feature_columns", "label_classes", "metadata"}


class ModelUnavailable(RuntimeError):
    """Raised by ModelClassifier.predict when no usable model artifact is loaded."""


def load_artifact(path: str | Path = MODEL_ARTIFACT_PATH) -> dict | None:
    """Load and validate a model.pkl artifact, returning None on any problem.

    Deliberately broad exception handling: joblib/pickle can raise many different
    exception types for a corrupt or version-incompatible file (UnpicklingError,
    EOFError, AttributeError, ModuleNotFoundError, ...) and every one of them means
    the same thing here -- "this artifact can't be trusted, fall back to rules."
    """
    path = Path(path)
    if not path.exists():
        return None
    try:
        artifact = joblib.load(path)
    except Exception:
        return None
    if not isinstance(artifact, dict) or not _REQUIRED_ARTIFACT_KEYS.issubset(artifact.keys()):
        return None
    if list(artifact.get("feature_columns") or []) != list(FEATURE_COLUMNS):
        # The live feature encoding has moved on since this model was trained --
        # refuse rather than risk silently-wrong predictions from misaligned columns.
        return None
    return artifact


class ModelClassifier:
    def __init__(self, artifact_path: str | Path = MODEL_ARTIFACT_PATH) -> None:
        self._artifact = load_artifact(artifact_path)

    @property
    def is_available(self) -> bool:
        return self._artifact is not None

    @property
    def metadata(self) -> dict | None:
        return self._artifact["metadata"] if self._artifact else None

    def predict(self, hotspot: dict) -> tuple[str, dict[str, float]]:
        if self._artifact is None:
            raise ModelUnavailable("No trained model artifact is loaded")
        model = self._artifact["model"]
        label_classes = self._artifact["label_classes"]
        vector = pd.DataFrame([hotspot_to_feature_dict(hotspot)]).reindex(columns=FEATURE_COLUMNS, fill_value=0.0)
        # xgboost's sklearn wrapper predicts an integer class index (0..n_classes-1),
        # not the original label string -- label_classes (from the training-time
        # LabelEncoder) decodes it back.
        predicted_index = int(model.predict(vector)[0])
        label = str(label_classes[predicted_index])
        probabilities: dict[str, float] = {}
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(vector)[0]
            probabilities = {str(cls): float(p) for cls, p in zip(label_classes, proba)}
        return label, probabilities
