from __future__ import annotations

import joblib
import pytest

from backend.classification.model_classifier import ModelClassifier, ModelUnavailable, load_artifact
from backend.classification.rules import CLASSES


def test_load_artifact_missing_file_returns_none(tmp_path):
    assert load_artifact(tmp_path / "does_not_exist.pkl") is None


def test_load_artifact_corrupt_file_returns_none(tmp_path):
    path = tmp_path / "corrupt.pkl"
    joblib.dump("not a dict", path)
    assert load_artifact(path) is None


def test_load_artifact_feature_mismatch_returns_none(tiny_fitted_artifact, tmp_path):
    artifact = joblib.load(tiny_fitted_artifact)
    artifact["feature_columns"] = ["totally", "different", "columns"]
    mismatched_path = tmp_path / "mismatched.pkl"
    joblib.dump(artifact, mismatched_path)
    assert load_artifact(mismatched_path) is None


def test_load_artifact_valid_file_loads(tiny_fitted_artifact):
    artifact = load_artifact(tiny_fitted_artifact)
    assert artifact is not None
    assert "model" in artifact


def test_model_classifier_predicts_known_label_with_probabilities(tiny_fitted_artifact, sample_hotspot_dict):
    model_classifier = ModelClassifier(tiny_fitted_artifact)
    assert model_classifier.is_available
    label, probabilities = model_classifier.predict(sample_hotspot_dict)
    assert label in CLASSES
    assert probabilities
    assert sum(probabilities.values()) == pytest.approx(1.0, abs=1e-3)


def test_model_classifier_unavailable_without_artifact(tmp_path, sample_hotspot_dict):
    model_classifier = ModelClassifier(tmp_path / "missing.pkl")
    assert not model_classifier.is_available
    with pytest.raises(ModelUnavailable):
        model_classifier.predict(sample_hotspot_dict)
