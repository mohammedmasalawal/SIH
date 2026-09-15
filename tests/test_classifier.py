from __future__ import annotations

import joblib
import pandas as pd

from backend.classification import classifier
from backend.classification.model_classifier import ModelClassifier
from backend.classification.rules import classify as rules_classify


def test_classify_hotspot_falls_back_to_rules_when_no_artifact(monkeypatch, tmp_path, sample_hotspot_dict):
    monkeypatch.setattr(classifier, "_model_classifier", ModelClassifier(tmp_path / "missing.pkl"))
    result = classifier.classify_hotspot(sample_hotspot_dict)
    expected_label = rules_classify(pd.Series(sample_hotspot_dict))
    assert result.label == expected_label
    assert result.label_source in ("rule", "unknown")
    assert not classifier.model_is_available()


def test_classify_hotspot_falls_back_to_rules_when_artifact_corrupt(monkeypatch, tmp_path, sample_hotspot_dict):
    corrupt_path = tmp_path / "corrupt.pkl"
    joblib.dump("not a dict", corrupt_path)
    monkeypatch.setattr(classifier, "_model_classifier", ModelClassifier(corrupt_path))
    result = classifier.classify_hotspot(sample_hotspot_dict)
    expected_label = rules_classify(pd.Series(sample_hotspot_dict))
    assert result.label == expected_label
    assert result.label_source in ("rule", "unknown")


def test_classify_hotspot_uses_model_when_available(monkeypatch, tiny_fitted_artifact, sample_hotspot_dict):
    monkeypatch.setattr(classifier, "_model_classifier", ModelClassifier(tiny_fitted_artifact))
    result = classifier.classify_hotspot(sample_hotspot_dict)
    assert result.label_source == "model"
    assert result.probabilities
    assert classifier.model_is_available()
