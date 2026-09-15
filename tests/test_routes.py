from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from backend.classification import classifier
from backend.classification.model_classifier import ModelClassifier
from backend.classification.rules import CLASSES
from main import app


@pytest.fixture
def client(monkeypatch, tmp_path):
    """Force 'no model loaded' for most tests so responses come from the deterministic rule fallback."""
    monkeypatch.setattr(classifier, "_model_classifier", ModelClassifier(tmp_path / "no_model.pkl"))
    return TestClient(app)


def _base_payload() -> dict:
    return {"latitude": 22.30, "longitude": 69.85, "acq_date": "2026-01-03", "acq_time": "0130", "satellite": "N"}


def test_valid_classify_request_returns_200(client):
    response = client.post("/api/classify", json=_base_payload())
    assert response.status_code == 200
    assert response.json()["label"] in CLASSES


def test_missing_latitude_returns_422(client):
    payload = _base_payload()
    del payload["latitude"]
    response = client.post("/api/classify", json=payload)
    assert response.status_code == 422
    assert "latitude" in response.text


def test_negative_distance_is_accepted_not_rejected(client):
    payload = _base_payload() | {"dist_to_industrial_m": -50.0}
    response = client.post("/api/classify", json=payload)
    assert response.status_code == 200


def test_null_landcover_is_accepted(client):
    payload = _base_payload() | {"landcover_class": None}
    response = client.post("/api/classify", json=payload)
    assert response.status_code == 200


def test_extreme_frp_is_accepted(client):
    payload = _base_payload() | {"frp": 999_999.0}
    response = client.post("/api/classify", json=payload)
    assert response.status_code == 200


def test_unrecognized_facility_type_is_accepted_not_rejected(client):
    payload = _base_payload() | {"nearest_heat_facility_type": "not-a-real-type"}
    response = client.post("/api/classify", json=payload)
    assert response.status_code == 200


def test_health_reflects_no_model_loaded(client):
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json()["model_loaded"] is False


def test_health_reflects_model_loaded(monkeypatch, tiny_fitted_artifact):
    monkeypatch.setattr(classifier, "_model_classifier", ModelClassifier(tiny_fitted_artifact))
    response = TestClient(app).get("/api/health")
    assert response.json()["model_loaded"] is True
