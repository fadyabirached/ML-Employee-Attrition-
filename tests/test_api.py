"""Tests for the FastAPI service in api.py.

Model loading is patched to a lightweight fake so these tests don't need a
real trained model or dataset.csv on disk -- they only check the API layer
(request validation, response shape), not the model itself.
"""
import numpy as np
import pytest
from fastapi.testclient import TestClient

import api

VALID_PAYLOAD = {
    "satisfaction_level": 5.0,
    "last_evaluation_rating": 6.0,
    "projects_worked_on": 3,
    "average_monthly_hours": 160,
    "time_spend_company": 3,
    "Work_accident": 0,
    "promotion_last_5years": 0,
    "Department": "sales",
    "salary": "low",
}


class _FakePreprocessor:
    def transform(self, row):
        return np.zeros((1, 3))


class _FakeIndex:
    def kneighbors(self, embedded):
        return np.array([[0.5]]), np.array([[0]])


class _FakeModel:
    def __init__(self):
        self.named_steps = {"prep": _FakePreprocessor()}

    def predict_proba(self, row):
        return np.array([[0.7, 0.3]])


@pytest.fixture
def client(monkeypatch):
    fake_bundle = {
        "model": _FakeModel(),
        "threshold": 0.384,
        "features": list(VALID_PAYLOAD.keys()),
        "base_rate": 0.24,
        "feature_ranges": {},
        "ood": {"index": _FakeIndex(), "threshold": 2.0},
    }
    monkeypatch.setattr(api, "load_or_train_model", lambda *a, **kw: fake_bundle)

    with TestClient(api.app) as test_client:
        yield test_client


def test_health_endpoint(client):
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_predict_returns_probability_flag_and_latency(client):
    response = client.post("/predict", json=VALID_PAYLOAD)

    assert response.status_code == 200
    body = response.json()
    assert body["probability_leave"] == pytest.approx(0.3)
    assert body["will_leave"] is False
    assert body["latency_ms"] >= 0


def test_predict_reports_in_distribution_profiles_as_such(client):
    response = client.post("/predict", json=VALID_PAYLOAD)

    body = response.json()
    assert body["out_of_distribution"] is False
    assert body["novelty_score"] == pytest.approx(0.25)  # 0.5 / 2.0


def test_predict_rejects_satisfaction_below_the_training_minimum(client):
    """0.5 is a plausible-looking score but no employee in the data scores
    below 0.9, so the model has no basis for it."""
    bad_payload = {**VALID_PAYLOAD, "satisfaction_level": 0.5}

    response = client.post("/predict", json=bad_payload)

    assert response.status_code == 422


def test_predict_rejects_satisfaction_above_the_training_maximum(client):
    bad_payload = {**VALID_PAYLOAD, "satisfaction_level": 15.0}

    response = client.post("/predict", json=bad_payload)

    assert response.status_code == 422


def test_predict_rejects_tenure_below_the_training_minimum(client):
    bad_payload = {**VALID_PAYLOAD, "time_spend_company": 1}

    response = client.post("/predict", json=bad_payload)

    assert response.status_code == 422


def test_predict_rejects_project_count_beyond_the_training_maximum(client):
    bad_payload = {**VALID_PAYLOAD, "projects_worked_on": 9}

    response = client.post("/predict", json=bad_payload)

    assert response.status_code == 422


def test_predict_rejects_unknown_salary_level(client):
    bad_payload = {**VALID_PAYLOAD, "salary": "platinum"}

    response = client.post("/predict", json=bad_payload)

    assert response.status_code == 422


def test_predict_rejects_missing_field(client):
    incomplete_payload = {k: v for k, v in VALID_PAYLOAD.items() if k != "Department"}

    response = client.post("/predict", json=incomplete_payload)

    assert response.status_code == 422
