"""Tests del endpoint de métricas de servicio (bonus: latencia, requests, errores)."""
from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

MODEL_PATH = "models/model.joblib"

pytestmark = pytest.mark.skipif(
    not Path(MODEL_PATH).exists(),
    reason="models/model.joblib no existe: correr 'python -m training.train' primero.",
)


@pytest.fixture(scope="module")
def client():
    from app.main import app

    with TestClient(app) as c:
        yield c


def test_metrics_endpoint_exposes_prometheus_format(client):
    # Generar al menos un request para que haya datos que exponer.
    client.get("/health")
    response = client.get("/metrics")
    assert response.status_code == 200
    assert "text/plain" in response.headers["content-type"]
    body = response.text
    assert "http_requests_total" in body
    assert "http_request_duration_seconds" in body


def test_metrics_counts_predictions(client):
    payload = {
        "CreditScore": 650,
        "Geography": "France",
        "Gender": "Female",
        "Age": 40,
        "Tenure": 3,
        "Balance": 75000.0,
        "NumOfProducts": 2,
        "HasCrCard": 1,
        "IsActiveMember": 1,
        "EstimatedSalary": 100000.0,
    }
    client.post("/predict", json=payload)
    response = client.get("/metrics")
    assert "churn_predictions_total" in response.text
