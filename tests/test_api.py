"""Tests de integración de la API (FastAPI TestClient)."""
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


VALID_PAYLOAD = {
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


def test_health_returns_ok(client):
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True


def test_root_returns_service_info(client):
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["service"] == "churn-api"


def test_predict_valid_payload_returns_prediction(client):
    response = client.post("/predict", json=VALID_PAYLOAD)
    assert response.status_code == 200
    body = response.json()
    assert body["churn_prediction"] in (0, 1)
    assert 0.0 <= body["churn_probability"] <= 1.0
    assert body["risk_level"] in ("low", "medium", "high")


def test_predict_invalid_payload_returns_422_not_500(client):
    invalid_payload = {**VALID_PAYLOAD, "Geography": "InvalidCountry"}
    response = client.post("/predict", json=invalid_payload)
    assert response.status_code == 422

def test_model_schema_returns_input_and_output_contracts(client):
    response = client.get("/model/schema")
    assert response.status_code == 200
    body = response.json()
    assert "model_version" in body
    assert body["input_schema"]["title"] == "ChurnPredictionRequest"
    assert body["output_schema"]["title"] == "ChurnPredictionResponse"
    assert "CreditScore" in body["input_schema"]["properties"]


def test_predict_missing_field_returns_422(client):
    payload = {k: v for k, v in VALID_PAYLOAD.items() if k != "CreditScore"}
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_predict_batch_returns_one_prediction_per_customer(client):
    response = client.post("/predict/batch", json={"customers": [VALID_PAYLOAD, VALID_PAYLOAD]})
    assert response.status_code == 200
    body = response.json()
    assert len(body["predictions"]) == 2
    assert body["n_customers"] == 2


def test_predict_batch_matches_individual_predict(client):
    """El batch no puede dar un resultado distinto al endpoint de a uno."""
    individual = client.post("/predict", json=VALID_PAYLOAD).json()["churn_probability"]
    batch = client.post("/predict/batch", json={"customers": [VALID_PAYLOAD]}).json()
    assert batch["predictions"][0]["churn_probability"] == individual


def test_predict_batch_summary_is_consistent_with_predictions(client):
    response = client.post(
        "/predict/batch", json={"customers": [VALID_PAYLOAD, VALID_PAYLOAD, VALID_PAYLOAD]}
    )
    body = response.json()
    probabilidades = [p["churn_probability"] for p in body["predictions"]]
    assert body["mean_churn_probability"] == round(sum(probabilidades) / len(probabilidades), 4)
    assert body["n_predicted_churn"] == sum(p["churn_prediction"] for p in body["predictions"])


def test_predict_batch_rejects_empty_list(client):
    response = client.post("/predict/batch", json={"customers": []})
    assert response.status_code == 422


def test_predict_batch_rejects_a_single_invalid_customer_in_the_list(client):
    invalid = {**VALID_PAYLOAD, "Geography": "InvalidCountry"}
    response = client.post("/predict/batch", json={"customers": [VALID_PAYLOAD, invalid]})
    assert response.status_code == 422


def test_predict_batch_rejects_batch_over_the_configured_limit(client):
    from app.config import get_settings

    limit = get_settings().max_batch_size
    response = client.post("/predict/batch", json={"customers": [VALID_PAYLOAD] * (limit + 1)})
    assert response.status_code == 413
