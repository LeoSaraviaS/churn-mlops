"""Tests del wrapper de predicción, usando el modelo real entrenado.

Requiere que `models/model.joblib` exista (se genera con
`python -m training.train`). El job `test` de CI/CD lo garantiza corriendo
`train-and-gate` -o un entrenamiento de smoke- antes de `pytest`.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from app.predictor import Predictor
from app.schemas import ChurnPredictionRequest

MODEL_PATH = "models/model.joblib"
METADATA_PATH = "models/metadata.json"

pytestmark = pytest.mark.skipif(
    not Path(MODEL_PATH).exists(),
    reason="models/model.joblib no existe: correr 'python -m training.train' primero.",
)


@pytest.fixture(scope="module")
def predictor() -> Predictor:
    p = Predictor(MODEL_PATH, METADATA_PATH)
    p.load()
    return p


def test_predict_returns_valid_probability(predictor: Predictor):
    request = ChurnPredictionRequest(
        CreditScore=650,
        Geography="France",
        Gender="Female",
        Age=40,
        Tenure=3,
        Balance=75000.0,
        NumOfProducts=2,
        HasCrCard=1,
        IsActiveMember=1,
        EstimatedSalary=100000.0,
    )
    prediction, probability = predictor.predict(request)
    assert prediction in (0, 1)
    assert 0.0 <= probability <= 1.0


def test_risk_level_thresholds():
    assert Predictor.risk_level(0.1) == "low"
    assert Predictor.risk_level(0.5) == "medium"
    assert Predictor.risk_level(0.9) == "high"


def test_predict_is_deterministic_for_same_input(predictor: Predictor):
    request = ChurnPredictionRequest(
        CreditScore=700,
        Geography="Germany",
        Gender="Male",
        Age=55,
        Tenure=8,
        Balance=120000.0,
        NumOfProducts=1,
        HasCrCard=0,
        IsActiveMember=0,
        EstimatedSalary=50000.0,
    )
    pred1, proba1 = predictor.predict(request)
    pred2, proba2 = predictor.predict(request)
    assert pred1 == pred2
    assert proba1 == proba2
