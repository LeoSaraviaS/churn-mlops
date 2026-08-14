"""Tests de validación de contratos Pydantic (garantizan 422, nunca 500,
ante input inválido)."""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.schemas import ChurnPredictionRequest

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


def test_valid_payload_parses_correctly():
    request = ChurnPredictionRequest(**VALID_PAYLOAD)
    assert request.CreditScore == 650


def test_invalid_geography_raises():
    payload = {**VALID_PAYLOAD, "Geography": "Chile"}
    with pytest.raises(ValidationError):
        ChurnPredictionRequest(**payload)


def test_credit_score_out_of_range_raises():
    payload = {**VALID_PAYLOAD, "CreditScore": 100}
    with pytest.raises(ValidationError):
        ChurnPredictionRequest(**payload)


def test_negative_balance_raises():
    payload = {**VALID_PAYLOAD, "Balance": -500}
    with pytest.raises(ValidationError):
        ChurnPredictionRequest(**payload)


def test_has_cr_card_must_be_zero_or_one():
    payload = {**VALID_PAYLOAD, "HasCrCard": 2}
    with pytest.raises(ValidationError):
        ChurnPredictionRequest(**payload)


def test_missing_required_field_raises():
    payload = {k: v for k, v in VALID_PAYLOAD.items() if k != "Age"}
    with pytest.raises(ValidationError):
        ChurnPredictionRequest(**payload)
