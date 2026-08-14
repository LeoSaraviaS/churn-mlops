"""Tests del módulo de entrenamiento: features, pipeline, split, gate."""
from __future__ import annotations

import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from training.data import (
    ID_COLUMNS,
    SchemaValidationError,
    class_balance,
    clean_dataset,
    get_dataset,
    validate_schema,
)
from training.evaluate import compute_metrics, summarize_for_gate
from training.model import AGE_BINS, AGE_LABELS, add_age_group, build_model

DATA_PATH = "data/Churn_Modelling.csv"


def test_add_age_group_assigns_correct_bins():
    df = pd.DataFrame({"Age": [20, 35, 60]})
    result = add_age_group(df)
    assert list(result["AgeGroup"]) == ["Young", "Adult", "Old"]
    assert AGE_BINS == [18, 30, 50, 100]
    assert AGE_LABELS == ["Young", "Adult", "Old"]


def test_build_model_returns_pipeline_with_expected_steps():
    model = build_model()
    assert isinstance(model, Pipeline)
    step_names = [name for name, _ in model.steps]
    assert step_names == ["age_group", "preprocessing", "clf"]


def test_validate_schema_raises_on_missing_columns():
    df = pd.DataFrame({"CreditScore": [1]})
    with pytest.raises(SchemaValidationError):
        validate_schema(df)


def test_clean_dataset_drops_identifier_columns():
    raw = pd.read_csv(DATA_PATH, nrows=50)
    clean = clean_dataset(raw)
    for col in ID_COLUMNS:
        assert col not in clean.columns


def test_class_balance_percentages_sum_to_100():
    raw = pd.read_csv(DATA_PATH)
    balance = class_balance(raw)
    total_pct = balance["class_0_stayed"]["pct"] + balance["class_1_churned"]["pct"]
    assert round(total_pct) == 100
    assert balance["total"] == len(raw)


def test_get_dataset_split_is_stratified():
    X_train, X_test, y_train, y_test = get_dataset(DATA_PATH, test_size=0.2, random_state=42)
    assert len(X_train) + len(X_test) > 0
    train_churn_rate = y_train.mean()
    test_churn_rate = y_test.mean()
    # La proporción de churn en train y test debe ser muy similar gracias a stratify=y.
    assert abs(train_churn_rate - test_churn_rate) < 0.02


def test_summarize_for_gate_fails_below_threshold():
    fake_metrics = {"roc_auc": 0.5}
    passed, msg = summarize_for_gate(fake_metrics, roc_auc_threshold=0.75)
    assert passed is False
    assert "NO PASA" in msg


def test_summarize_for_gate_passes_above_threshold():
    fake_metrics = {"roc_auc": 0.9}
    passed, msg = summarize_for_gate(fake_metrics, roc_auc_threshold=0.75)
    assert passed is True
    assert "PASA" in msg


def test_compute_metrics_returns_expected_keys():
    y_true = [0, 1, 0, 1]
    y_pred = [0, 1, 0, 0]
    y_proba = [0.1, 0.9, 0.2, 0.4]
    metrics = compute_metrics(y_true, y_pred, y_proba)
    for key in ["accuracy", "precision", "recall", "f1_score", "roc_auc", "confusion_matrix"]:
        assert key in metrics
