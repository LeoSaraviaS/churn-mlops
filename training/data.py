"""Carga y preparación de datos para el modelo de churn.

Responsabilidades:
    - Leer el CSV crudo (Churn_Modelling.csv).
    - Eliminar columnas identificadoras que no deben usarse como features
      (RowNumber, CustomerId, Surname) — no aportan señal predictiva real
      y, en el caso de CustomerId/Surname, son datos que además se excluyen
      explícitamente de la documentación pública del proyecto (README).
    - Validar que el esquema esperado esté presente (fail-fast si el CSV
      cambia de forma inesperada).
    - Generar el split train/test.

Decisión documentada (desviación respecto al notebook de referencia):
    El notebook original (ih8asham/customer-churn-dataset) usa
    `train_test_split(..., test_size=0.2, random_state=0)` SIN `stratify`.
    Aquí se agrega `stratify=y` porque el dataset tiene desbalance de clases
    (~20% churn / ~80% no-churn); sin estratificar, el split aleatorio puede
    generar folds de test con una proporción de churn distinta a la real,
    lo que sesga la evaluación. Se documenta como mejora explícita, no como
    un error del notebook original.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

# Columnas identificadoras: no son features, se descartan siempre.
ID_COLUMNS = ["RowNumber", "CustomerId", "Surname"]

# Columna objetivo.
TARGET_COLUMN = "Exited"

# Columnas de features esperadas después de eliminar ID_COLUMNS y TARGET_COLUMN.
EXPECTED_FEATURE_COLUMNS = [
    "CreditScore",
    "Geography",
    "Gender",
    "Age",
    "Tenure",
    "Balance",
    "NumOfProducts",
    "HasCrCard",
    "IsActiveMember",
    "EstimatedSalary",
]


class SchemaValidationError(ValueError):
    """Se lanza cuando el CSV no tiene las columnas esperadas."""


def load_raw_dataset(csv_path: str | Path) -> pd.DataFrame:
    """Lee el CSV crudo tal cual viene, sin transformar."""
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"No se encontró el dataset en: {csv_path}")
    return pd.read_csv(csv_path)


def validate_schema(df: pd.DataFrame) -> None:
    """Verifica que estén todas las columnas necesarias antes de continuar."""
    required = set(ID_COLUMNS + EXPECTED_FEATURE_COLUMNS + [TARGET_COLUMN])
    missing = required - set(df.columns)
    if missing:
        raise SchemaValidationError(
            f"Faltan columnas requeridas en el dataset: {sorted(missing)}"
        )


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Elimina columnas identificadoras que no deben usarse como features."""
    validate_schema(df)
    return df.drop(columns=ID_COLUMNS)


def class_balance(df: pd.DataFrame, target_column: str = TARGET_COLUMN) -> dict:
    """Retorna el desbalance de clases real como conteos y porcentajes.

    Se usa tanto en el pipeline de entrenamiento (para loguearlo en
    metadata.json) como para documentar el porcentaje real en el README.
    """
    counts = df[target_column].value_counts().sort_index()
    total = int(counts.sum())
    return {
        "total": total,
        "class_0_stayed": {
            "count": int(counts.get(0, 0)),
            "pct": round(100 * counts.get(0, 0) / total, 2),
        },
        "class_1_churned": {
            "count": int(counts.get(1, 0)),
            "pct": round(100 * counts.get(1, 0) / total, 2),
        },
    }


def get_dataset(
    csv_path: str | Path,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Carga, limpia y separa el dataset en train/test.

    Devuelve (X_train, X_test, y_train, y_test). El split usa
    `stratify=y` para preservar la proporción de churn en ambos folds
    (ver docstring del módulo).
    """
    raw = load_raw_dataset(csv_path)
    clean = clean_dataset(raw)

    X = clean.drop(columns=[TARGET_COLUMN])
    y = clean[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )
    return X_train, X_test, y_train, y_test
