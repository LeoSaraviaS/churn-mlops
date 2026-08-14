"""Definición del pipeline de modelado: regresión logística para churn.

Fiel al notebook de referencia (ih8asham/customer-churn-dataset, celda 27,
"COMPLETE WORKING PIPELINE") en cuanto a: features utilizadas, ingeniería de
`AgeGroup` a partir de `Age`, y algoritmo base (LogisticRegression).

Desviaciones documentadas respecto al notebook (mejoras explícitas para
servir el modelo como API, no cambios al criterio de modelado):

1. `pd.get_dummies` -> `ColumnTransformer` + `OneHotEncoder(drop="first",
   handle_unknown="ignore")`. El notebook usa `pd.get_dummies` sobre todo
   el DataFrame, lo cual funciona para evaluación batch pero se rompe en
   producción: al recibir una sola fila por request HTTP, `get_dummies`
   generaría columnas dummy distintas según qué categorías aparecen en esa
   fila (training/serving skew). `ColumnTransformer` con `OneHotEncoder`
   fittea las categorías conocidas una sola vez (en entrenamiento) y las
   aplica de forma consistente a cualquier fila nueva, incluida una sola
   fila de inferencia.
2. `LogisticRegression(class_weight="balanced")`. El notebook no pondera
   clases. Dado el desbalance real del dataset (~20%/80%), se agrega
   `class_weight="balanced"` para que el modelo no colapse prediciendo
   siempre "no churn". Se reporta precision/recall/F1/ROC-AUC (no sólo
   accuracy) precisamente por esta razón.
3. `AgeGroup` se calcula dentro del pipeline (`FunctionTransformer`) en vez
   de precalcularse sobre todo el DataFrame antes del split. Así se
   garantiza que la misma transformación se aplique en tiempo de inferencia
   sobre datos crudos (una fila nueva desde la API), sin duplicar lógica
   entre entrenamiento y servicio.
"""
from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

# Bins fieles al notebook de referencia (celda 25): Young/Adult/Old.
AGE_BINS = [18, 30, 50, 100]
AGE_LABELS = ["Young", "Adult", "Old"]

CATEGORICAL_FEATURES = ["Geography", "Gender", "AgeGroup"]
NUMERIC_FEATURES = [
    "CreditScore",
    "Age",
    "Tenure",
    "Balance",
    "NumOfProducts",
    "HasCrCard",
    "IsActiveMember",
    "EstimatedSalary",
]


def add_age_group(df: pd.DataFrame) -> pd.DataFrame:
    """Deriva `AgeGroup` a partir de `Age` (fiel al notebook, celda 25).

    Se implementa como paso del pipeline (no como preprocesamiento externo)
    para que se aplique automáticamente tanto en entrenamiento como en cada
    predicción individual servida por la API.
    """
    df = df.copy()
    df["AgeGroup"] = pd.cut(df["Age"], bins=AGE_BINS, labels=AGE_LABELS)
    return df


def build_model() -> Pipeline:
    """Construye el pipeline completo: ingeniería de features + escalado +
    codificación + regresión logística.
    """
    preprocessing = ColumnTransformer(
        transformers=[
            (
                "cat",
                OneHotEncoder(drop="first", handle_unknown="ignore"),
                CATEGORICAL_FEATURES,
            ),
            ("num", StandardScaler(), NUMERIC_FEATURES),
        ]
    )

    return Pipeline(
        steps=[
            ("age_group", FunctionTransformer(add_age_group)),
            ("preprocessing", preprocessing),
            (
                "clf",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=1000,
                    random_state=42,
                ),
            ),
        ]
    )
