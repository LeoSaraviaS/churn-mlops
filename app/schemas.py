"""Contratos de entrada/salida de la API (Pydantic).

Estas clases son la razón por la que la API nunca responde 500 ante un
input inválido: FastAPI valida el body contra `ChurnPredictionRequest`
antes de que el código de negocio se ejecute, y devuelve 422 con el detalle
del campo que falló.

Los rangos (`ge`/`le`) y los sets cerrados (`Geography`, `Gender`) reflejan
los valores reales observados en `Churn_Modelling.csv`; se documentan como
tal para que el equipo pueda justificarlos en la defensa oral.
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class ChurnPredictionRequest(BaseModel):
    CreditScore: int = Field(..., ge=300, le=900, description="Puntaje crediticio")
    Geography: Literal["France", "Spain", "Germany"]
    Gender: Literal["Male", "Female"]
    Age: int = Field(..., ge=18, le=100)
    Tenure: int = Field(..., ge=0, le=10, description="Años como cliente")
    Balance: float = Field(..., ge=0)
    NumOfProducts: int = Field(..., ge=1, le=4)
    HasCrCard: Literal[0, 1]
    IsActiveMember: Literal[0, 1]
    EstimatedSalary: float = Field(..., ge=0)

    model_config = {
        "json_schema_extra": {
            "example": {
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
        }
    }


class ChurnPredictionResponse(BaseModel):
    churn_prediction: Literal[0, 1] = Field(
        ..., description="1 = el cliente probablemente abandona, 0 = se queda"
    )
    churn_probability: float = Field(..., ge=0, le=1)
    risk_level: Literal["low", "medium", "high"]
    model_version: str


class BatchPredictionRequest(BaseModel):
    """Lista de clientes para /predict/batch.

    min_length=1 evita un batch vacio (422 en vez de una respuesta sin
    sentido); el tope maximo se valida en el endpoint contra
    settings.max_batch_size, no aqui, para no acoplar el contrato a la
    configuracion de runtime.
    """

    customers: list[ChurnPredictionRequest] = Field(..., min_length=1)


class BatchPredictionResponse(BaseModel):
    predictions: list[ChurnPredictionResponse]
    n_customers: int
    n_predicted_churn: int = Field(..., description="Cuantos de la lista quedaron en clase 1")
    mean_churn_probability: float = Field(..., ge=0, le=1)


class HealthResponse(BaseModel):
    status: Literal["ok"]
    model_loaded: bool
    model_version: str
