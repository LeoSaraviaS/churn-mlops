"""Punto de entrada de la API FastAPI.

Endpoints:
    GET  /health          -> liveness/readiness probe (usado también por
                              HEALTHCHECK de Docker y por el smoke test del
                              pipeline de CI/CD).
    GET  /                -> info básica del servicio.
    POST /predict         -> predicción de churn para un cliente.
    POST /predict/batch   -> predicción para una lista de clientes.
    GET  /metrics         -> métricas en formato Prometheus (agregado por
                              app/metrics.py vía prometheus-fastapi-instrumentator).

El modelo se carga una sola vez al arrancar el proceso (lifespan), no en
cada request, para no pagar el costo de deserializar el .joblib en cada
predicción.
"""
from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException

from app.config import get_settings
from app.metrics import churn_prediction_errors_total, churn_predictions_total, setup_metrics
from app.predictor import ModelNotLoadedError, get_predictor, init_predictor
from app.schemas import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    ChurnPredictionRequest,
    ChurnPredictionResponse,
    HealthResponse,
)

settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_predictor(settings.model_path, settings.metadata_path)
    yield


app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="API de predicción de churn (regresión logística) — servicio MLOps end-to-end.",
    lifespan=lifespan,
)

setup_metrics(app)


@app.get("/health", response_model=HealthResponse, tags=["ops"])
def health() -> HealthResponse:
    try:
        predictor = get_predictor()
        loaded = predictor.is_loaded
        version = predictor.version
    except ModelNotLoadedError:
        loaded = False
        version = "unknown"
    return HealthResponse(status="ok", model_loaded=loaded, model_version=version)


@app.get("/", tags=["ops"])
def root() -> dict:
    return {
        "service": settings.app_name,
        "version": settings.app_version,
        "docs": "/docs",
        "health": "/health",
        "metrics": "/metrics",
        "model_schema": "/model/schema",
    }


@app.get("/model/schema", tags=["ops"])
def model_schema() -> dict:
    try:
        version = get_predictor().version
    except ModelNotLoadedError:
        version = "unknown"

    return {
        "model_version": version,
        "input_schema": ChurnPredictionRequest.model_json_schema(),
        "output_schema": ChurnPredictionResponse.model_json_schema(),
    }



@app.post("/predict", response_model=ChurnPredictionResponse, tags=["inference"])
def predict(payload: ChurnPredictionRequest) -> ChurnPredictionResponse:
    try:
        predictor = get_predictor()
        prediction, probability = predictor.predict(payload)
    except ModelNotLoadedError as exc:
        churn_prediction_errors_total.inc()
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001 - se traduce a 500 explícito, nunca silencioso
        churn_prediction_errors_total.inc()
        raise HTTPException(status_code=500, detail=f"Error interno al predecir: {exc}") from exc

    churn_predictions_total.labels(prediction=str(prediction)).inc()

    return ChurnPredictionResponse(
        churn_prediction=prediction,
        churn_probability=round(probability, 4),
        risk_level=get_predictor().risk_level(probability),
        model_version=get_predictor().version,
    )


@app.post("/predict/batch", response_model=BatchPredictionResponse, tags=["inference"])
def predict_batch(payload: BatchPredictionRequest) -> BatchPredictionResponse:
    n = len(payload.customers)
    if n > settings.max_batch_size:
        raise HTTPException(
            status_code=413,
            detail=(
                f"El batch trae {n} clientes y el máximo es {settings.max_batch_size}. "
                "Divide la petición en lotes más pequeños."
            ),
        )

    try:
        predictor = get_predictor()
        results = predictor.predict_batch(payload.customers)
    except ModelNotLoadedError as exc:
        churn_prediction_errors_total.inc()
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001 - se traduce a 500 explícito, nunca silencioso
        churn_prediction_errors_total.inc()
        raise HTTPException(status_code=500, detail=f"Error interno al predecir: {exc}") from exc

    predictions = [
        ChurnPredictionResponse(
            churn_prediction=prediction,
            churn_probability=round(probability, 4),
            risk_level=predictor.risk_level(probability),
            model_version=predictor.version,
        )
        for prediction, probability in results
    ]
    for prediction, _probability in results:
        churn_predictions_total.labels(prediction=str(prediction)).inc()

    return BatchPredictionResponse(
        predictions=predictions,
        n_customers=n,
        n_predicted_churn=sum(p.churn_prediction for p in predictions),
        mean_churn_probability=round(sum(p.churn_probability for p in predictions) / n, 4),
    )
