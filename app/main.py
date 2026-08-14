"""Punto de entrada de la API FastAPI.

Endpoints:
    GET  /health   -> liveness/readiness probe (usado también por HEALTHCHECK
                       de Docker y por el smoke test del pipeline de CI/CD).
    GET  /         -> info básica del servicio.
    POST /predict  -> predicción de churn para un cliente.
    GET  /metrics  -> métricas en formato Prometheus (agregado por
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
from app.schemas import ChurnPredictionRequest, ChurnPredictionResponse, HealthResponse

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
