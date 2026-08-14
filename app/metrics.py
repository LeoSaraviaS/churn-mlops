"""Métricas de servicio expuestas (bonus: latencia, conteo de requests, errores).

Se usa `prometheus-fastapi-instrumentator` porque es el estándar de facto
para instrumentar servicios FastAPI con métricas en formato Prometheus, y
Cloud Run / cualquier scraper de métricas puede consumir `/metrics`
directamente sin configuración adicional. Provee automáticamente:

    - http_requests_total{method,handler,status}      -> conteo de requests
    - http_request_duration_seconds{method,handler}    -> latencia (histograma)
    - http_requests_inprogress{method,handler}          -> requests en curso

Además se agrega un contador de negocio (`churn_predictions_total`) para
poder ver, sin mirar logs, cuántas predicciones se hicieron y con qué
resultado (0/1) — útil tanto para monitoreo técnico como para justificar el
bonus de métricas expuestas en la defensa oral.
"""
from __future__ import annotations

from prometheus_client import Counter
from prometheus_fastapi_instrumentator import Instrumentator

churn_predictions_total = Counter(
    "churn_predictions_total",
    "Cantidad de predicciones de churn realizadas, por resultado.",
    labelnames=["prediction"],
)

churn_prediction_errors_total = Counter(
    "churn_prediction_errors_total",
    "Cantidad de errores al intentar predecir (input inválido o fallo interno).",
)


def setup_metrics(app) -> Instrumentator:
    """Instrumenta la app FastAPI y expone el endpoint /metrics.

    Debe llamarse una sola vez, después de crear la instancia de FastAPI y
    antes de que el servidor empiece a recibir tráfico.
    """
    instrumentator = Instrumentator(
        should_group_status_codes=True,
        should_ignore_untemplated=True,
        excluded_handlers=["/metrics", "/health"],
    )
    instrumentator.instrument(app).expose(app, endpoint="/metrics", include_in_schema=False)
    return instrumentator
