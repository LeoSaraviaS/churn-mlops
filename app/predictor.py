"""Carga del modelo y lógica de predicción.

El modelo se carga UNA VEZ (singleton, cargado en el lifespan de FastAPI en
`app/main.py`) y se reutiliza en cada request. Cargar el .joblib en cada
predicción sería correcto pero añadiría latencia innecesaria a cada request;
por eso `Predictor` se instancia una sola vez al arrancar el proceso.
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd

from app.schemas import ChurnPredictionRequest


class ModelNotLoadedError(RuntimeError):
    """El modelo aún no fue cargado (fallo de arranque)."""


class Predictor:
    def __init__(self, model_path: str, metadata_path: str):
        self._model_path = Path(model_path)
        self._metadata_path = Path(metadata_path)
        self._model = None
        self._metadata: dict = {}

    def load(self) -> None:
        if not self._model_path.exists():
            raise FileNotFoundError(
                f"No se encontró el modelo en {self._model_path}. "
                "Ejecuta 'python -m training.train' primero."
            )
        self._model = joblib.load(self._model_path)
        if self._metadata_path.exists():
            self._metadata = json.loads(self._metadata_path.read_text())

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    @property
    def version(self) -> str:
        return self._metadata.get("trained_at_utc", "unknown")

    def predict(self, request: ChurnPredictionRequest) -> tuple[int, float]:
        if self._model is None:
            raise ModelNotLoadedError("El modelo no está cargado.")

        # Una sola fila -> DataFrame de una fila. El pipeline (ColumnTransformer
        # + OneHotEncoder ya fitteado) se encarga del resto de forma
        # consistente con el entrenamiento, sin necesidad de recrear dummies
        # manualmente.
        row = pd.DataFrame([request.model_dump()])
        proba = float(self._model.predict_proba(row)[0, 1])
        prediction = int(proba >= 0.5)
        return prediction, proba

    def predict_batch(self, requests: list[ChurnPredictionRequest]) -> list[tuple[int, float]]:
        """Predice sobre N clientes en una sola llamada vectorizada.

        Construir un DataFrame de N filas y llamar a predict_proba una vez
        es lo que hace que un batch de 200 clientes no tarde 200 veces lo que
        tarda una prediccion sola: el costo fijo de la llamada al modelo se
        paga una unica vez, no por fila.
        """
        if self._model is None:
            raise ModelNotLoadedError("El modelo no está cargado.")

        frame = pd.DataFrame([r.model_dump() for r in requests])
        probabilities = self._model.predict_proba(frame)[:, 1]
        return [(int(proba >= 0.5), float(proba)) for proba in probabilities]

    @staticmethod
    def risk_level(probability: float) -> str:
        if probability < 0.33:
            return "low"
        if probability < 0.66:
            return "medium"
        return "high"


_predictor: Predictor | None = None


def get_predictor() -> Predictor:
    global _predictor
    if _predictor is None:
        raise ModelNotLoadedError(
            "El predictor no ha sido inicializado. Revisa el lifespan de la app."
        )
    return _predictor


def init_predictor(model_path: str, metadata_path: str) -> Predictor:
    global _predictor
    _predictor = Predictor(model_path, metadata_path)
    _predictor.load()
    return _predictor
