"""Script de entrenamiento end-to-end.

Uso:
    python -m training.train
    python -m training.train --data data/Churn_Modelling.csv --auc-threshold 0.75

Este mismo script lo ejecuta el job `train-and-gate` del pipeline de CI/CD:
entrena con datos reales, calcula métricas reales sobre el set de test, y
falla (`sys.exit(1)`) si el modelo no supera el umbral de calidad mínimo.
Si el modelo pasa el gate, se guardan `models/model.joblib` y
`models/metadata.json`, que luego consume la etapa `docker-build`.
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
from datetime import UTC, datetime
from pathlib import Path

import joblib
import sklearn

from training.data import class_balance, get_dataset, load_raw_dataset
from training.evaluate import compute_metrics, summarize_for_gate
from training.model import build_model

DEFAULT_DATA_PATH = Path("data/Churn_Modelling.csv")
DEFAULT_MODEL_DIR = Path("models")
DEFAULT_AUC_THRESHOLD = 0.75
# Semilla fija para reproducibilidad: mismo valor usado en el split
# (training/data.py) y en LogisticRegression (training/model.py).
DEFAULT_RANDOM_STATE = 42


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Entrena el modelo de churn.")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument(
        "--auc-threshold",
        type=float,
        default=DEFAULT_AUC_THRESHOLD,
        help="ROC-AUC mínimo para que el modelo pase el gate de CI/CD.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=DEFAULT_RANDOM_STATE,
        help="Semilla para el split train/test, para resultados reproducibles.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.model_dir.mkdir(parents=True, exist_ok=True)

    print(f"[train] Cargando datos desde {args.data}")
    raw = load_raw_dataset(args.data)
    balance = class_balance(raw)
    print(f"[train] Balance de clases real: {balance}")

    X_train, X_test, y_train, y_test = get_dataset(
        args.data, random_state=args.random_state
    )
    print(f"[train] Train: {len(X_train)} filas | Test: {len(X_test)} filas")

    model = build_model()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    metrics = compute_metrics(y_test, y_pred, y_proba)
    print(f"[train] Métricas sobre test set: {json.dumps(metrics, indent=2)}")

    passed, gate_msg = summarize_for_gate(metrics, args.auc_threshold)
    print(f"[train] Gate de calidad: {gate_msg}")

    model_path = args.model_dir / "model.joblib"
    metadata_path = args.model_dir / "metadata.json"

    joblib.dump(model, model_path)

    metadata = {
        "model_type": "LogisticRegression",
        "trained_at_utc": datetime.now(UTC).isoformat(),
        "sklearn_version": sklearn.__version__,
        "python_version": platform.python_version(),
        "train_rows": len(X_train),
        "test_rows": len(X_test),
        "class_balance": balance,
        "metrics_test_set": metrics,
        "auc_threshold": args.auc_threshold,
        "random_state": args.random_state,
        "gate_passed": passed,
        "features": {
            "categorical": ["Geography", "Gender", "AgeGroup (derivado de Age)"],
            "numeric": [
                "CreditScore",
                "Age",
                "Tenure",
                "Balance",
                "NumOfProducts",
                "HasCrCard",
                "IsActiveMember",
                "EstimatedSalary",
            ],
            "excluded_identifiers": ["RowNumber", "CustomerId", "Surname"],
        },
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False))
    print(f"[train] Modelo guardado en {model_path}")
    print(f"[train] Metadata guardada en {metadata_path}")

    if not passed:
        print(
            f"[train] ERROR: el modelo no superó el umbral de ROC-AUC "
            f"({metrics['roc_auc']} < {args.auc_threshold}). "
            f"El pipeline de CI/CD se detiene aquí.",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
