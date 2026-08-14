"""Métricas de evaluación del modelo.

Se reportan varias métricas -no sólo accuracy- porque el dataset tiene
desbalance de clases (~20% churn / ~80% no-churn): un modelo que siempre
prediga "no churn" tendría ~80% de accuracy sin ser útil. Precision, recall,
F1 y ROC-AUC (por clase 1 = churn) muestran el desempeño real del modelo
sobre la clase minoritaria, que es la que interesa desde el punto de vista
de negocio (detectar clientes en riesgo de fuga).
"""
from __future__ import annotations

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def compute_metrics(y_true, y_pred, y_proba) -> dict:
    """Calcula el set de métricas usado tanto para el reporte del README
    como para el gate de calidad en el pipeline de CI/CD.
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "precision": round(float(precision_score(y_true, y_pred)), 4),
        "recall": round(float(recall_score(y_true, y_pred)), 4),
        "f1_score": round(float(f1_score(y_true, y_pred)), 4),
        "roc_auc": round(float(roc_auc_score(y_true, y_proba)), 4),
        "confusion_matrix": {
            "true_negative": int(tn),
            "false_positive": int(fp),
            "false_negative": int(fn),
            "true_positive": int(tp),
        },
    }


def summarize_for_gate(metrics: dict, roc_auc_threshold: float) -> tuple[bool, str]:
    """Determina si el modelo pasa el gate de calidad del pipeline CI/CD.

    El job `train-and-gate` de GitHub Actions llama a esta función: si
    retorna False, el pipeline se detiene ANTES de construir la imagen
    Docker y desplegar, evitando publicar un modelo peor de lo esperado.
    """
    auc = metrics["roc_auc"]
    passed = auc >= roc_auc_threshold
    msg = (
        f"ROC-AUC={auc:.4f} vs umbral={roc_auc_threshold:.4f} -> "
        f"{'PASA' if passed else 'NO PASA'}"
    )
    return passed, msg
