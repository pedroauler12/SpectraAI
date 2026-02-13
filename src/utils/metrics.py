from sklearn.metrics import (
    accuracy_score,
    f1_score,
    recall_score,
    precision_score,
    roc_auc_score,
    average_precision_score,
    balanced_accuracy_score,
    precision_recall_curve,
    mean_absolute_error,
    mean_squared_error,
    r2_score
)
import numpy as np


def classification_metrics(y_true, y_pred, y_prob=None):
    results = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }

    if y_prob is not None:
        try:
            results["roc_auc"] = roc_auc_score(y_true, y_prob)
        except:
            results["roc_auc"] = None

    return results


def classification_metrics_extended(y_true, y_pred, y_prob=None):
    """
    Métricas de classificação com suporte a baseline binário mais completo.

    Inclui accuracy, precision, recall, f1 e balanced_accuracy.
    Se y_prob for informado, inclui roc_auc e pr_auc.
    """
    results = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
    }

    if y_prob is not None:
        try:
            results["roc_auc"] = roc_auc_score(y_true, y_prob)
        except Exception:
            results["roc_auc"] = None
        try:
            results["pr_auc"] = average_precision_score(y_true, y_prob)
        except Exception:
            results["pr_auc"] = None

    return results


def positive_scores_from_model(model, X):
    """
    Extrai score positivo [0, 1] de modelos com predict_proba ou decision_function.
    """
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]

    if hasattr(model, "decision_function"):
        scores = model.decision_function(X)
        min_s, max_s = scores.min(), scores.max()
        if max_s - min_s < 1e-12:
            return np.full_like(scores, 0.5, dtype=float)
        return (scores - min_s) / (max_s - min_s)

    raise ValueError("Modelo nao possui predict_proba nem decision_function")


def select_threshold_by_f1(y_true, y_score, default_threshold=0.5):
    """
    Seleciona threshold que maximiza F1 a partir da curva precision-recall.
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_score)
    f1_vals = 2 * precision * recall / (precision + recall + 1e-12)

    # thresholds possui tamanho len(precision) - 1
    if len(thresholds) == 0:
        return default_threshold

    best_idx = int(np.nanargmax(f1_vals[:-1]))
    return float(thresholds[best_idx])


def regression_metrics(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))

    return {
        "mae": mean_absolute_error(y_true, y_pred),
        "rmse": rmse,
        "r2": r2_score(y_true, y_pred)
    }
