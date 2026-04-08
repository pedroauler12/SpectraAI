"""
Avaliação: métricas de classificação, curvas ROC/PR, matrizes de confusão
e geração automática de visualizações padronizadas.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.utils.metrics import (
    classification_metrics_extended,
    select_threshold_by_f1,
)
from src.analise_visual.graficos import (
    plot_class_balance,
    plot_confusion_matrix,
)

logger = logging.getLogger(__name__)

CLASS_NAMES = ("Negativo", "Positivo")


# ── métricas ───────────────────────────────────────────────────────────────
def evaluate_model(
    model,
    test_ds,
    y_test: np.ndarray,
    *,
    threshold: float | None = None,
) -> dict[str, Any]:
    """
    Avalia o modelo no conjunto de teste.

    1. Calcula probabilidades de classe positiva.
    2. Seleciona threshold ótimo por F1 (se não informado).
    3. Calcula métricas estendidas.

    Returns
    -------
    dict
        ``metrics``, ``y_prob``, ``y_pred``, ``threshold``.
    """
    logger.info("Avaliando modelo no conjunto de teste …")

    # Coletar probabilidades
    y_prob_batches = []
    for batch in test_ds:
        xb = batch[0] if isinstance(batch, tuple) else batch
        prob = model.predict(xb, verbose=0)
        y_prob_batches.append(prob)

    y_prob = np.concatenate(y_prob_batches, axis=0).squeeze()

    # Threshold
    if threshold is None:
        threshold = select_threshold_by_f1(y_test, y_prob)
        logger.info("Threshold ótimo (F1): %.4f", threshold)

    y_pred = (y_prob >= threshold).astype(int)

    metrics = classification_metrics_extended(y_test, y_pred, y_prob)
    metrics["threshold"] = float(threshold)

    logger.info("═" * 50)
    logger.info("RESULTADOS NO TESTE")
    logger.info("═" * 50)
    for k, v in metrics.items():
        if v is not None:
            logger.info("  %-20s %.4f", k, v)
    logger.info("═" * 50)

    return {
        "metrics": metrics,
        "y_prob": y_prob,
        "y_pred": y_pred,
        "y_test": y_test,
        "threshold": threshold,
    }


# ── visualizações ──────────────────────────────────────────────────────────
def _save_fig(fig: plt.Figure, path: Path, dpi: int = 150) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(path), dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    logger.info("  Salvo: %s", path)


def plot_training_curves(
    history_p1: dict,
    history_p2: dict,
    save_dir: Path,
) -> None:
    """Gera gráficos de loss e accuracy por época (ambas fases)."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Loss
    ax = axes[0]
    epochs_p1 = range(1, len(history_p1["loss"]) + 1)
    epochs_p2 = range(
        len(epochs_p1) + 1,
        len(epochs_p1) + len(history_p2["loss"]) + 1,
    )
    ax.plot(epochs_p1, history_p1["loss"], label="Train (P1)")
    ax.plot(epochs_p1, history_p1["val_loss"], label="Val (P1)")
    ax.plot(epochs_p2, history_p2["loss"], label="Train (P2)")
    ax.plot(epochs_p2, history_p2["val_loss"], label="Val (P2)")
    ax.axvline(len(epochs_p1) + 0.5, color="gray", linestyle="--", alpha=0.5, label="Fine-tune start")
    ax.set_title("Loss por época")
    ax.set_xlabel("Época")
    ax.set_ylabel("Loss")
    ax.legend()

    # Accuracy
    ax = axes[1]
    ax.plot(epochs_p1, history_p1["accuracy"], label="Train (P1)")
    ax.plot(epochs_p1, history_p1["val_accuracy"], label="Val (P1)")
    ax.plot(epochs_p2, history_p2["accuracy"], label="Train (P2)")
    ax.plot(epochs_p2, history_p2["val_accuracy"], label="Val (P2)")
    ax.axvline(len(epochs_p1) + 0.5, color="gray", linestyle="--", alpha=0.5, label="Fine-tune start")
    ax.set_title("Accuracy por época")
    ax.set_xlabel("Época")
    ax.set_ylabel("Accuracy")
    ax.legend()

    fig.suptitle("Curvas de treinamento (Fase 1 + Fase 2)")
    fig.tight_layout()
    _save_fig(fig, save_dir / "training_curves.png")


def plot_roc_pr_curves(
    y_test: np.ndarray,
    y_prob: np.ndarray,
    save_dir: Path,
) -> None:
    """Gera curvas ROC e Precision-Recall."""
    from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # ROC
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    axes[0].plot(fpr, tpr, label=f"ROC (AUC = {roc_auc:.3f})")
    axes[0].plot([0, 1], [0, 1], "k--", alpha=0.4)
    axes[0].set_title("Curva ROC")
    axes[0].set_xlabel("FPR")
    axes[0].set_ylabel("TPR")
    axes[0].legend()

    # PR
    precision, recall, _ = precision_recall_curve(y_test, y_prob)
    ap = average_precision_score(y_test, y_prob)
    axes[1].plot(recall, precision, label=f"PR (AP = {ap:.3f})")
    axes[1].set_title("Curva Precision-Recall")
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].legend()

    fig.suptitle("Curvas de avaliação")
    fig.tight_layout()
    _save_fig(fig, save_dir / "roc_pr_curves.png")


def plot_confusion_matrices(
    y_test: np.ndarray,
    y_pred: np.ndarray,
    save_dir: Path,
) -> None:
    """Gera matrizes de confusão (absoluta e normalizada)."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    plot_confusion_matrix(
        y_test, y_pred, normalize=None, ax=axes[0],
        title="Matriz de confusão (absoluta)",
    )
    plot_confusion_matrix(
        y_test, y_pred, normalize="true", ax=axes[1],
        title="Matriz de confusão (normalizada)",
    )

    fig.suptitle("Matrizes de confusão — conjunto de teste")
    fig.tight_layout()
    _save_fig(fig, save_dir / "confusion_matrices.png")


def plot_class_distribution(
    y_train: np.ndarray,
    y_val: np.ndarray,
    y_test: np.ndarray,
    save_dir: Path,
) -> None:
    """Gera gráfico de balanceamento de classes por split."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, y, title in zip(axes, [y_train, y_val, y_test], ["Treino", "Validação", "Teste"]):
        plot_class_balance(y, class_names=CLASS_NAMES, ax=ax)
        ax.set_title(f"Classes — {title}")
    fig.tight_layout()
    _save_fig(fig, save_dir / "class_distribution.png")


# ── geração automática de resultados ──────────────────────────────────────
def run_evaluation(
    cfg: dict,
    model,
    tf_data: dict[str, Any],
    splits: dict[str, Any],
    history_phase1: dict,
    history_phase2: dict,
    *,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    """
    Executa avaliação completa e salva todos os artefatos.

    1. Avalia no conjunto de teste (métricas + threshold ótimo).
    2. Gera visualizações: curvas de treinamento, ROC/PR, matrizes de confusão.
    3. Salva métricas em JSON e predições em CSV.

    Returns
    -------
    dict
        ``eval_result``, ``metrics_path``, ``predictions_path``, ``viz_dir``.
    """
    if output_dir is None:
        artifact_root = Path(cfg.get("_artifact_root", "."))
        output_dir = artifact_root
    output_dir = Path(output_dir)

    metrics_dir = output_dir / cfg["paths"]["outputs_metrics"]
    viz_dir = output_dir / cfg["paths"]["outputs_viz"]
    preds_dir = output_dir / cfg["paths"]["outputs_preds"]
    for d in (metrics_dir, viz_dir, preds_dir):
        d.mkdir(parents=True, exist_ok=True)

    eval_cfg = cfg.get("evaluation", {})
    threshold_default = eval_cfg.get("threshold_default", 0.5)

    # 1 — Métricas ──────────────────────────────────────────────────────────
    eval_result = evaluate_model(
        model,
        tf_data["test_ds"],
        splits["y_test"],
        threshold=None,  # threshold otimizado automaticamente
    )

    # 2 — Salvar métricas em JSON ───────────────────────────────────────────
    metrics_path = metrics_dir / "test_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(eval_result["metrics"], f, indent=2, default=str)
    logger.info("Métricas salvas em: %s", metrics_path)

    # 3 — Salvar predições em CSV ───────────────────────────────────────────
    preds_df = pd.DataFrame({
        "y_true": eval_result["y_test"],
        "y_pred": eval_result["y_pred"],
        "y_prob": eval_result["y_prob"],
    })
    preds_path = preds_dir / "test_predictions.csv"
    preds_df.to_csv(preds_path, index=False)
    logger.info("Predições salvas em: %s", preds_path)

    # 4 — Visualizações ─────────────────────────────────────────────────────
    logger.info("Gerando visualizações …")
    plot_training_curves(history_phase1, history_phase2, viz_dir)
    plot_roc_pr_curves(eval_result["y_test"], eval_result["y_prob"], viz_dir)
    plot_confusion_matrices(eval_result["y_test"], eval_result["y_pred"], viz_dir)
    plot_class_distribution(splits["y_train"], splits["y_val"], splits["y_test"], viz_dir)

    return {
        "eval_result": eval_result,
        "metrics_path": str(metrics_path),
        "predictions_path": str(preds_path),
        "viz_dir": str(viz_dir),
    }
