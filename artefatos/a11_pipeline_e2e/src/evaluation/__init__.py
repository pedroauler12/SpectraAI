from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    RocCurveDisplay,
    confusion_matrix,
)

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def build_summary(
    *,
    config: dict[str, Any],
    split_meta: dict[str, Any],
    result: dict[str, Any],
    model_path: Path,
    history_path: Path | None,
    predictions_path: Path | None,
) -> dict[str, Any]:
    return {
        "config_used": str(config["_config_path"]),
        "timestamp": result["timestamp"],
        "seed": int(config["seed"]),
        "model_name": config["model"]["base_config_name"],
        "evaluation_split": result.get("evaluation_split", "test"),
        "threshold": float(config["evaluation"]["threshold_default"]),
        "n_total": int(split_meta["n_total"]),
        "n_valid": int(split_meta["n_valid"]),
        "n_train": int(split_meta["n_train"]),
        "n_val": int(split_meta["n_val"]),
        "n_test": int(split_meta["n_test"]),
        "test_accuracy": result.get("test_accuracy"),
        "test_precision": result.get("test_precision"),
        "test_recall": result.get("test_recall"),
        "test_f1": result.get("test_f1"),
        "test_balanced_accuracy": result.get("test_balanced_accuracy"),
        "test_roc_auc": result.get("test_roc_auc"),
        "test_pr_auc": result.get("test_pr_auc"),
        "training_time_seconds": result.get("training_time_seconds"),
        "total_epochs": result.get("total_epochs"),
        "head_epochs": result.get("head_epochs"),
        "ft_epochs": result.get("ft_epochs"),
        "model_path": str(model_path),
        "history_path": str(history_path) if history_path is not None else None,
        "predictions_path": str(predictions_path) if predictions_path is not None else None,
        "experiment_dir": result.get("experiment_dir"),
    }


def save_summary_files(
    *,
    summary: dict[str, Any],
    json_path: str | Path,
    csv_path: str | Path,
) -> None:
    json_path = Path(json_path)
    csv_path = Path(csv_path)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    with json_path.open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2, ensure_ascii=False)

    pd.DataFrame([summary]).to_csv(csv_path, index=False)


def save_visualizations(
    *,
    predictions_df: pd.DataFrame,
    output_dir: str | Path,
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _save_confusion_matrix(predictions_df, output_dir / "confusion_matrix.png")
    _save_roc_pr_curves(predictions_df, output_dir / "roc_pr_curves.png")


def _save_confusion_matrix(predictions_df: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(5, 4))
    cm = confusion_matrix(predictions_df["y_true"], predictions_df["y_pred"])
    ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Negativo", "Positivo"],
    ).plot(ax=ax, colorbar=False)
    ax.set_title("Matriz de Confusao - Teste")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _save_roc_pr_curves(predictions_df: pd.DataFrame, output_path: Path) -> None:
    y_true = predictions_df["y_true"].to_numpy()
    y_score = predictions_df["y_score"].to_numpy()

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    RocCurveDisplay.from_predictions(y_true, y_score, ax=axes[0])
    PrecisionRecallDisplay.from_predictions(y_true, y_score, ax=axes[1])
    axes[0].set_title("ROC - Teste")
    axes[1].set_title("Precision-Recall - Teste")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_training_curves(
    *,
    history_path: "Path | None",
    output_dir: "str | Path",
) -> None:
    """Gera e salva o grafico de curvas de aprendizado (loss e accuracy por epoca)."""
    if history_path is None or not Path(history_path).exists():
        return
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _save_training_curves_plot(Path(history_path), output_dir / "training_curves.png")


def _save_training_curves_plot(history_path: Path, output_path: Path) -> None:
    with history_path.open("r", encoding="utf-8") as f:
        history = json.load(f)

    # Detecta fronteira head/fine-tune pela primeira queda significativa de learning rate
    lr = history.get("learning_rate", [])
    head_end = 0
    for i in range(1, len(lr)):
        if lr[i] < lr[i - 1] * 0.5:
            head_end = i
            break

    epochs = list(range(1, len(history["loss"]) + 1))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(epochs, history["loss"], label="treino")
    axes[0].plot(epochs, history["val_loss"], label="validacao")
    if head_end:
        axes[0].axvline(x=head_end + 0.5, color="gray", linestyle="--", alpha=0.6, label="inicio fine-tune")
    axes[0].set_title("Loss por Epoca")
    axes[0].set_xlabel("Epoca")
    axes[0].set_ylabel("Loss")
    axes[0].legend()

    axes[1].plot(epochs, history["accuracy"], label="treino")
    axes[1].plot(epochs, history["val_accuracy"], label="validacao")
    if head_end:
        axes[1].axvline(x=head_end + 0.5, color="gray", linestyle="--", alpha=0.6, label="inicio fine-tune")
    axes[1].set_title("Acuracia por Epoca")
    axes[1].set_xlabel("Epoca")
    axes[1].set_ylabel("Acuracia")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
