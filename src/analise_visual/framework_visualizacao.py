"""
Funções reutilizáveis de visualização para treinamento e avaliação de modelos CNN.

Este módulo fornece utilitários para:
- Plotar curvas de loss (treino e validação)
- Plotar curvas de acurácia
- Plotar matriz de confusão (normalizada ou absoluta)

As funções retornam o objeto matplotlib.figure.Figure,
permitindo customizações adicionais externas.

"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import auc, precision_recall_curve, roc_curve


def _prepare_output_path(save_path: str | Path | None) -> Path | None:
    if save_path is None:
        return None

    output_path = Path(save_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    return output_path


def _finalize_plot(fig: plt.Figure, save_path: str | Path | None, show: bool) -> plt.Figure:
    output_path = _prepare_output_path(save_path)
    if output_path is not None:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")

    if show:
        plt.show()

    return fig


def plot_loss_curve(
    train_loss: Sequence[float],
    val_loss: Sequence[float] | None = None,
    *,
    title: str = "Curva de perda",
    xlabel: str = "Época",
    ylabel: str = "Loss",
    save_path: str | Path | None = None,
    show: bool = False,
) -> plt.Figure:
    """Plota a curva de perda de treino (e validação, opcional)."""
    fig, ax = plt.subplots(figsize=(8, 5))

    epochs = np.arange(1, len(train_loss) + 1)
    ax.plot(epochs, train_loss, label="Treino", marker="o")

    if val_loss is not None:
        if len(val_loss) != len(train_loss):
            raise ValueError("val_loss deve ter o mesmo tamanho de train_loss.")
        ax.plot(epochs, val_loss, label="Validação", marker="o")

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend()

    return _finalize_plot(fig, save_path, show)


def plot_accuracy(
    train_accuracy: Sequence[float],
    val_accuracy: Sequence[float] | None = None,
    *,
    title: str = "Curva de acurácia",
    xlabel: str = "Época",
    ylabel: str = "Acurácia",
    save_path: str | Path | None = None,
    show: bool = False,
) -> plt.Figure:
    """Plota a curva de acurácia de treino (e validação, opcional)."""
    fig, ax = plt.subplots(figsize=(8, 5))

    epochs = np.arange(1, len(train_accuracy) + 1)
    ax.plot(epochs, train_accuracy, label="Treino", marker="o")

    if val_accuracy is not None:
        if len(val_accuracy) != len(train_accuracy):
            raise ValueError("val_accuracy deve ter o mesmo tamanho de train_accuracy.")
        ax.plot(epochs, val_accuracy, label="Validação", marker="o")

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_ylim(0, 1)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend()

    return _finalize_plot(fig, save_path, show)


def plot_confusion_matrix(
    confusion_matrix: Sequence[Sequence[int]] | np.ndarray,
    class_names: Iterable[str] | None = None,
    *,
    normalize: bool = False,
    cmap: str = "Blues",
    title: str = "Matriz de confusão",
    save_path: str | Path | None = None,
    show: bool = False,
) -> plt.Figure:
    """Plota uma matriz de confusão com anotações por célula."""
    matrix = np.array(confusion_matrix)

    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("confusion_matrix deve ser uma matriz quadrada 2D.")

    if normalize:
        row_sums = matrix.sum(axis=1, keepdims=True)
        matrix = np.divide(matrix, row_sums, where=row_sums != 0)

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(matrix, interpolation="nearest", cmap=cmap)
    fig.colorbar(im, ax=ax)

    size = matrix.shape[0]
    labels = list(class_names) if class_names is not None else [str(i) for i in range(size)]

    if len(labels) != size:
        raise ValueError("class_names deve ter o mesmo tamanho da dimensão da matriz.")

    ax.set_title(title)
    ax.set_xticks(np.arange(size))
    ax.set_yticks(np.arange(size))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    ax.set_ylabel("Classe real")
    ax.set_xlabel("Classe predita")

    threshold = matrix.max() / 2 if matrix.size else 0
    value_fmt = ".2f" if normalize else "d"

    for i in range(size):
        for j in range(size):
            value = matrix[i, j]
            ax.text(
                j,
                i,
                format(value, value_fmt),
                ha="center",
                va="center",
                color="white" if value > threshold else "black",
            )

    fig.tight_layout()
    return _finalize_plot(fig, save_path, show)


def plot_probability_distributions(
    y_true: Sequence[int] | np.ndarray,
    y_score: Sequence[float] | np.ndarray,
    thresholds: Mapping[str, float] | Sequence[float] | None = None,
    *,
    class_names: tuple[str, str] = ("Negativo", "Positivo"),
    title: str = "Distribuicao de probabilidades por classe",
    bins: int = 20,
    alpha: float = 0.55,
    save_path: str | Path | None = None,
    show: bool = False,
) -> plt.Figure:
    """
    Plota a distribuicao de ``P(classe positiva)`` separada por classe real.

    Parameters
    ----------
    y_true : Sequence[int] | np.ndarray
        Rotulos verdadeiros binarios.
    y_score : Sequence[float] | np.ndarray
        Probabilidades previstas para a classe positiva.
    thresholds : Mapping[str, float] | Sequence[float] | None, optional
        Thresholds a desenhar no grafico. Pode ser dict ``label -> valor`` ou
        uma sequencia simples de valores.
    """
    y_true_arr = np.asarray(y_true).reshape(-1)
    y_score_arr = np.asarray(y_score, dtype=float).reshape(-1)

    if y_true_arr.shape[0] != y_score_arr.shape[0]:
        raise ValueError("y_true e y_score devem ter o mesmo tamanho.")
    if y_true_arr.size == 0:
        raise ValueError("Informe ao menos uma amostra para plotar.")
    if len(class_names) != 2:
        raise ValueError("class_names deve conter exatamente dois nomes.")

    fig, ax = plt.subplots(figsize=(9, 5))

    mask_neg = y_true_arr == 0
    mask_pos = y_true_arr == 1

    ax.hist(
        y_score_arr[mask_neg],
        bins=bins,
        alpha=alpha,
        color="steelblue",
        label=f"{class_names[0]} (y=0)",
        density=True,
    )
    ax.hist(
        y_score_arr[mask_pos],
        bins=bins,
        alpha=alpha,
        color="darkorange",
        label=f"{class_names[1]} (y=1)",
        density=True,
    )

    if thresholds is not None:
        if isinstance(thresholds, Mapping):
            threshold_items = list(thresholds.items())
        else:
            threshold_items = [(f"thr={float(v):.3f}", float(v)) for v in thresholds]

        for label, value in threshold_items:
            ax.axvline(
                x=float(value),
                linestyle="--",
                linewidth=1.5,
                alpha=0.9,
                label=label,
            )

    ax.set_title(title)
    ax.set_xlabel("Probabilidade prevista da classe positiva")
    ax.set_ylabel("Densidade")
    ax.set_xlim(0, 1)
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend()

    fig.tight_layout()
    return _finalize_plot(fig, save_path, show)


def plot_probability_boxplot(
    y_true: Sequence[int] | np.ndarray,
    y_score: Sequence[float] | np.ndarray,
    *,
    class_names: tuple[str, str] = ("Negativo", "Positivo"),
    title: str = "Boxplot das probabilidades por classe real",
    ylabel: str = "Probabilidade prevista da classe positiva",
    save_path: str | Path | None = None,
    show: bool = False,
) -> plt.Figure:
    """
    Plota boxplots de ``P(classe positiva)`` separados pela classe real.
    """
    y_true_arr = np.asarray(y_true).reshape(-1)
    y_score_arr = np.asarray(y_score, dtype=float).reshape(-1)

    if y_true_arr.shape[0] != y_score_arr.shape[0]:
        raise ValueError("y_true e y_score devem ter o mesmo tamanho.")
    if len(class_names) != 2:
        raise ValueError("class_names deve conter exatamente dois nomes.")

    data = [
        y_score_arr[y_true_arr == 0],
        y_score_arr[y_true_arr == 1],
    ]

    fig, ax = plt.subplots(figsize=(8, 5))
    bp = ax.boxplot(
        data,
        patch_artist=True,
        labels=[f"{class_names[0]} (y=0)", f"{class_names[1]} (y=1)"],
        medianprops={"color": "black", "linewidth": 1.5},
    )
    colors = ["steelblue", "darkorange"]
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.65)

    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_ylim(0, 1)
    ax.grid(True, linestyle="--", alpha=0.3, axis="y")

    fig.tight_layout()
    return _finalize_plot(fig, save_path, show)


def plot_roc_pr_curves(
    y_true: Sequence[int] | np.ndarray,
    y_score: Sequence[float] | np.ndarray,
    *,
    title_prefix: str = "Curvas de desempenho probabilistico",
    save_path: str | Path | None = None,
    show: bool = False,
) -> plt.Figure:
    """
    Plota curvas ROC e Precision-Recall lado a lado.
    """
    y_true_arr = np.asarray(y_true).reshape(-1)
    y_score_arr = np.asarray(y_score, dtype=float).reshape(-1)

    if y_true_arr.shape[0] != y_score_arr.shape[0]:
        raise ValueError("y_true e y_score devem ter o mesmo tamanho.")

    fpr, tpr, _ = roc_curve(y_true_arr, y_score_arr)
    precision, recall, _ = precision_recall_curve(y_true_arr, y_score_arr)
    roc_auc_value = auc(fpr, tpr)
    pr_auc_value = auc(recall, precision)
    positive_rate = float(np.mean(y_true_arr == 1))

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].plot(fpr, tpr, color="tab:blue", linewidth=2, label=f"AUC = {roc_auc_value:.3f}")
    axes[0].plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1)
    axes[0].set_title(f"{title_prefix} - ROC")
    axes[0].set_xlabel("False Positive Rate")
    axes[0].set_ylabel("True Positive Rate")
    axes[0].grid(True, linestyle="--", alpha=0.3)
    axes[0].legend(loc="lower right")

    axes[1].plot(recall, precision, color="tab:orange", linewidth=2, label=f"AUC = {pr_auc_value:.3f}")
    axes[1].axhline(positive_rate, linestyle="--", color="gray", linewidth=1, label=f"Baseline = {positive_rate:.3f}")
    axes[1].set_title(f"{title_prefix} - Precision-Recall")
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].set_xlim(0, 1)
    axes[1].set_ylim(0, 1)
    axes[1].grid(True, linestyle="--", alpha=0.3)
    axes[1].legend(loc="lower left")

    fig.tight_layout()
    return _finalize_plot(fig, save_path, show)


def plot_threshold_sweep(
    y_true: Sequence[int] | np.ndarray,
    y_score: Sequence[float] | np.ndarray,
    *,
    thresholds: Sequence[float] | np.ndarray | None = None,
    title: str = "Variacao das metricas por threshold",
    save_path: str | Path | None = None,
    show: bool = False,
) -> plt.Figure:
    """
    Plota precision, recall, F1 e accuracy ao longo de uma grade de thresholds.
    """
    y_true_arr = np.asarray(y_true).reshape(-1).astype(int)
    y_score_arr = np.asarray(y_score, dtype=float).reshape(-1)

    if y_true_arr.shape[0] != y_score_arr.shape[0]:
        raise ValueError("y_true e y_score devem ter o mesmo tamanho.")

    if thresholds is None:
        thresholds_arr = np.linspace(0.0, 1.0, 101)
    else:
        thresholds_arr = np.asarray(thresholds, dtype=float).reshape(-1)

    precision_values = []
    recall_values = []
    f1_values = []
    accuracy_values = []

    for threshold in thresholds_arr:
        y_pred = (y_score_arr >= threshold).astype(int)
        tp = int(np.sum((y_true_arr == 1) & (y_pred == 1)))
        tn = int(np.sum((y_true_arr == 0) & (y_pred == 0)))
        fp = int(np.sum((y_true_arr == 0) & (y_pred == 1)))
        fn = int(np.sum((y_true_arr == 1) & (y_pred == 0)))

        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        accuracy = (tp + tn) / max(len(y_true_arr), 1)
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

        precision_values.append(precision)
        recall_values.append(recall)
        accuracy_values.append(accuracy)
        f1_values.append(f1)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(thresholds_arr, precision_values, label="Precision", linewidth=2)
    ax.plot(thresholds_arr, recall_values, label="Recall", linewidth=2)
    ax.plot(thresholds_arr, f1_values, label="F1", linewidth=2)
    ax.plot(thresholds_arr, accuracy_values, label="Accuracy", linewidth=2)
    ax.set_title(title)
    ax.set_xlabel("Threshold")
    ax.set_ylabel("Valor da metrica")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend(loc="best")

    fig.tight_layout()
    return _finalize_plot(fig, save_path, show)


def _normalize_preview_channel(channel: np.ndarray) -> np.ndarray:
    channel = np.asarray(channel, dtype=np.float32)
    finite_mask = np.isfinite(channel)
    if not np.any(finite_mask):
        return np.zeros_like(channel, dtype=np.float32)

    valid_values = channel[finite_mask]
    low, high = np.percentile(valid_values, [2, 98])
    if float(high) <= float(low):
        return np.zeros_like(channel, dtype=np.float32)

    normalized = (channel - low) / (high - low)
    return np.clip(normalized, 0.0, 1.0).astype(np.float32, copy=False)


def _chip_to_rgb_preview(
    chip: np.ndarray,
    *,
    rgb_band_indices: tuple[int, int, int] = (2, 1, 0),
) -> np.ndarray:
    chip_arr = np.asarray(chip, dtype=np.float32)
    if chip_arr.ndim != 3:
        raise ValueError(f"Cada chip deve ser 3D (H, W, C), recebido shape={chip_arr.shape}.")

    channels = []
    for idx in rgb_band_indices:
        if idx < 0 or idx >= chip_arr.shape[-1]:
            raise ValueError(f"Indice de banda invalido: {idx} para chip com {chip_arr.shape[-1]} canais.")
        channels.append(_normalize_preview_channel(chip_arr[..., idx]))
    return np.stack(channels, axis=-1)


def plot_marked_sample_chips(
    chips: Sequence[np.ndarray] | np.ndarray,
    *,
    sample_ids: Sequence[str] | None = None,
    labels: Sequence[int] | np.ndarray | None = None,
    class_names: tuple[str, str] = ("Negativo", "Positivo"),
    rgb_band_indices: tuple[int, int, int] = (2, 1, 0),
    point_xy: tuple[int, int] | None = None,
    ncols: int = 3,
    title: str = "Exemplos de chips com ponto central marcado",
    save_path: str | Path | None = None,
    show: bool = False,
) -> plt.Figure:
    """
    Plota uma grade de chips RGB com o ponto central marcado.
    """
    chips_arr = np.asarray(chips, dtype=np.float32)
    if chips_arr.ndim != 4:
        raise ValueError(f"Esperado tensor 4D (N, H, W, C), recebido shape={chips_arr.shape}.")

    n_samples = chips_arr.shape[0]
    if n_samples == 0:
        raise ValueError("Informe ao menos um chip para plotar.")
    if ncols <= 0:
        raise ValueError("ncols deve ser > 0.")

    sample_ids_list = list(sample_ids) if sample_ids is not None else [f"sample_{idx}" for idx in range(n_samples)]
    if len(sample_ids_list) != n_samples:
        raise ValueError("sample_ids deve ter o mesmo tamanho de chips.")

    labels_arr = None if labels is None else np.asarray(labels).reshape(-1)
    if labels_arr is not None and labels_arr.shape[0] != n_samples:
        raise ValueError("labels deve ter o mesmo tamanho de chips.")

    height, width = chips_arr.shape[1], chips_arr.shape[2]
    center_x = width // 2 if point_xy is None else int(point_xy[0])
    center_y = height // 2 if point_xy is None else int(point_xy[1])

    nrows = int(np.ceil(n_samples / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 4.2 * nrows))
    axes_arr = np.atleast_1d(axes).reshape(nrows, ncols)

    for idx, ax in enumerate(axes_arr.flat):
        if idx >= n_samples:
            ax.axis("off")
            continue

        preview = _chip_to_rgb_preview(chips_arr[idx], rgb_band_indices=rgb_band_indices)
        ax.imshow(preview)
        ax.scatter(center_x, center_y, c="red", marker="x", s=60, linewidths=2)
        ax.axvline(center_x, color="white", linestyle="--", linewidth=0.8, alpha=0.7)
        ax.axhline(center_y, color="white", linestyle="--", linewidth=0.8, alpha=0.7)

        title_lines = [sample_ids_list[idx]]
        if labels_arr is not None:
            label_value = int(labels_arr[idx])
            label_text = class_names[label_value] if label_value in (0, 1) else str(label_value)
            title_lines.append(f"Classe real: {label_text}")
        ax.set_title("\n".join(title_lines), fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])

    fig.suptitle(title, fontsize=14, y=1.02)
    fig.tight_layout()
    return _finalize_plot(fig, save_path, show)


"""

Exemplo de como usar:
from framework.visualizacao.py import plot_loss_curve

# Simulando histórico de treino
train_loss = [0.9, 0.7, 0.5, 0.4, 0.35]
val_loss = [1.0, 0.8, 0.6, 0.55, 0.50]

fig = plot_loss_curve(
    train_loss=train_loss,
    val_loss=val_loss,
    save_path="outputs/loss_curve.png",
    show=True
)"""
