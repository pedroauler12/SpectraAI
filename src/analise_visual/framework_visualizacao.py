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
