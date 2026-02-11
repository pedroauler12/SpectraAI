# -*- coding: utf-8 -*-
"""
Funcoes de visualizacao para EDA e avaliacao.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix


@dataclass
class PlotSpec:
    """Descricao de grafico obrigatorio para a entrega."""

    nome: str
    descricao: str
    quando_aplicavel: str


def list_required_plots() -> List[PlotSpec]:
    """
    Lista os graficos obrigatorios solicitados na tarefa.
    """
    return [
        PlotSpec(
            nome="Histograma",
            descricao="Distribuicao de variaveis numericas.",
            quando_aplicavel="Sempre que houver features numericas.",
        ),
        PlotSpec(
            nome="Boxplot",
            descricao="Dispersao e outliers por variavel.",
            quando_aplicavel="Sempre que houver features numericas.",
        ),
        PlotSpec(
            nome="Balanceamento por classe",
            descricao="Contagem ou proporcao por classe.",
            quando_aplicavel="Sempre que houver rotulos/classes.",
        ),
        PlotSpec(
            nome="Exemplos visuais",
            descricao="Amostras de imagens/tiles ou registros representativos.",
            quando_aplicavel="Quando houver dados visuais.",
        ),
        PlotSpec(
            nome="Mapa de calor/espacial",
            descricao="Densidade/variavel em coordenadas ou grade.",
            quando_aplicavel="Quando houver coordenadas/geometrias.",
        ),
        PlotSpec(
            nome="Matriz de confusao",
            descricao="Desempenho por classe.",
            quando_aplicavel="Quando houver y_true/y_pred.",
        ),
        PlotSpec(
            nome="Matriz de confusao espacial",
            descricao="Erros mapeados no espaco.",
            quando_aplicavel="Quando houver coordenadas ou grade espacial.",
        ),
    ]


def _resolve_numeric_columns(df: pd.DataFrame, cols: Optional[Sequence[str]] = None) -> List[str]:
    if cols is not None:
        return list(cols)
    return [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]


def plot_histograms(
    df: pd.DataFrame,
    cols: Optional[Sequence[str]] = None,
    bins: int = 30,
    kde: bool = False,
    max_cols: int = 3,
    figsize: Tuple[int, int] = (12, 8),
) -> plt.Figure:
    """
    Plota histogramas para colunas numericas.
    """
    cols = _resolve_numeric_columns(df, cols)
    if not cols:
        raise ValueError("Nenhuma coluna numerica encontrada para histograma.")

    n = len(cols)
    ncols = min(max_cols, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)
    axes = axes.flatten()

    for ax, col in zip(axes, cols):
        sns.histplot(df[col].dropna(), bins=bins, kde=kde, ax=ax)
        ax.set_title(f"Histograma: {col}")
    for ax in axes[n:]:
        ax.axis("off")

    fig.tight_layout()
    return fig


def plot_boxplots(
    df: pd.DataFrame,
    cols: Optional[Sequence[str]] = None,
    max_cols: int = 3,
    figsize: Tuple[int, int] = (12, 8),
) -> plt.Figure:
    """
    Plota boxplots para colunas numericas.
    """
    cols = _resolve_numeric_columns(df, cols)
    if not cols:
        raise ValueError("Nenhuma coluna numerica encontrada para boxplot.")

    n = len(cols)
    ncols = min(max_cols, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)
    axes = axes.flatten()

    for ax, col in zip(axes, cols):
        sns.boxplot(x=df[col].dropna(), ax=ax)
        ax.set_title(f"Boxplot: {col}")
    for ax in axes[n:]:
        ax.axis("off")

    fig.tight_layout()
    return fig


def plot_class_balance(
    y: Sequence,
    class_names: Optional[Sequence[str]] = None,
    normalize: bool = False,
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """
    Plota balanceamento por classe (contagem ou proporcao).
    """
    series = pd.Series(y)
    counts = series.value_counts(normalize=normalize).sort_index()
    labels = class_names if class_names is not None else counts.index.astype(str)

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 4))

    ax.bar(labels, counts.values)
    ax.set_title("Balanceamento por classe")
    ax.set_ylabel("Proporcao" if normalize else "Contagem")
    ax.set_xlabel("Classe")
    return ax


def plot_confusion_matrix(
    y_true: Sequence,
    y_pred: Sequence,
    labels: Optional[Sequence[str]] = None,
    normalize: Optional[str] = None,
    ax: Optional[plt.Axes] = None,
    title: str = "Matriz de confusao",
) -> plt.Axes:
    """
    Plota matriz de confusao com opcao de normalizacao.

    normalize: None, 'true', 'pred', 'all'
    """
    cm = confusion_matrix(y_true, y_pred, labels=labels, normalize=normalize)

    if ax is None:
        _, ax = plt.subplots(figsize=(6, 5))

    sns.heatmap(cm, annot=True, fmt=".2f" if normalize else "d", cmap="Blues", ax=ax)
    ax.set_title(title)
    ax.set_ylabel("Verdadeiro")
    ax.set_xlabel("Predito")
    return ax


def plot_spatial_confusion(
    y_true: Sequence,
    y_pred: Sequence,
    coords: Optional[np.ndarray] = None,
    grid_shape: Optional[Tuple[int, int]] = None,
    ax: Optional[plt.Axes] = None,
    title: str = "Matriz de confusao espacial",
) -> plt.Axes:
    """
    Plota erros espaciais como mapa (correto vs incorreto).

    Use coords (N,2) com [x,y] ou grid_shape para reshape em grade.
    """
    correct = np.array(y_true) == np.array(y_pred)

    if ax is None:
        _, ax = plt.subplots(figsize=(6, 5))

    if coords is not None:
        coords = np.asarray(coords)
        if coords.ndim != 2 or coords.shape[1] != 2:
            raise ValueError("coords deve ter shape (N, 2) com [x, y].")
        ax.scatter(coords[:, 0], coords[:, 1], c=correct, cmap="coolwarm", s=12)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
    elif grid_shape is not None:
        grid = correct.reshape(grid_shape)
        ax.imshow(grid, cmap="coolwarm", interpolation="nearest")
    else:
        raise ValueError("Informe coords ou grid_shape para o mapa espacial.")

    ax.set_title(title)
    return ax


def plot_examples(
    images: Sequence,
    labels: Optional[Sequence] = None,
    preds: Optional[Sequence] = None,
    n: int = 12,
    ncols: int = 4,
    figsize: Tuple[int, int] = (10, 8),
    title: str = "Exemplos visuais",
) -> plt.Figure:
    """
    Plota exemplos de imagens/tiles.
    """
    images = list(images)
    n = min(n, len(images))
    nrows = int(np.ceil(n / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)
    axes = axes.flatten()

    for i in range(n):
        ax = axes[i]
        ax.imshow(images[i])
        ax.axis("off")
        if labels is not None:
            label = labels[i]
            pred = preds[i] if preds is not None else None
            if pred is None:
                ax.set_title(f"y: {label}")
            else:
                ax.set_title(f"y: {label} | y_hat: {pred}")

    for ax in axes[n:]:
        ax.axis("off")

    fig.suptitle(title)
    fig.tight_layout()
    return fig


def plot_spatial_heatmap(
    coords: np.ndarray,
    values: Optional[Sequence] = None,
    bins: int = 50,
    ax: Optional[plt.Axes] = None,
    title: str = "Mapa de calor espacial",
) -> plt.Axes:
    """
    Mapa de calor espacial usando hexbin.
    """
    coords = np.asarray(coords)
    if coords.ndim != 2 or coords.shape[1] != 2:
        raise ValueError("coords deve ter shape (N, 2) com [x, y].")

    if ax is None:
        _, ax = plt.subplots(figsize=(6, 5))

    if values is None:
        hb = ax.hexbin(coords[:, 0], coords[:, 1], gridsize=bins, cmap="magma")
    else:
        hb = ax.hexbin(coords[:, 0], coords[:, 1], C=values, gridsize=bins, cmap="magma")
    ax.set_title(title)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    plt.colorbar(hb, ax=ax)
    return ax


def criar_grid_visualizacao(
    image_data_list: Sequence[np.ndarray],
    main_title: Optional[str] = None,
    subplot_titles: Optional[Sequence[str]] = None,
    figsize_mult: float = 5.0,
    cmap: str = "gray",
    interpolation: str = "nearest",
) -> plt.Figure:
    """
    Cria e retorna um grid de visualizacao de imagens usando matplotlib.

    Funciona com listas de arrays 2D ou 3D (para visualizar bandas individuais).
    O layout eh ajustado automaticamente para um grid aproximadamente quadrado.

    Parameters
    ----------
    image_data_list : list of np.ndarray
        Uma lista de arrays NumPy 2D (altura, largura), onde cada array eh uma imagem ou banda.
    main_title : str, optional
        Titulo principal da figura (suptitle). Default eh None.
    subplot_titles : list of str, optional
        Lista de titulos para cada subplot. Se nao fornecida, sera usado "Imagem {i+1}".
    figsize_mult : float, optional
        Multiplicador para o tamanho da figura. Default eh 5.0 (figsize = (num_cols*5, num_rows*5)).
    cmap : str, optional
        Colormap para exibicao. Default eh 'gray'.
    interpolation : str, optional
        Tipo de interpolacao na exibicao. Default eh 'nearest'.

    Returns
    -------
    fig : plt.Figure
        A figura matplotlib criada.
    """
    if not image_data_list:
        print("A lista de imagens esta vazia. Nada para exibir.")
        return None

    num_subplots = len(image_data_list)

    # Calcula numero de linhas e colunas para um layout aproximadamente quadrado
    num_cols = int(np.ceil(np.sqrt(num_subplots)))
    num_rows = int(np.ceil(num_subplots / num_cols))

    # Cria figura e subplots
    figsize = (num_cols * figsize_mult, num_rows * figsize_mult)
    fig, axs = plt.subplots(num_rows, num_cols, figsize=figsize, squeeze=False)

    # Plota cada imagem
    for i, img_array in enumerate(image_data_list):
        row_idx = i // num_cols
        col_idx = i % num_cols
        ax = axs[row_idx, col_idx]

        ax.imshow(img_array, cmap=cmap, interpolation=interpolation)

        # Define titulo do subplot
        if subplot_titles and i < len(subplot_titles):
            ax.set_title(subplot_titles[i])
        else:
            ax.set_title(f"Imagem {i + 1}")

        ax.axis("off")

    # Oculta subplots vazios
    for i in range(num_subplots, num_rows * num_cols):
        row_idx = i // num_cols
        col_idx = i % num_cols
        axs[row_idx, col_idx].set_visible(False)

    # Adiciona titulo principal se fornecido
    if main_title:
        fig.suptitle(main_title, fontsize=16, fontweight="bold")
        rect = [0, 0.03, 1, 0.95]
    else:
        rect = [0, 0, 1, 1]

    fig.tight_layout(rect=rect)
    return fig


<<<<<<< Updated upstream
=======
def adicionar_marcador_mineral(
    ax: plt.Axes,
    marker_x: float,
    marker_y: float,
    mineral_type: Optional[str] = None,
    color: str = "red",
    markersize: int = 10,
    markeredgewidth: int = 2,
) -> None:
    """
    Adiciona um marcador circular ao redor de um ponto específico em um subplot,
    sem cobrir o ponto exato.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        O objeto Axes (subplot) onde o marcador será plotado.
    marker_x : float
        Coordenada X (coluna) do marcador dentro do subplot.
    marker_y : float
        Coordenada Y (linha) do marcador dentro do subplot.
    mineral_type : str, optional
        Tipo de mineral associado ao ponto, para possível uso futuro (não usado diretamente no plot).
    color : str, optional
        Cor do marcador (padrão: 'red').
    markersize : int, optional
        Tamanho do marcador (padrão: 10).
    markeredgewidth : int, optional
        Largura da borda do marcador (padrão: 2).

    Returns
    -------
    None
    """
    # 'o' para círculo, 'fillstyle='none'' para círculo vazado
    ax.plot(
        marker_x,
        marker_y,
        marker="o",
        fillstyle="none",
        color=color,
        markersize=markersize,
        markeredgewidth=markeredgewidth,
    )


>>>>>>> Stashed changes
def analysis_questions() -> List[str]:
    """
    Perguntas de analise e vieses para documentar no notebook.
    """
    return [
        "As distribuicoes das variaveis numericas indicam assimetria ou outliers extremos?",
        "Existe correlacao forte entre features que pode indicar redundancia?",
        "O balanceamento por classe e adequado para o modelo proposto?",
        "Ha classes com representacao espacial muito concentrada?",
        "Erros espaciais estao agrupados em certas regioes?",
        "Ha diferencas de desempenho por classe (precisao/recall) que indiquem vies?",
        "Alguma classe minoritaria esta sendo consistentemente sub-predita?",
        "Os exemplos visuais indicam padroes de rotulagem inconsistentes?",
        "Ha sinais de vies geografico (ex.: mais erros em certas areas)?",
        "Que fatores externos podem estar influenciando as variaveis (sazonalidade, iluminacao, sensor)?",
    ]
