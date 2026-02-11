import pytest
import numpy as np
import pandas as pd


def test_graficos_basic():
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    pytest.importorskip("seaborn")

    from src.analise_visual.graficos import (
        plot_histograms,
        plot_boxplots,
        plot_class_balance,
        plot_confusion_matrix,
        plot_spatial_confusion,
        plot_examples,
        plot_spatial_heatmap,
        list_required_plots,
        analysis_questions,
    )

    df = pd.DataFrame({"a": [1, 2, 3, 4], "b": [4, 3, 2, 1]})

    fig1 = plot_histograms(df)
    fig2 = plot_boxplots(df)

    ax1 = plot_class_balance([0, 1, 1, 0])
    ax2 = plot_confusion_matrix([0, 1, 0, 1], [0, 1, 1, 0])

    coords = np.array([[0, 0], [1, 1], [2, 2], [3, 3]])
    ax3 = plot_spatial_confusion([0, 1, 0, 1], [0, 0, 0, 1], coords=coords)

    images = [np.zeros((10, 10)) for _ in range(4)]
    fig3 = plot_examples(images, labels=[0, 1, 0, 1])

    ax4 = plot_spatial_heatmap(coords)

    assert fig1 is not None
    assert fig2 is not None
    assert fig3 is not None
    assert ax1 is not None
    assert ax2 is not None
    assert ax3 is not None
    assert ax4 is not None

    assert len(list_required_plots()) > 0
    assert len(analysis_questions()) > 0
