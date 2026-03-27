from pathlib import Path

import pytest


def test_plot_probability_distributions_smoke(tmp_path: Path):
    pytest.importorskip("pandas")
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")

    from src.analise_visual.framework_visualizacao import (
        plot_probability_distributions,
    )

    save_path = tmp_path / "prob_dist.png"
    fig = plot_probability_distributions(
        y_true=[0, 0, 1, 1, 1],
        y_score=[0.1, 0.3, 0.65, 0.8, 0.9],
        thresholds={"0.5": 0.5, "F1": 0.62},
        save_path=save_path,
        show=False,
    )

    assert fig is not None
    assert save_path.exists()
