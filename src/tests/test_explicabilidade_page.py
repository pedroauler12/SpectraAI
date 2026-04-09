from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd
import pytest


def _load_module():
    pytest.importorskip("streamlit")

    page_path = Path(__file__).resolve().parents[1] / "apps" / "pages" / "a12_explicabilidade_modelo.py"
    spec = importlib.util.spec_from_file_location("explainability_page", page_path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_prepare_explainability_target_table_merges_metadata_and_filters_positive():
    module = _load_module()

    ranking_df = pd.DataFrame(
        [
            {"image_id": "31712", "y_true": 1, "y_pred": 1, "y_score": 0.83, "rank": 1, "tier": "Muito Alto"},
            {"image_id": "23350", "y_true": 0, "y_pred": 0, "y_score": 0.04, "rank": 2, "tier": "Baixo"},
            {"image_id": "32471", "y_true": 1, "y_pred": 1, "y_score": 0.79, "rank": 3, "tier": "Alto"},
        ]
    )
    metadata_df = pd.DataFrame(
        [
            {
                "numero_amostra": "31712",
                "latitude_wgs84_decimal": -18.2,
                "longitude_wgs84_decimal": -46.3,
                "classe_balanceamento": "POSITIVO",
                "litologia_padronizada": "carbonatito",
            },
            {
                "numero_amostra": "32471",
                "latitude_wgs84_decimal": -18.4,
                "longitude_wgs84_decimal": -46.5,
                "classe_balanceamento": "POSITIVO",
                "litologia_padronizada": "laterita",
            },
        ]
    )

    result = module.prepare_explainability_target_table(
        ranking_df,
        metadata_df,
        top_n=10,
        min_score=0.50,
        only_positive=True,
    )

    assert list(result["image_id"]) == ["31712", "32471"]
    assert list(result["classe_prevista"]) == ["Positivo", "Positivo"]
    assert list(result["classe_real"]) == ["Positivo", "Positivo"]
    assert result.loc[result["image_id"] == "31712", "litologia_padronizada"].iloc[0] == "carbonatito"


def test_build_executive_takeaways_highlights_metrics_and_caution():
    module = _load_module()

    takeaways = module.build_executive_takeaways(
        {
            "threshold": 0.5,
            "test_f1": 0.851,
            "test_recall": 0.87,
            "test_precision": 0.83,
            "test_roc_auc": 0.93,
            "test_pr_auc": 0.87,
            "n_test": 59,
        }
    )

    assert len(takeaways) >= 4
    assert any("F1" in item and "ROC-AUC" in item for item in takeaways)
    assert any("59 amostras" in item for item in takeaways)


def test_summarize_cross_validation_returns_means():
    module = _load_module()

    cv_df = pd.DataFrame(
        [
            {"accuracy": 0.70, "f1": 0.56, "roc_auc": 0.82, "pr_auc": 0.65, "balanced_accuracy": 0.66},
            {"accuracy": 0.86, "f1": 0.84, "roc_auc": 0.90, "pr_auc": 0.76, "balanced_accuracy": 0.87},
            {"accuracy": 0.88, "f1": 0.85, "roc_auc": 0.89, "pr_auc": 0.88, "balanced_accuracy": 0.87},
        ]
    )

    summary = module.summarize_cross_validation(cv_df)

    assert summary["accuracy_mean"] == pytest.approx((0.70 + 0.86 + 0.88) / 3)
    assert summary["f1_mean"] == pytest.approx((0.56 + 0.84 + 0.85) / 3)
    assert "roc_auc_std" in summary


def test_build_nontechnical_sample_explanation_returns_plain_language_story():
    module = _load_module()

    row = pd.Series(
        {
            "image_id": "31712",
            "y_score": 0.83,
            "rank": 1,
            "tier": "Muito Alto",
            "classe_prevista": "Positivo",
            "classe_real": "Positivo",
            "litologia_padronizada": "carbonatito",
        }
    )

    story = module.build_nontechnical_sample_explanation(
        row,
        threshold=0.5,
        total_samples=12,
    )

    assert "31712" in story["headline"]
    assert "prioritaria" in story["business_meaning"]
    assert story["confidence_title"] == "Alta"
    assert "shortlist" in story["next_step"]
    assert "carbonatito" in story["geology_text"]


def test_summarize_gradcam_heatmap_describes_center_focus():
    module = _load_module()

    heatmap = [
        [0.0, 0.1, 0.0, 0.0],
        [0.1, 0.9, 0.8, 0.1],
        [0.0, 0.8, 1.0, 0.0],
        [0.0, 0.1, 0.0, 0.0],
    ]

    summary = module.summarize_gradcam_heatmap(heatmap)

    assert summary["title"] in {"Atencao focada", "Atencao moderadamente concentrada"}
    assert "atencao" in summary["summary"].lower()
