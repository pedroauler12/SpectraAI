from __future__ import annotations

import pandas as pd
import pytest


def test_build_promising_region_clusters_groups_nearby_high_score_samples():
    from src.inference.explainability_analysis import build_promising_region_clusters

    ranking_df = pd.DataFrame(
        [
            {
                "rank": 1,
                "image_id": "A1",
                "y_score": 0.82,
                "latitude_wgs84_decimal": -18.4262,
                "longitude_wgs84_decimal": -46.6592,
                "litologia_padronizada": "carbonatito",
            },
            {
                "rank": 2,
                "image_id": "A2",
                "y_score": 0.79,
                "latitude_wgs84_decimal": -18.4264,
                "longitude_wgs84_decimal": -46.6591,
                "litologia_padronizada": "carbonatito",
            },
            {
                "rank": 3,
                "image_id": "B1",
                "y_score": 0.74,
                "latitude_wgs84_decimal": -18.4300,
                "longitude_wgs84_decimal": -46.6500,
                "litologia_padronizada": "laterita",
            },
            {
                "rank": 4,
                "image_id": "C1",
                "y_score": 0.40,
                "latitude_wgs84_decimal": -18.5000,
                "longitude_wgs84_decimal": -46.7000,
                "litologia_padronizada": "gnaisse",
            },
        ]
    )

    cluster_df = build_promising_region_clusters(
        ranking_df,
        min_score=0.70,
        radius_km=0.10,
        min_samples=2,
    )

    assert len(cluster_df) == 1
    assert cluster_df.loc[0, "n_amostras"] == 2
    assert cluster_df.loc[0, "litologia_dominante"] == "carbonatito"
    assert "A1" in cluster_df.loc[0, "top_amostras"]


def test_build_sample_comparison_table_returns_selected_rows_in_rank_order():
    from src.inference.explainability_analysis import build_sample_comparison_table

    df = pd.DataFrame(
        [
            {"rank": 3, "image_id": "C1", "y_score": 0.61, "tier": "Alto"},
            {"rank": 1, "image_id": "A1", "y_score": 0.82, "tier": "Muito Alto"},
            {"rank": 2, "image_id": "B1", "y_score": 0.75, "tier": "Muito Alto"},
        ]
    )

    comparison_df = build_sample_comparison_table(df, ["B1", "A1"])

    assert list(comparison_df["image_id"]) == ["A1", "B1"]
    assert list(comparison_df["rank"]) == [1, 2]


def test_describe_region_opportunity_returns_business_language():
    from src.inference.explainability_analysis import describe_region_opportunity

    description = describe_region_opportunity(
        pd.Series(
            {
                "n_amostras": 4,
                "score_medio": 0.78,
                "score_maximo": 0.84,
                "litologia_dominante": "carbonatito",
                "top_amostras": "A1, A2, B1",
            }
        )
    )

    assert "regiao" in description.lower()
    assert "4 amostras" in description
    assert "carbonatito" in description


def test_build_campaign_suggestion_prioritizes_clusters_and_returns_editable_columns():
    from src.inference.explainability_analysis import build_campaign_suggestion

    ranking_df = pd.DataFrame(
        [
            {"rank": 1, "image_id": "A1", "y_score": 0.84, "tier": "Muito Alto", "latitude_wgs84_decimal": -18.4, "longitude_wgs84_decimal": -46.6},
            {"rank": 2, "image_id": "A2", "y_score": 0.80, "tier": "Muito Alto", "latitude_wgs84_decimal": -18.4, "longitude_wgs84_decimal": -46.6},
            {"rank": 3, "image_id": "B1", "y_score": 0.76, "tier": "Alto", "latitude_wgs84_decimal": -18.5, "longitude_wgs84_decimal": -46.7},
            {"rank": 4, "image_id": "C1", "y_score": 0.65, "tier": "Alto", "latitude_wgs84_decimal": -18.6, "longitude_wgs84_decimal": -46.8},
        ]
    )
    cluster_df = pd.DataFrame(
        [
            {
                "region_label": "Regiao Promissora 1",
                "score_maximo": 0.84,
                "n_amostras": 2,
                "amostras": ["A1", "A2"],
            }
        ]
    )

    suggestion_df = build_campaign_suggestion(
        ranking_df,
        cluster_df,
        max_targets=3,
        min_score=0.65,
    )

    assert len(suggestion_df) == 3
    assert "region_label" in suggestion_df.columns
    assert "prioridade" in suggestion_df.columns
    assert "status" in suggestion_df.columns
    assert "notas" in suggestion_df.columns
    assert "A1" in suggestion_df["image_id"].tolist()
