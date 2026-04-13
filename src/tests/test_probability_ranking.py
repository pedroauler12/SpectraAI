from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest


def _write_ranking_fixture_files(project_root: Path) -> None:
    data_dir = project_root / "data"
    outputs_dir = project_root / "outputs" / "a09_interpretabilidade_visualizacao"
    (project_root / "src").mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)
    outputs_dir.mkdir(parents=True, exist_ok=True)

    pixels_df = pd.DataFrame(
        [
            {
                "path": "/tmp/ASTER_IMG/A1/chip_128x128_multiband.tif",
                "count": 1,
                "height": 2,
                "width": 2,
                "pixel_0": 0.2,
                "pixel_1": 0.2,
                "pixel_2": 0.2,
                "pixel_3": 0.2,
            },
            {
                "path": "/tmp/ASTER_IMG/B2/chip_128x128_multiband.tif",
                "count": 1,
                "height": 2,
                "width": 2,
                "pixel_0": 0.8,
                "pixel_1": 0.8,
                "pixel_2": 0.8,
                "pixel_3": 0.8,
            },
            {
                "path": "/tmp/ASTER_IMG/A1/chip_128x128_multiband.tif",
                "count": 1,
                "height": 2,
                "width": 2,
                "pixel_0": 0.8,
                "pixel_1": 0.2,
                "pixel_2": 0.8,
                "pixel_3": 0.2,
            },
            {
                "path": "/tmp/ASTER_IMG/C3/chip_128x128_multiband.tif",
                "count": 1,
                "height": 2,
                "width": 2,
                "pixel_0": 0.6,
                "pixel_1": 0.6,
                "pixel_2": 0.6,
                "pixel_3": 0.6,
            },
        ]
    )
    pixels_df.to_csv(data_dir / "pixels_dataset.csv", index=False)

    metadata_df = pd.DataFrame(
        [
            {
                "numero_amostra": "A1",
                "latitude_wgs84_decimal": -18.1,
                "longitude_wgs84_decimal": -46.1,
                "classe_balanceamento": "NEGATIVO",
                "litologia_padronizada": "gnaisse",
            },
            {
                "numero_amostra": "B2",
                "latitude_wgs84_decimal": -18.2,
                "longitude_wgs84_decimal": -46.2,
                "classe_balanceamento": "POSITIVO",
                "litologia_padronizada": "carbonatito",
            },
            {
                "numero_amostra": "C3",
                "latitude_wgs84_decimal": -18.3,
                "longitude_wgs84_decimal": -46.3,
                "classe_balanceamento": "POSITIVO",
                "litologia_padronizada": "laterita",
            },
        ]
    )
    metadata_df.to_excel(
        data_dir / "banco.xlsx",
        sheet_name="Banco de Dados Positivo-Negativ",
        index=False,
    )

    pd.DataFrame(
        [
            {"threshold_name": "threshold_f1", "threshold": 0.65},
            {"threshold_name": "threshold_0.5", "threshold": 0.5},
        ]
    ).to_csv(outputs_dir / "test_metrics_comparison.csv", index=False)


class MeanProbabilityModel:
    def predict(self, batch, batch_size=32, verbose=0):
        np = pytest.importorskip("numpy")
        return np.mean(batch, axis=(1, 2, 3), keepdims=True)


class ZeroProbabilityModel:
    def predict(self, batch, batch_size=32, verbose=0):
        np = pytest.importorskip("numpy")
        return np.zeros((batch.shape[0], 1), dtype=float)


def _make_bundle(project_root: Path, model) -> SimpleNamespace:
    return SimpleNamespace(
        project_root=project_root,
        model=model,
        model_name="Transfer Learning (A08)",
        normalizer=None,
        target_size=(2, 2),
        target_channels=1,
        normalization="none",
        class_names=("Negativo", "Positivo"),
        decision_threshold=0.65,
        decision_threshold_name="threshold_f1",
        dataset_csv=project_root / "data" / "pixels_dataset.csv",
    )


def test_build_dataset_probability_ranking_generates_unique_sorted_rows(tmp_path: Path, monkeypatch):
    pd = pytest.importorskip("pandas")
    from src.inference import transfer_geo_inference as geo

    _write_ranking_fixture_files(tmp_path)

    monkeypatch.setattr(geo, "adapt_cnn_input_tensor", lambda x, **kwargs: (x, kwargs.get("normalizer")))
    monkeypatch.setattr(
        geo,
        "_load_transfer_bundle_for_threshold",
        lambda project_root, threshold_mode, model_key=None: _make_bundle(project_root, MeanProbabilityModel()),
    )

    ranking_df = geo.build_dataset_probability_ranking(
        project_root=tmp_path,
        threshold_mode="threshold_f1",
        force_refresh=True,
        chunksize=2,
        batch_size=2,
    )

    assert isinstance(ranking_df, pd.DataFrame)
    assert list(ranking_df["numero_amostra"]) == ["B2", "C3", "A1"]
    assert list(ranking_df["rank"]) == [1, 2, 3]
    assert ranking_df.loc[ranking_df["numero_amostra"] == "A1", "prob_pos"].iloc[0] == pytest.approx(0.35)
    assert ranking_df.loc[ranking_df["numero_amostra"] == "B2", "pred_label_threshold"].iloc[0] == "Positivo"
    assert ranking_df.loc[ranking_df["numero_amostra"] == "A1", "pred_label_threshold"].iloc[0] == "Negativo"
    assert ranking_df.loc[ranking_df["numero_amostra"] == "C3", "litologia_padronizada"].iloc[0] == "laterita"
    assert (tmp_path / "outputs" / "a09_interpretabilidade_visualizacao" / geo.RANKING_CACHE_FILENAME).exists()


def test_build_dataset_probability_ranking_uses_cache_and_force_refresh(tmp_path: Path, monkeypatch):
    from src.inference import transfer_geo_inference as geo

    _write_ranking_fixture_files(tmp_path)
    monkeypatch.setattr(geo, "adapt_cnn_input_tensor", lambda x, **kwargs: (x, kwargs.get("normalizer")))
    monkeypatch.setattr(
        geo,
        "_load_transfer_bundle_for_threshold",
        lambda project_root, threshold_mode, model_key=None: _make_bundle(project_root, MeanProbabilityModel()),
    )

    first_df = geo.build_dataset_probability_ranking(
        project_root=tmp_path,
        threshold_mode="threshold_f1",
        force_refresh=True,
    )

    def fail_if_called(*args, **kwargs):
        raise AssertionError("O modelo nao deveria ser recarregado quando o cache existe.")

    monkeypatch.setattr(geo, "_load_transfer_bundle_for_threshold", fail_if_called)
    cached_df = geo.build_dataset_probability_ranking(
        project_root=tmp_path,
        threshold_mode="threshold_f1",
        force_refresh=False,
    )
    assert list(cached_df["prob_pos"]) == list(first_df["prob_pos"])

    monkeypatch.setattr(
        geo,
        "_load_transfer_bundle_for_threshold",
        lambda project_root, threshold_mode, model_key=None: _make_bundle(project_root, ZeroProbabilityModel()),
    )
    refreshed_df = geo.build_dataset_probability_ranking(
        project_root=tmp_path,
        threshold_mode="threshold_f1",
        force_refresh=True,
    )
    assert refreshed_df["prob_pos"].max() == pytest.approx(0.0)


def test_build_dataset_probability_ranking_changes_threshold_without_reordering(tmp_path: Path, monkeypatch):
    from src.inference import transfer_geo_inference as geo

    _write_ranking_fixture_files(tmp_path)
    monkeypatch.setattr(geo, "adapt_cnn_input_tensor", lambda x, **kwargs: (x, kwargs.get("normalizer")))
    monkeypatch.setattr(
        geo,
        "_load_transfer_bundle_for_threshold",
        lambda project_root, threshold_mode, model_key=None: _make_bundle(project_root, MeanProbabilityModel()),
    )

    threshold_f1_df = geo.build_dataset_probability_ranking(
        project_root=tmp_path,
        threshold_mode="threshold_f1",
        force_refresh=True,
    )

    monkeypatch.setattr(
        geo,
        "_load_transfer_bundle_for_threshold",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("Nao deveria recomputar para trocar threshold.")),
    )
    threshold_05_df = geo.build_dataset_probability_ranking(
        project_root=tmp_path,
        threshold_mode="threshold_0.5",
        force_refresh=False,
    )

    assert list(threshold_f1_df["numero_amostra"]) == list(threshold_05_df["numero_amostra"])
    assert threshold_f1_df["decision_threshold"].iloc[0] == pytest.approx(0.65)
    assert threshold_05_df["decision_threshold"].iloc[0] == pytest.approx(0.5)
    assert threshold_f1_df.loc[threshold_f1_df["numero_amostra"] == "C3", "pred_label_threshold"].iloc[0] == "Negativo"
    assert threshold_05_df.loc[threshold_05_df["numero_amostra"] == "C3", "pred_label_threshold"].iloc[0] == "Positivo"


def test_probability_ranking_page_helpers_smoke(tmp_path: Path):
    pytest.importorskip("streamlit")
    folium = pytest.importorskip("folium")

    page_path = Path(__file__).resolve().parents[1] / "apps" / "pages" / "a10_ranking_probabilidades.py"
    spec = importlib.util.spec_from_file_location("probability_ranking_page", page_path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)

    ranking_df = pd.DataFrame(
        [
            {
                "rank": 1,
                "numero_amostra": "B2",
                "prob_pos": 0.8,
                "prob_neg": 0.2,
                "pred_label_threshold": "Positivo",
                "decision_threshold": 0.65,
                "decision_threshold_name": "threshold_f1",
                "latitude_wgs84_decimal": -18.2,
                "longitude_wgs84_decimal": -46.2,
                "classe_balanceamento": "POSITIVO",
                "litologia_padronizada": "carbonatito",
                "path": "/tmp/ASTER_IMG/B2/chip_128x128_multiband.tif",
            },
            {
                "rank": 2,
                "numero_amostra": "C3",
                "prob_pos": 0.6,
                "prob_neg": 0.4,
                "pred_label_threshold": "Negativo",
                "decision_threshold": 0.65,
                "decision_threshold_name": "threshold_f1",
                "latitude_wgs84_decimal": -18.3,
                "longitude_wgs84_decimal": -46.3,
                "classe_balanceamento": "POSITIVO",
                "litologia_padronizada": "laterita",
                "path": "/tmp/ASTER_IMG/C3/chip_128x128_multiband.tif",
            },
        ]
    )

    filtered_df = module.filter_ranking_frame(
        ranking_df,
        sample_query="B",
        top_n=10,
        min_probability=0.5,
    )
    fmap = module.make_probability_ranking_map(filtered_df)

    assert list(filtered_df["numero_amostra"]) == ["B2"]
    assert isinstance(fmap, folium.Map)


def test_enrich_ranking_frame_adds_operational_columns(tmp_path: Path):
    pytest.importorskip("streamlit")

    page_path = Path(__file__).resolve().parents[1] / "apps" / "pages" / "a11_painel_de_decisao.py"
    spec = importlib.util.spec_from_file_location("operational_page", page_path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)

    df = pd.DataFrame(
        [
            {
                "rank": 1,
                "numero_amostra": "B2",
                "prob_pos": 0.84,
                "pred_label_threshold": "Positivo",
                "decision_threshold": 0.65,
                "decision_threshold_name": "threshold_f1",
                "classe_balanceamento": "POSITIVO",
                "litologia_padronizada": "carbonatito",
                "latitude_wgs84_decimal": -18.2,
                "longitude_wgs84_decimal": -46.2,
            },
            {
                "rank": 2,
                "numero_amostra": "A1",
                "prob_pos": 0.31,
                "pred_label_threshold": "Negativo",
                "decision_threshold": 0.65,
                "decision_threshold_name": "threshold_f1",
                "classe_balanceamento": "NEGATIVO",
                "litologia_padronizada": "gnaisse",
                "latitude_wgs84_decimal": -18.1,
                "longitude_wgs84_decimal": -46.1,
            },
        ]
    )

    enriched = module.enrich_ranking_frame(df)

    assert list(enriched["tier"]) == ["Muito Alto", "Baixo"]
    assert list(enriched["prioridade_operacional"]) == ["Campo imediato", "Baixa prioridade"]
    assert "recomendacao" in enriched.columns
