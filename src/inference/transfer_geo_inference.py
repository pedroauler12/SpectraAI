"""
Camada reutilizavel para inferencia geoespacial com modelos transfer learning.

Este modulo conecta:
- artefatos treinados no A08 e no A11;
- reconstrucao do normalizador de treino;
- pre-processamento de chips ASTER multibanda;
- busca e recorte de dados ASTER via EarthData.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence
import hashlib
import re
import shutil
import warnings

import numpy as np
import pandas as pd

try:
    from models.cnn_data_prep import dataframe_to_cnn_tensor, fit_channel_normalizer, prepare_grouped_cnn_splits
    from models.cnn_tf_data_pipeline import adapt_cnn_input_tensor
except ImportError:  # pragma: no cover - fallback para imports via pacote src.*
    from src.models.cnn_data_prep import dataframe_to_cnn_tensor, fit_channel_normalizer, prepare_grouped_cnn_splits
    from src.models.cnn_tf_data_pipeline import adapt_cnn_input_tensor


DEFAULT_CLASS_NAMES = ("Negativo", "Positivo")
DEFAULT_BAND_ORDER = ("B01", "B02", "B03N", "B04", "B05", "B06", "B07", "B08", "B09")
RANKING_CACHE_FILENAME = "full_dataset_probability_ranking.csv"
A11_RANKING_CACHE_FILENAME = "full_dataset_probability_ranking_a11_pipeline_e2e.csv"


@dataclass
class TransferInferenceBundle:
    """Objetos e parametros necessarios para inferencia consistente com o A08."""

    project_root: Path
    model: Any
    model_name: str
    normalizer: dict[str, Any]
    target_size: tuple[int, int]
    target_channels: int
    normalization: str
    class_names: tuple[str, str]
    decision_threshold: float
    decision_threshold_name: str
    seed: int
    dataset_csv: Path
    extracted_codes_json: Path
    model_path: Path


@dataclass
class PredictionResult:
    """Resultado padronizado de inferencia geoespacial."""

    lat: float
    lon: float
    prob_pos: float
    pred_class: int
    pred_label: str
    chip_path: str | None
    granule_id: str | None
    acquisition_time: str | None
    cloud_cover: float | None
    bbox_wgs84: tuple[float, float, float, float] | None
    chip_shape: tuple[int, int, int] | None = None
    preview_png_path: str | None = None
    preview_false_color_path: str | None = None
    preview_rgb_path: str | None = None
    quality_report: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def resolve_project_root(start_path: str | Path | None = None) -> Path:
    """Resolve a raiz do projeto a partir do cwd ou de um caminho informado."""
    if start_path is None:
        candidates = [Path.cwd().resolve()]
    else:
        candidates = [Path(start_path).resolve()]

    for candidate in candidates:
        if (candidate / "src").exists():
            return candidate
        for parent in [candidate, *candidate.parents]:
            if (parent / "src").exists():
                return parent

    raise FileNotFoundError("Nao foi possivel localizar a raiz do projeto.")


def _load_keras_model(model_path: Path):
    from tensorflow import keras  # type: ignore[import-untyped]

    return keras.models.load_model(model_path)


def _clip_prob(prob: float) -> float:
    return float(np.clip(prob, 0.0, 1.0))


def _sanitize_filename(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text)


def _predict_positive_probability(bundle: TransferInferenceBundle, batch: np.ndarray) -> float:
    raw_pred = bundle.model.predict(batch, verbose=0)
    prob_pos = np.asarray(raw_pred, dtype=float).reshape(-1)[0]
    return _clip_prob(float(prob_pos))


def overlay_activation_heatmap(
    image_rgb: np.ndarray,
    heatmap: np.ndarray,
    *,
    alpha: float = 0.45,
    colormap: str = "jet",
) -> np.ndarray:
    """
    Sobrepoe um heatmap normalizado sobre uma imagem RGB.
    """
    from matplotlib import colormaps

    base = np.asarray(image_rgb, dtype=np.float32)
    heat = np.asarray(heatmap, dtype=np.float32)

    if base.ndim != 3 or base.shape[-1] != 3:
        raise ValueError(f"Esperado image_rgb com shape (H, W, 3), recebido {base.shape}.")
    if heat.ndim != 2:
        raise ValueError(f"Esperado heatmap 2D, recebido {heat.shape}.")
    if base.shape[:2] != heat.shape:
        raise ValueError("image_rgb e heatmap devem ter a mesma dimensao espacial.")

    heat = np.clip(heat, 0.0, 1.0)
    cmap = colormaps.get_cmap(colormap)
    heat_rgb = cmap(heat)[..., :3].astype(np.float32)
    overlay = (1.0 - float(alpha)) * np.clip(base, 0.0, 1.0) + float(alpha) * heat_rgb
    return np.clip(overlay, 0.0, 1.0).astype(np.float32, copy=False)


def _normalize_preview_band(band: np.ndarray) -> np.ndarray:
    band = np.asarray(band, dtype=np.float32)
    finite_mask = np.isfinite(band)
    if not np.any(finite_mask):
        return np.zeros_like(band, dtype=np.float32)

    valid_values = band[finite_mask]
    low, high = np.percentile(valid_values, [2, 98])
    if float(high) <= float(low):
        return np.zeros_like(band, dtype=np.float32)

    norm = (band - low) / (high - low)
    return np.clip(norm, 0.0, 1.0).astype(np.float32, copy=False)


def build_false_color_preview(
    chip_array: np.ndarray,
    band_order: Sequence[str] = DEFAULT_BAND_ORDER,
    rgb_bands: tuple[str, str, str] = ("B06", "B05", "B02"),
) -> np.ndarray:
    """
    Gera composicao false-color mineralogica a partir do chip multibanda.
    """
    chip = np.asarray(chip_array, dtype=np.float32)
    if chip.ndim != 3:
        raise ValueError(f"Esperado chip 3D (H, W, C), recebido shape={chip.shape}.")
    if chip.shape[-1] != len(band_order):
        raise ValueError(
            f"Numero de bandas ({chip.shape[-1]}) difere do band_order ({len(band_order)})."
        )

    band_index = {band_name: idx for idx, band_name in enumerate(band_order)}
    channels = []
    for band_name in rgb_bands:
        if band_name not in band_index:
            raise ValueError(f"Banda '{band_name}' nao encontrada em band_order.")
        channels.append(_normalize_preview_band(chip[..., band_index[band_name]]))

    return np.stack(channels, axis=-1)


def build_rgb_preview(
    chip_array: np.ndarray,
    band_order: Sequence[str] = DEFAULT_BAND_ORDER,
    rgb_bands: tuple[str, str, str] = ("B03N", "B02", "B01"),
) -> np.ndarray:
    """
    Gera preview RGB mais proximo da visualizacao natural do ASTER.
    """
    return build_false_color_preview(
        chip_array,
        band_order=band_order,
        rgb_bands=rgb_bands,
    )


def save_preview_png(preview_rgb: np.ndarray, output_path: str | Path) -> Path:
    """
    Salva preview RGB em PNG para visualizacao no notebook/app.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(np.clip(preview_rgb, 0, 1))
    ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    return out_path


def _resolve_config_path(base_dir: Path, path_value: str | Path) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return (base_dir / path).resolve()


def _normalize_model_key(model_key: str) -> str:
    normalized = str(model_key).strip()
    if normalized not in {"a08_transfer_learning", "a11_pipeline_e2e"}:
        raise ValueError(f"Modelo nao suportado para inferencia/ranking: {model_key}")
    return normalized


def _build_transfer_inference_bundle(
    *,
    root: Path,
    model_file: Path,
    dataset_csv_path: Path,
    codes_path: Path,
    test_size: float,
    val_size: float,
    seed: int,
    target_size: tuple[int, int],
    target_channels: int,
    normalization: str,
    class_names: tuple[str, str],
    decision_threshold: float,
    decision_threshold_name: str,
    model_name: str,
) -> TransferInferenceBundle:
    if not dataset_csv_path.exists():
        raise FileNotFoundError(f"Arquivo nao encontrado: {dataset_csv_path}")
    if not codes_path.exists():
        raise FileNotFoundError(f"Arquivo nao encontrado: {codes_path}")
    if not model_file.exists():
        raise FileNotFoundError(f"Arquivo nao encontrado: {model_file}")

    df = pd.read_csv(dataset_csv_path)
    split_data = prepare_grouped_cnn_splits(
        df,
        extracted_codes_path=codes_path,
        data_format="channels_last",
        test_size=float(test_size),
        val_size=float(val_size),
        seed=int(seed),
    )

    x_train = np.asarray(split_data["X_train"], dtype=np.float32)
    normalizer = fit_channel_normalizer(
        x_train,
        method=normalization,
        data_format="channels_last",
    )

    model = _load_keras_model(model_file)
    resolved_target_channels = int(
        target_channels or split_data.get("shape_info", {}).get("n_channels", x_train.shape[-1])
    )
    return TransferInferenceBundle(
        project_root=root,
        model=model,
        model_name=model_name,
        normalizer=normalizer,
        target_size=tuple(int(v) for v in target_size),
        target_channels=resolved_target_channels,
        normalization=normalization,
        class_names=class_names,
        decision_threshold=float(decision_threshold),
        decision_threshold_name=str(decision_threshold_name),
        seed=int(seed),
        dataset_csv=dataset_csv_path,
        extracted_codes_json=codes_path,
        model_path=model_file,
    )


def _resolve_threshold_from_metrics(
    project_root: Path,
    *,
    threshold_name: str,
    fallback_threshold: float,
) -> tuple[float, str]:
    metrics_path = project_root / "outputs" / "a09_interpretabilidade_visualizacao" / "test_metrics_comparison.csv"
    if not metrics_path.exists():
        return float(fallback_threshold), "threshold_0.5"

    try:
        metrics_df = pd.read_csv(metrics_path)
    except Exception:
        return float(fallback_threshold), "threshold_0.5"

    if {"threshold_name", "threshold"} - set(metrics_df.columns):
        return float(fallback_threshold), "threshold_0.5"

    matched = metrics_df.loc[metrics_df["threshold_name"] == threshold_name, "threshold"]
    if matched.empty:
        return float(fallback_threshold), "threshold_0.5"

    try:
        resolved = float(matched.iloc[0])
    except Exception:
        return float(fallback_threshold), "threshold_0.5"
    return resolved, threshold_name


def _resolve_threshold_config(
    project_root: Path,
    *,
    threshold_mode: str,
    model_key: str,
) -> tuple[float, str]:
    normalized_model_key = _normalize_model_key(model_key)
    if threshold_mode == "threshold_0.5":
        return 0.5, "threshold_0.5"

    if normalized_model_key == "a11_pipeline_e2e":
        config_path = project_root / "artefatos" / "a11_pipeline_e2e" / "config.yaml"
        if not config_path.exists():
            return 0.5, "threshold_0.5"

        try:
            import yaml

            with config_path.open("r", encoding="utf-8") as file:
                config = yaml.safe_load(file)
            return float(config["evaluation"]["threshold_default"]), "threshold_default"
        except Exception:
            return 0.5, "threshold_0.5"
    return _resolve_threshold_from_metrics(
        project_root,
        threshold_name=threshold_mode,
        fallback_threshold=0.5,
    )


def _ranking_cache_path(project_root: Path, *, model_key: str) -> Path:
    normalized_model_key = _normalize_model_key(model_key)
    filename = (
        RANKING_CACHE_FILENAME
        if normalized_model_key == "a08_transfer_learning"
        else A11_RANKING_CACHE_FILENAME
    )
    return project_root / "outputs" / "a09_interpretabilidade_visualizacao" / filename


def _extract_sample_id_from_path(path_value: str) -> str:
    match = re.search(r"/ASTER_IMG/([^/]+)/", str(path_value))
    return match.group(1) if match else ""


def _load_dataset_rows_for_sample(dataset_csv: str | Path, sample_id: str) -> pd.DataFrame:
    dataset_path = Path(dataset_csv)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Arquivo nao encontrado: {dataset_path}")

    dataset_df = pd.read_csv(dataset_path)
    extracted_sample_ids = dataset_df["path"].astype(str).map(_extract_sample_id_from_path)
    filtered_df = dataset_df.loc[extracted_sample_ids == str(sample_id)].copy()
    if filtered_df.empty:
        raise ValueError(f"Nenhuma linha encontrada no dataset para a amostra {sample_id}.")
    return filtered_df
def _load_points_metadata(project_root: Path) -> pd.DataFrame:
    excel_path = project_root / "data" / "banco.xlsx"
    metadata = pd.read_excel(excel_path, sheet_name="Banco de Dados Positivo-Negativ").copy()
    metadata["numero_amostra"] = metadata["numero_amostra"].astype(str)
    return metadata


def _finalize_probability_ranking(
    ranking_df: pd.DataFrame,
    *,
    decision_threshold: float,
    decision_threshold_name: str,
    class_names: Sequence[str] = DEFAULT_CLASS_NAMES,
) -> pd.DataFrame:
    if ranking_df.empty:
        columns = [
            "rank",
            "numero_amostra",
            "prob_pos",
            "prob_neg",
            "pred_label_threshold",
            "decision_threshold",
            "decision_threshold_name",
            "latitude_wgs84_decimal",
            "longitude_wgs84_decimal",
            "classe_balanceamento",
            "litologia_padronizada",
            "path",
        ]
        return pd.DataFrame(columns=columns)

    df = ranking_df.copy()
    df["numero_amostra"] = df["numero_amostra"].astype(str)
    df["prob_pos"] = pd.to_numeric(df["prob_pos"], errors="coerce").fillna(0.0).clip(0.0, 1.0)
    df["prob_neg"] = 1.0 - df["prob_pos"]

    pred_idx = (df["prob_pos"] >= float(decision_threshold)).astype(int)
    df["pred_label_threshold"] = pred_idx.map({0: class_names[0], 1: class_names[1]})
    df["decision_threshold"] = float(decision_threshold)
    df["decision_threshold_name"] = str(decision_threshold_name)

    df = df.sort_values(["prob_pos", "numero_amostra"], ascending=[False, True]).reset_index(drop=True)
    if "rank" in df.columns:
        df = df.drop(columns=["rank"])
    df.insert(0, "rank", np.arange(1, len(df) + 1, dtype=int))

    preferred_columns = [
        "rank",
        "numero_amostra",
        "prob_pos",
        "prob_neg",
        "pred_label_threshold",
        "decision_threshold",
        "decision_threshold_name",
        "latitude_wgs84_decimal",
        "longitude_wgs84_decimal",
        "classe_balanceamento",
        "litologia_padronizada",
        "path",
    ]
    ordered_columns = preferred_columns + [col for col in df.columns if col not in preferred_columns]
    return df[ordered_columns]


def _cache_supports_probability_ranking(df: pd.DataFrame) -> bool:
    required_columns = {"numero_amostra", "prob_pos", "path"}
    return required_columns.issubset(df.columns)


def _load_transfer_bundle_for_threshold(
    project_root: Path,
    threshold_mode: str,
    *,
    model_key: str,
) -> TransferInferenceBundle:
    normalized_model_key = _normalize_model_key(model_key)
    if normalized_model_key == "a11_pipeline_e2e":
        if threshold_mode == "threshold_0.5":
            return load_a11_transfer_inference_bundle(
                project_root=project_root,
                decision_threshold=0.5,
                decision_threshold_name="threshold_0.5",
            )
        return load_a11_transfer_inference_bundle(
            project_root=project_root,
        )
    if threshold_mode == "threshold_0.5":
        return load_transfer_inference_bundle(
            project_root=project_root,
            decision_threshold=0.5,
            decision_threshold_name="threshold_0.5",
        )

    return load_transfer_inference_bundle(
        project_root=project_root,
        decision_threshold_name=threshold_mode,
    )


def build_dataset_probability_ranking(
    project_root: str | Path | None = None,
    threshold_mode: str = "threshold_f1",
    force_refresh: bool = False,
    *,
    model_key: str = "a08_transfer_learning",
    chunksize: int = 32,
    batch_size: int = 32,
) -> pd.DataFrame:
    """
    Gera ranking completo da base `pixels_dataset.csv` usando o modelo selecionado.
    """
    root = resolve_project_root(project_root)
    normalized_model_key = _normalize_model_key(model_key)
    cache_path = _ranking_cache_path(root, model_key=normalized_model_key)
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    decision_threshold, decision_threshold_name = _resolve_threshold_config(
        root,
        threshold_mode=threshold_mode,
        model_key=normalized_model_key,
    )

    if cache_path.exists() and not force_refresh:
        cached_df = pd.read_csv(cache_path)
        if _cache_supports_probability_ranking(cached_df):
            return _finalize_probability_ranking(
                cached_df,
                decision_threshold=decision_threshold,
                decision_threshold_name=decision_threshold_name,
            )

    bundle = _load_transfer_bundle_for_threshold(
        root,
        threshold_mode,
        model_key=normalized_model_key,
    )
    metadata = _load_points_metadata(root)
    predictions: list[pd.DataFrame] = []

    reader = pd.read_csv(bundle.dataset_csv, chunksize=int(chunksize))
    for chunk_df in reader:
        if chunk_df.empty:
            continue

        x_chunk, _ = dataframe_to_cnn_tensor(
            chunk_df,
            data_format="channels_last",
            dtype=np.float32,
        )
        x_ready, _ = adapt_cnn_input_tensor(
            x_chunk,
            data_format="channels_last",
            resize_to=bundle.target_size,
            target_channels=bundle.target_channels,
            normalization=bundle.normalization,
            normalizer=bundle.normalizer,
        )
        raw_pred = bundle.model.predict(x_ready, batch_size=int(batch_size), verbose=0)
        prob_pos = np.clip(np.asarray(raw_pred, dtype=float).reshape(-1), 0.0, 1.0)

        predictions.append(
            pd.DataFrame(
                {
                    "numero_amostra": chunk_df["path"].map(_extract_sample_id_from_path).astype(str),
                    "path": chunk_df["path"].astype(str),
                    "prob_pos": prob_pos.astype(float),
                }
            )
        )

    if not predictions:
        raise RuntimeError("Nao foi possivel gerar o ranking: dataset sem linhas processaveis.")

    ranking_base = pd.concat(predictions, ignore_index=True)
    ranking_base = ranking_base[ranking_base["numero_amostra"] != ""].copy()
    ranking_base = (
        ranking_base.groupby("numero_amostra", as_index=False)
        .agg(
            {
                "prob_pos": "mean",
                "path": "first",
            }
        )
    )

    ranking_base["numero_amostra"] = ranking_base["numero_amostra"].astype(str)
    ranking_base = ranking_base.merge(metadata, on="numero_amostra", how="left")

    final_df = _finalize_probability_ranking(
        ranking_base,
        decision_threshold=decision_threshold,
        decision_threshold_name=decision_threshold_name,
        class_names=bundle.class_names,
    )
    final_df.to_csv(cache_path, index=False)
    return final_df


def compute_gradcam_heatmap(
    model: Any,
    image: np.ndarray,
    *,
    class_index: int = 1,
) -> np.ndarray:
    """
    Calcula o mapa Grad-CAM para uma unica amostra ja pronta para o modelo.
    """
    import tensorflow as tf
    from tensorflow import keras  # type: ignore[import-untyped]

    image_array = np.asarray(image, dtype=np.float32)
    if image_array.ndim != 3:
        raise ValueError(f"Esperado tensor 3D (H, W, C), recebido shape={image_array.shape}.")

    img_tensor = tf.expand_dims(tf.cast(image_array, tf.float32), axis=0)
    last_conv_output = None

    with tf.GradientTape() as tape:
        x = img_tensor
        for layer in model.layers:
            if isinstance(layer, keras.layers.InputLayer):
                continue
            x = layer(x)
            if len(x.shape) == 4:
                last_conv_output = x
                tape.watch(last_conv_output)
        predictions = x

        if last_conv_output is None:
            raise ValueError("Nenhuma camada convolucional 4D encontrada para Grad-CAM.")

        if predictions.shape[-1] == 1:
            class_output = 1.0 - predictions[:, 0] if int(class_index) == 0 else predictions[:, 0]
        else:
            class_output = predictions[:, int(class_index)]

    grads = tape.gradient(class_output, last_conv_output)
    if grads is None:
        raise RuntimeError("Nao foi possivel calcular gradientes para o Grad-CAM.")

    weights = tf.reduce_mean(grads, axis=(0, 1, 2))
    cam = tf.reduce_sum(last_conv_output[0] * weights, axis=-1).numpy()
    cam = np.maximum(cam, 0.0)
    if float(np.max(cam)) > 0.0:
        cam = cam / float(np.max(cam))

    cam_resized = tf.image.resize(
        cam[..., np.newaxis],
        size=(image_array.shape[0], image_array.shape[1]),
        method="bilinear",
        antialias=True,
    ).numpy()[..., 0]
    return np.clip(cam_resized, 0.0, 1.0).astype(np.float32, copy=False)


def generate_dataset_sample_gradcam(
    sample_id: str,
    *,
    project_root: str | Path | None = None,
    model_key: str = "a11_pipeline_e2e",
    threshold_mode: str | None = None,
    class_index: int = 1,
) -> dict[str, Any]:
    """
    Gera Grad-CAM e overlays para uma amostra do dataset consolidado.
    """
    root = resolve_project_root(project_root)
    normalized_model_key = _normalize_model_key(model_key)
    resolved_threshold_mode = threshold_mode or (
        "threshold_default" if normalized_model_key == "a11_pipeline_e2e" else "threshold_f1"
    )
    bundle = _load_transfer_bundle_for_threshold(
        root,
        resolved_threshold_mode,
        model_key=normalized_model_key,
    )

    sample_rows = _load_dataset_rows_for_sample(bundle.dataset_csv, sample_id)
    sample_df = sample_rows.iloc[[0]].copy()
    raw_tensor, _ = dataframe_to_cnn_tensor(
        sample_df,
        data_format="channels_last",
        dtype=np.float32,
    )
    raw_chip = np.asarray(raw_tensor[0], dtype=np.float32)

    resized_raw_batch, _ = adapt_cnn_input_tensor(
        np.expand_dims(raw_chip, axis=0),
        data_format="channels_last",
        resize_to=bundle.target_size,
        target_channels=bundle.target_channels,
        normalization="none",
    )
    model_batch, _ = adapt_cnn_input_tensor(
        np.expand_dims(raw_chip, axis=0),
        data_format="channels_last",
        resize_to=bundle.target_size,
        target_channels=bundle.target_channels,
        normalization=bundle.normalization,
        normalizer=bundle.normalizer,
    )

    resized_chip = np.asarray(resized_raw_batch[0], dtype=np.float32)
    model_input = np.asarray(model_batch[0], dtype=np.float32)
    prob_pos = _predict_positive_probability(bundle, model_batch)
    pred_class = int(prob_pos >= float(bundle.decision_threshold))
    pred_label = str(bundle.class_names[pred_class])
    heatmap = compute_gradcam_heatmap(
        bundle.model,
        model_input,
        class_index=int(class_index),
    )
    false_color = build_false_color_preview(resized_chip, band_order=DEFAULT_BAND_ORDER)
    rgb_preview = build_rgb_preview(resized_chip, band_order=DEFAULT_BAND_ORDER)
    overlay = overlay_activation_heatmap(false_color, heatmap, alpha=0.45, colormap="jet")

    return {
        "sample_id": str(sample_id),
        "path": str(sample_df["path"].iloc[0]),
        "prob_pos": float(prob_pos),
        "pred_class": pred_class,
        "pred_label": pred_label,
        "decision_threshold": float(bundle.decision_threshold),
        "decision_threshold_name": str(bundle.decision_threshold_name),
        "class_index": int(class_index),
        "class_name": str(bundle.class_names[int(class_index)]),
        "raw_shape": tuple(int(v) for v in raw_chip.shape),
        "input_shape": tuple(int(v) for v in model_input.shape),
        "false_color_preview": false_color,
        "rgb_preview": rgb_preview,
        "heatmap": heatmap,
        "overlay": overlay,
    }
def assess_chip_quality(
    chip_array: np.ndarray,
    *,
    band_order: Sequence[str] = DEFAULT_BAND_ORDER,
) -> dict[str, Any]:
    """
    Gera sinais simples de qualidade para apoiar a leitura do resultado.
    """
    chip = np.asarray(chip_array, dtype=np.float32)
    if chip.ndim != 3:
        raise ValueError(f"Esperado chip 3D (H, W, C), recebido shape={chip.shape}.")

    finite_mask = np.isfinite(chip)
    finite_ratio = float(np.mean(finite_mask))

    if not np.any(finite_mask):
        return {
            "severity": "critical",
            "warnings": ["chip_sem_pixels_validos"],
            "finite_ratio": 0.0,
            "zero_ratio": 1.0,
            "bright_ratio": 0.0,
            "dark_ratio": 1.0,
            "low_dynamic_bands": list(band_order),
        }

    valid_values = chip[finite_mask]
    zero_ratio = float(np.mean(np.isclose(valid_values, 0.0)))

    p2 = np.nanpercentile(chip, 2, axis=(0, 1))
    p98 = np.nanpercentile(chip, 98, axis=(0, 1))
    dynamic_span = p98 - p2
    low_dynamic_bands = [
        str(band_order[idx])
        for idx, span in enumerate(dynamic_span)
        if (not np.isfinite(span)) or float(span) <= 1e-6
    ]

    rgb_preview = build_rgb_preview(chip, band_order=band_order)
    rgb_brightness = np.mean(rgb_preview, axis=-1)
    bright_ratio = float(np.mean(rgb_brightness >= 0.92))
    dark_ratio = float(np.mean(rgb_brightness <= 0.08))

    warnings_list: list[str] = []
    severity = "ok"

    if finite_ratio < 0.98:
        warnings_list.append("baixo_percentual_de_pixels_finitos")
        severity = "warning"
    if zero_ratio > 0.20:
        warnings_list.append("muitos_pixels_nulos_ou_sem_resposta")
        severity = "warning"
    if len(low_dynamic_bands) >= 3:
        warnings_list.append("bandas_com_baixa_variacao_espectral")
        severity = "warning"
    if bright_ratio > 0.20:
        warnings_list.append("possivel_nuvem_ou_saturacao")
        severity = "warning"
    if dark_ratio > 0.35:
        warnings_list.append("possivel_sombra_ou_no_data")
        severity = "warning"

    if finite_ratio < 0.90 or bright_ratio > 0.35 or len(low_dynamic_bands) >= 6:
        severity = "critical"

    return {
        "severity": severity,
        "warnings": warnings_list,
        "finite_ratio": finite_ratio,
        "zero_ratio": zero_ratio,
        "bright_ratio": bright_ratio,
        "dark_ratio": dark_ratio,
        "low_dynamic_bands": low_dynamic_bands,
    }


def load_transfer_inference_bundle(
    project_root: str | Path | None = None,
    *,
    dataset_csv: str | Path | None = None,
    extracted_codes_json: str | Path | None = None,
    model_path: str | Path | None = None,
    test_size: float = 0.20,
    val_size: float = 0.20,
    seed: int = 42,
    target_size: tuple[int, int] = (160, 160),
    normalization: str = "zscore",
    class_names: tuple[str, str] = DEFAULT_CLASS_NAMES,
    decision_threshold: float | None = None,
    decision_threshold_name: str = "threshold_f1",
) -> TransferInferenceBundle:
    """
    Carrega o modelo do A08 e reconstrói o normalizador do treino.
    """
    root = resolve_project_root(project_root)
    dataset_csv_path = Path(dataset_csv) if dataset_csv is not None else root / "data" / "pixels_dataset.csv"
    codes_path = Path(extracted_codes_json) if extracted_codes_json is not None else root / "data" / "extracted_codes.json"
    model_file = Path(model_path) if model_path is not None else root / "outputs" / "a08_transfer_learning" / "best_model.keras"
    resolved_threshold, resolved_threshold_name = _resolve_threshold_from_metrics(
        root,
        threshold_name=decision_threshold_name,
        fallback_threshold=0.5 if decision_threshold is None else float(decision_threshold),
    )
    if decision_threshold is not None:
        resolved_threshold = float(decision_threshold)
        resolved_threshold_name = decision_threshold_name or "manual"

    return _build_transfer_inference_bundle(
        root=root,
        model_file=model_file,
        dataset_csv_path=dataset_csv_path,
        codes_path=codes_path,
        test_size=test_size,
        val_size=val_size,
        seed=seed,
        target_size=target_size,
        target_channels=9,
        normalization=normalization,
        class_names=class_names,
        decision_threshold=resolved_threshold,
        decision_threshold_name=resolved_threshold_name,
        model_name="Transfer Learning (A08)",
    )


def load_a11_transfer_inference_bundle(
    project_root: str | Path | None = None,
    *,
    config_path: str | Path | None = None,
    model_path: str | Path | None = None,
    class_names: tuple[str, str] = DEFAULT_CLASS_NAMES,
    decision_threshold: float | None = None,
    decision_threshold_name: str = "threshold_default",
) -> TransferInferenceBundle:
    """
    Carrega o modelo final do A11 a partir do config oficial do artefato.
    """
    import yaml

    root = resolve_project_root(project_root)
    resolved_config_path = (
        Path(config_path)
        if config_path is not None
        else root / "artefatos" / "a11_pipeline_e2e" / "config.yaml"
    )
    resolved_config_path = resolved_config_path.resolve()
    if not resolved_config_path.exists():
        raise FileNotFoundError(f"Arquivo nao encontrado: {resolved_config_path}")

    with resolved_config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    config_dir = resolved_config_path.parent
    dataset_csv_path = _resolve_config_path(config_dir, config["paths"]["dataset_csv"])
    codes_path = _resolve_config_path(config_dir, config["paths"]["extracted_codes_json"])
    outputs_models_dir = _resolve_config_path(config_dir, config["paths"]["outputs_models"])
    model_file = Path(model_path).resolve() if model_path is not None else outputs_models_dir / "best_model.keras"

    resolved_threshold = (
        float(config["evaluation"]["threshold_default"])
        if decision_threshold is None
        else float(decision_threshold)
    )
    resolved_threshold_name = (
        "threshold_default" if decision_threshold is None else str(decision_threshold_name)
    )

    return _build_transfer_inference_bundle(
        root=root,
        model_file=model_file,
        dataset_csv_path=dataset_csv_path,
        codes_path=codes_path,
        test_size=float(config["data"]["test_size"]),
        val_size=float(config["data"]["val_size"]),
        seed=int(config["seed"]),
        target_size=tuple(int(v) for v in config["model"]["resize_to"]),
        target_channels=int(config["data"]["num_bands"]),
        normalization=str(config["data"]["normalization_method"]),
        class_names=class_names,
        decision_threshold=resolved_threshold,
        decision_threshold_name=resolved_threshold_name,
        model_name="Pipeline Final (A11)",
    )


def predict_chip_array(
    bundle: TransferInferenceBundle,
    chip_9bands: np.ndarray,
    *,
    lat: float | None = None,
    lon: float | None = None,
    chip_path: str | Path | None = None,
    granule_id: str | None = None,
    acquisition_time: str | None = None,
    cloud_cover: float | None = None,
    bbox_wgs84: tuple[float, float, float, float] | None = None,
    preview_png_path: str | Path | None = None,
    false_color_preview_path: str | Path | None = None,
    rgb_preview_path: str | Path | None = None,
) -> PredictionResult:
    """
    Aplica pre-processamento consistente com o A08 e executa inferencia.
    """
    chip_arr = np.asarray(chip_9bands, dtype=np.float32)
    if chip_arr.ndim != 3:
        raise ValueError(f"Esperado chip 3D (H, W, C), recebido shape={chip_arr.shape}.")
    if chip_arr.shape[-1] != bundle.target_channels:
        raise ValueError(
            f"Numero de canais inesperado: recebido={chip_arr.shape[-1]}, esperado={bundle.target_channels}."
        )

    batch, _ = adapt_cnn_input_tensor(
        np.expand_dims(chip_arr, axis=0),
        data_format="channels_last",
        resize_to=bundle.target_size,
        target_channels=bundle.target_channels,
        normalization=bundle.normalization,
        normalizer=bundle.normalizer,
    )

    prob_pos = _predict_positive_probability(bundle, batch)
    pred_class = int(prob_pos >= bundle.decision_threshold)
    if false_color_preview_path is None and preview_png_path is not None:
        false_color_preview_path = preview_png_path

    preview_path_value: str | None = None
    preview_false_color_value: str | None = None
    preview_rgb_value: str | None = None

    if false_color_preview_path is not None:
        preview_false = build_false_color_preview(chip_arr, band_order=DEFAULT_BAND_ORDER)
        preview_false_color_value = str(save_preview_png(preview_false, false_color_preview_path))
        preview_path_value = preview_false_color_value

    if rgb_preview_path is not None:
        preview_rgb = build_rgb_preview(chip_arr, band_order=DEFAULT_BAND_ORDER)
        preview_rgb_value = str(save_preview_png(preview_rgb, rgb_preview_path))

    quality_report = assess_chip_quality(chip_arr, band_order=DEFAULT_BAND_ORDER)

    return PredictionResult(
        lat=float(lat) if lat is not None else np.nan,
        lon=float(lon) if lon is not None else np.nan,
        prob_pos=prob_pos,
        pred_class=pred_class,
        pred_label=bundle.class_names[pred_class],
        chip_path=str(chip_path) if chip_path is not None else None,
        granule_id=granule_id,
        acquisition_time=acquisition_time,
        cloud_cover=float(cloud_cover) if cloud_cover is not None else None,
        bbox_wgs84=bbox_wgs84,
        chip_shape=tuple(int(v) for v in chip_arr.shape),
        preview_png_path=preview_path_value,
        preview_false_color_path=preview_false_color_value,
        preview_rgb_path=preview_rgb_value,
        quality_report=quality_report,
    )


def search_earthdata_granules(**kwargs):
    import earthaccess

    return earthaccess.search_data(**kwargs)


def _granule_identifier(granule: Any) -> str | None:
    for attr_name in ("meta", "umm"):
        data = getattr(granule, attr_name, None)
        if isinstance(data, dict):
            for key in ("native-id", "native_id", "GranuleUR", "ShortName", "EntryTitle"):
                value = data.get(key)
                if value:
                    return str(value)
    for attr_name in ("id", "granule_id", "title"):
        value = getattr(granule, attr_name, None)
        if value:
            return str(value)
    return str(granule) if granule is not None else None


def _point_cache_dir(cache_root: Path, lat: float, lon: float) -> Path:
    key = f"{lat:.6f}_{lon:.6f}"
    digest = hashlib.md5(key.encode("utf-8")).hexdigest()[:12]
    safe_name = _sanitize_filename(f"{lat:.6f}_{lon:.6f}_{digest}")
    out_dir = cache_root / safe_name
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def read_chip_tif(chip_path: str | Path) -> np.ndarray:
    """
    Lê GeoTIFF multibanda e devolve array em channels_last.
    """
    import rasterio

    with rasterio.open(chip_path) as src:
        chip = src.read().astype(np.float32)
    return np.moveaxis(chip, 0, -1)


def clear_cache_dir(cache_root: str | Path) -> None:
    """
    Remove o diretório de cache do app local.
    """
    cache_path = Path(cache_root)
    if cache_path.exists():
        shutil.rmtree(cache_path)


def predict_point_from_earthdata(
    bundle: TransferInferenceBundle,
    lat: float,
    lon: float,
    *,
    netrc_path: str,
    cache_root: str | Path | None = None,
    start_date: str = "2000-01-01",
    end_date: str = "2007-12-31",
    chip_side_m: float = 2000.0,
    margin_m: float = 200.0,
    max_granules: int = 10,
    force_refresh: bool = False,
) -> PredictionResult:
    """
    Busca um granule ASTER no EarthData, recorta o chip e roda inferencia.
    """
    try:
        from tiles.bbox import bbox_with_point_inside
        from tiles.config import Config
        from tiles.earthaccess_utils import (
            build_search_kwargs,
            choose_best_granule,
            cloud_cover_value,
            download_granule,
            filter_granules_by_date,
            granule_start_time,
            login_earthdata,
        )
        from tiles.multiband import crop_and_stack_multiband, list_band_tifs
    except ImportError:  # pragma: no cover - fallback para imports via pacote src.*
        from src.tiles.bbox import bbox_with_point_inside
        from src.tiles.config import Config
        from src.tiles.earthaccess_utils import (
            build_search_kwargs,
            choose_best_granule,
            cloud_cover_value,
            download_granule,
            filter_granules_by_date,
            granule_start_time,
            login_earthdata,
        )
        from src.tiles.multiband import crop_and_stack_multiband, list_band_tifs

    cache_root_path = Path(cache_root) if cache_root is not None else bundle.project_root / "outputs" / "a09_streamlit_cache"
    point_dir = _point_cache_dir(cache_root_path, lat=float(lat), lon=float(lon))
    chip_path = point_dir / "chip_earthdata_multiband.tif"
    preview_path = point_dir / "chip_false_color.png"
    preview_rgb_path = point_dir / "chip_rgb.png"

    bbox = bbox_with_point_inside(
        lat=float(lat),
        lon=float(lon),
        chip_side_m=float(chip_side_m),
        margin_m=float(margin_m),
        seed=bundle.seed,
    )

    if force_refresh:
        for path in (chip_path, preview_path, preview_rgb_path):
            if path.exists():
                path.unlink()

    login_earthdata(netrc_path)

    cfg = Config(
        start_date=start_date,
        end_date=end_date,
        max_granules=max_granules,
        out_root=str(cache_root_path),
        chip_side_m=chip_side_m,
        margin_m=margin_m,
        keep_downloads=True,
    )
    search_kwargs = build_search_kwargs(cfg, bbox)
    raw_granules = list(search_earthdata_granules(**search_kwargs))
    granules = filter_granules_by_date(raw_granules, start_date, end_date)
    if raw_granules and not granules:
        warnings.warn(
            "A busca do EarthData retornou granules, mas o filtro local por data descartou todos; "
            "usando os resultados originais da busca porque o periodo temporal ja foi informado ao EarthData.",
            RuntimeWarning,
            stacklevel=2,
        )
        granules = raw_granules
    if not granules:
        raise RuntimeError("Nenhum granule ASTER encontrado para o ponto selecionado.")

    best_granule = choose_best_granule(granules)
    if best_granule is None:
        raise RuntimeError("Nao foi possivel selecionar um granule ASTER valido.")

    if force_refresh or not chip_path.exists():
        download_granule(best_granule, point_dir, dry_run=False)
        tif_paths = list_band_tifs(point_dir)
        if len(tif_paths) < len(DEFAULT_BAND_ORDER):
            raise RuntimeError(
                f"Granule com bandas insuficientes para inferencia: encontrado={len(tif_paths)}, esperado={len(DEFAULT_BAND_ORDER)}."
            )

        generated_chip = crop_and_stack_multiband(
            tif_paths=tif_paths[: len(DEFAULT_BAND_ORDER)],
            out_tif=chip_path,
            bbox_wgs84=bbox,
        )
        if generated_chip is None or not chip_path.exists():
            raise RuntimeError("Falha ao recortar e empilhar o chip ASTER multibanda.")

    chip_array = read_chip_tif(chip_path)
    if chip_array.shape[-1] != len(DEFAULT_BAND_ORDER):
        raise RuntimeError(
            f"Chip multibanda invalido para o modelo: shape={chip_array.shape}."
        )

    acquisition_time = granule_start_time(best_granule)
    return predict_chip_array(
        bundle,
        chip_array,
        lat=float(lat),
        lon=float(lon),
        chip_path=chip_path,
        granule_id=_granule_identifier(best_granule),
        acquisition_time=acquisition_time.isoformat() if acquisition_time is not None else None,
        cloud_cover=cloud_cover_value(best_granule),
        bbox_wgs84=bbox,
        preview_png_path=preview_path,
        false_color_preview_path=preview_path,
        rgb_preview_path=preview_rgb_path,
    )
