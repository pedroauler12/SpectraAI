"""
Camada reutilizavel para inferencia geoespacial com o modelo de transfer learning.

Este modulo conecta:
- modelo treinado no A08;
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
    from models.cnn_data_prep import fit_channel_normalizer, prepare_grouped_cnn_splits
    from models.cnn_tf_data_pipeline import adapt_cnn_input_tensor
except ImportError:  # pragma: no cover - fallback para imports via pacote src.*
    from src.models.cnn_data_prep import fit_channel_normalizer, prepare_grouped_cnn_splits
    from src.models.cnn_tf_data_pipeline import adapt_cnn_input_tensor


DEFAULT_CLASS_NAMES = ("Negativo", "Positivo")
DEFAULT_BAND_ORDER = ("B01", "B02", "B03N", "B04", "B05", "B06", "B07", "B08", "B09")


@dataclass
class TransferInferenceBundle:
    """Objetos e parametros necessarios para inferencia consistente com o A08."""

    project_root: Path
    model: Any
    normalizer: dict[str, Any]
    target_size: tuple[int, int]
    target_channels: int
    normalization: str
    class_names: tuple[str, str]
    decision_threshold: float
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


def save_preview_png(preview_rgb: np.ndarray, output_path: str | Path) -> Path:
    """
    Salva preview RGB em PNG para visualizacao no notebook/app.
    """
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
    decision_threshold: float = 0.5,
) -> TransferInferenceBundle:
    """
    Carrega o modelo do A08 e reconstrói o normalizador do treino.
    """
    root = resolve_project_root(project_root)
    dataset_csv_path = Path(dataset_csv) if dataset_csv is not None else root / "data" / "pixels_dataset.csv"
    codes_path = Path(extracted_codes_json) if extracted_codes_json is not None else root / "data" / "extracted_codes.json"
    model_file = Path(model_path) if model_path is not None else root / "outputs" / "a08_transfer_learning" / "best_model.keras"

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
        test_size=test_size,
        val_size=val_size,
        seed=seed,
    )

    x_train = np.asarray(split_data["X_train"], dtype=np.float32)
    normalizer = fit_channel_normalizer(
        x_train,
        method=normalization,
        data_format="channels_last",
    )

    model = _load_keras_model(model_file)
    target_channels = int(split_data["shape_info"]["n_channels"])

    return TransferInferenceBundle(
        project_root=root,
        model=model,
        normalizer=normalizer,
        target_size=tuple(int(v) for v in target_size),
        target_channels=target_channels,
        normalization=normalization,
        class_names=class_names,
        decision_threshold=float(decision_threshold),
        seed=int(seed),
        dataset_csv=dataset_csv_path,
        extracted_codes_json=codes_path,
        model_path=model_file,
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
    preview_path_value: str | None = None

    if preview_png_path is not None:
        preview_rgb = build_false_color_preview(chip_arr, band_order=DEFAULT_BAND_ORDER)
        preview_path_value = str(save_preview_png(preview_rgb, preview_png_path))

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

    bbox = bbox_with_point_inside(
        lat=float(lat),
        lon=float(lon),
        chip_side_m=float(chip_side_m),
        margin_m=float(margin_m),
        seed=bundle.seed,
    )

    if force_refresh:
        for path in (chip_path, preview_path):
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
    )
