"""
Utilitarios para preparar dados tabulares de pixels para entrada em CNN.

Fluxo principal:
1) Ler colunas ``pixel_*`` e reorganizar para tensor 4D.
2) Ajustar formato de canais (``channels_last`` ou ``channels_first``).
3) Aplicar normalizacao por canal, com estatisticas reaproveitaveis.
4) (Opcional) Extrair labels a partir de ``extracted_codes.json``.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


def get_ordered_pixel_columns(
    df: pd.DataFrame,
    pixel_prefix: str = "pixel_",
) -> list[str]:
    """
    Retorna colunas de pixel ordenadas por indice numerico.
    """
    pixel_cols = [c for c in df.columns if c.startswith(pixel_prefix)]
    if not pixel_cols:
        raise ValueError(f"Nenhuma coluna encontrada com prefixo '{pixel_prefix}'.")

    def _sort_key(col_name: str):
        suffix = col_name[len(pixel_prefix):]
        return (0, int(suffix)) if suffix.isdigit() else (1, suffix)

    pixel_cols.sort(key=_sort_key)
    return pixel_cols


def infer_cnn_shape(
    df: pd.DataFrame,
    *,
    pixel_columns: list[str] | None = None,
    n_channels: int | None = None,
    height: int | None = None,
    width: int | None = None,
) -> dict:
    """
    Infere ``n_channels``, ``height`` e ``width`` de um DataFrame com ``pixel_*``.
    """
    pixel_columns = pixel_columns or get_ordered_pixel_columns(df)
    n_pixels_total = len(pixel_columns)

    if n_channels is None and "count" in df.columns:
        unique_counts = df["count"].dropna().unique()
        if len(unique_counts) == 1 and int(unique_counts[0]) > 0:
            n_channels = int(unique_counts[0])

    if height is None and "height" in df.columns:
        unique_heights = df["height"].dropna().unique()
        if len(unique_heights) == 1 and int(unique_heights[0]) > 0:
            height = int(unique_heights[0])

    if width is None and "width" in df.columns:
        unique_widths = df["width"].dropna().unique()
        if len(unique_widths) == 1 and int(unique_widths[0]) > 0:
            width = int(unique_widths[0])

    if (height is None) ^ (width is None):
        raise ValueError("Informe height e width juntos, ou deixe ambos como None.")

    if n_channels is not None and n_channels <= 0:
        raise ValueError("n_channels deve ser > 0.")

    if height is not None and height <= 0:
        raise ValueError("height deve ser > 0.")

    if width is not None and width <= 0:
        raise ValueError("width deve ser > 0.")

    if n_channels is not None and height is not None and width is not None:
        expected = n_channels * height * width
        if expected != n_pixels_total:
            raise ValueError(
                f"Shape inconsistente: n_channels*height*width={expected}, "
                f"mas existem {n_pixels_total} colunas pixel."
            )
        return {
            "n_channels": n_channels,
            "height": height,
            "width": width,
            "pixels_per_channel": height * width,
        }

    if height is not None and width is not None and n_channels is None:
        pixels_per_channel = height * width
        if n_pixels_total % pixels_per_channel != 0:
            raise ValueError(
                f"Nao foi possivel inferir n_channels: {n_pixels_total} nao e "
                f"multiplo de height*width ({pixels_per_channel})."
            )
        n_channels = n_pixels_total // pixels_per_channel
        return {
            "n_channels": int(n_channels),
            "height": int(height),
            "width": int(width),
            "pixels_per_channel": int(pixels_per_channel),
        }

    if n_channels is not None and height is None and width is None:
        if n_pixels_total % n_channels != 0:
            raise ValueError(
                f"Nao foi possivel inferir dimensoes espaciais: {n_pixels_total} "
                f"nao e multiplo de n_channels ({n_channels})."
            )
        pixels_per_channel = n_pixels_total // n_channels
        side = int(np.sqrt(pixels_per_channel))
        if side * side != pixels_per_channel:
            raise ValueError(
                "Nao foi possivel inferir height/width automaticamente. "
                "Informe height e width explicitamente."
            )
        return {
            "n_channels": int(n_channels),
            "height": int(side),
            "width": int(side),
            "pixels_per_channel": int(pixels_per_channel),
        }

    raise ValueError(
        "Informacoes insuficientes para inferir shape CNN. "
        "Forneca n_channels e/ou height/width (ou colunas count/height/width no DataFrame)."
    )


def dataframe_to_cnn_tensor(
    df: pd.DataFrame,
    *,
    pixel_prefix: str = "pixel_",
    n_channels: int | None = None,
    height: int | None = None,
    width: int | None = None,
    data_format: str = "channels_last",
    dtype: str | np.dtype = np.float32,
) -> tuple[np.ndarray, dict]:
    """
    Converte um DataFrame com colunas ``pixel_*`` para tensor 4D de CNN.

    Retorna:
    - ``X``: shape ``(N, H, W, C)`` quando ``channels_last``.
    - ``X``: shape ``(N, C, H, W)`` quando ``channels_first``.
    """
    if data_format not in {"channels_last", "channels_first"}:
        raise ValueError("data_format deve ser 'channels_last' ou 'channels_first'.")

    pixel_columns = get_ordered_pixel_columns(df, pixel_prefix=pixel_prefix)
    shape_info = infer_cnn_shape(
        df,
        pixel_columns=pixel_columns,
        n_channels=n_channels,
        height=height,
        width=width,
    )

    flat = df[pixel_columns].to_numpy(dtype=dtype, copy=True)
    n_samples = flat.shape[0]

    x_channels_first = flat.reshape(
        n_samples,
        shape_info["n_channels"],
        shape_info["height"],
        shape_info["width"],
    )

    if data_format == "channels_first":
        x_out = x_channels_first
    else:
        x_out = np.transpose(x_channels_first, (0, 2, 3, 1))

    meta = {
        **shape_info,
        "n_samples": int(n_samples),
        "pixel_columns": pixel_columns,
        "data_format": data_format,
    }
    return x_out, meta


def labels_from_extracted_codes(
    paths: Iterable[str],
    extracted_codes_path: str | Path,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Gera labels binarios a partir de paths e do arquivo ``extracted_codes.json``.

    Convencao:
    - 1: positivo
    - 0: negativo
    - -1: nao encontrado nas listas
    """
    extracted_codes_path = Path(extracted_codes_path)
    with extracted_codes_path.open("r", encoding="utf-8") as f:
        codes = json.load(f)

    positives = set(codes.get("positivos", []))
    negatives = set(codes.get("negativos", []))
    all_ids = sorted(positives | negatives, key=len, reverse=True)

    if not all_ids:
        raise ValueError("extracted_codes.json nao contem codigos em positivos/negativos.")

    pattern = "|".join(re.escape(code) for code in all_ids)
    path_series = pd.Series(list(paths), dtype=str)
    image_ids = path_series.str.extract(rf"({pattern})", expand=False)

    labels = np.where(
        image_ids.isin(positives),
        1,
        np.where(image_ids.isin(negatives), 0, -1),
    ).astype(np.int64)

    image_ids = image_ids.fillna("").to_numpy(dtype=object)
    return labels, image_ids


def fit_channel_normalizer(
    x: np.ndarray,
    *,
    method: str = "zscore",
    data_format: str = "channels_last",
    eps: float = 1e-8,
) -> dict:
    """
    Ajusta normalizacao por canal usando apenas o conjunto de treino.
    """
    if x.ndim != 4:
        raise ValueError(f"Esperado tensor 4D, recebido shape={x.shape}.")
    if data_format not in {"channels_last", "channels_first"}:
        raise ValueError("data_format deve ser 'channels_last' ou 'channels_first'.")
    if method not in {"zscore", "minmax"}:
        raise ValueError("method deve ser 'zscore' ou 'minmax'.")

    axes = (0, 1, 2) if data_format == "channels_last" else (0, 2, 3)
    keepdims = True

    if method == "zscore":
        mean = x.mean(axis=axes, keepdims=keepdims)
        std = x.std(axis=axes, keepdims=keepdims)
        std = np.where(std < eps, 1.0, std)
        return {
            "method": "zscore",
            "data_format": data_format,
            "mean": mean.astype(np.float32),
            "std": std.astype(np.float32),
            "eps": float(eps),
        }

    min_val = x.min(axis=axes, keepdims=keepdims)
    max_val = x.max(axis=axes, keepdims=keepdims)
    span = max_val - min_val
    span = np.where(np.abs(span) < eps, 1.0, span)
    return {
        "method": "minmax",
        "data_format": data_format,
        "min": min_val.astype(np.float32),
        "span": span.astype(np.float32),
        "eps": float(eps),
    }


def apply_channel_normalizer(
    x: np.ndarray,
    normalizer: dict,
    *,
    clip_minmax: bool = True,
) -> np.ndarray:
    """
    Aplica normalizacao por canal a um tensor 4D.
    """
    if x.ndim != 4:
        raise ValueError(f"Esperado tensor 4D, recebido shape={x.shape}.")

    method = normalizer.get("method")
    if method == "zscore":
        x_out = (x - normalizer["mean"]) / normalizer["std"]
        return x_out.astype(np.float32, copy=False)

    if method == "minmax":
        x_out = (x - normalizer["min"]) / normalizer["span"]
        if clip_minmax:
            x_out = np.clip(x_out, 0.0, 1.0)
        return x_out.astype(np.float32, copy=False)

    raise ValueError("Normalizador invalido: campo 'method' deve ser 'zscore' ou 'minmax'.")


def prepare_cnn_inputs(
    df: pd.DataFrame,
    *,
    labels: Iterable[int] | np.ndarray | None = None,
    extracted_codes_path: str | Path | None = None,
    path_column: str = "path",
    drop_invalid_labels: bool = True,
    pixel_prefix: str = "pixel_",
    n_channels: int | None = None,
    height: int | None = None,
    width: int | None = None,
    data_format: str = "channels_last",
    dtype: str | np.dtype = np.float32,
    normalization: str = "none",
    normalizer: dict | None = None,
) -> dict:
    """
    Pipeline completo de preparacao de tensores para CNN.
    """
    x, shape_info = dataframe_to_cnn_tensor(
        df,
        pixel_prefix=pixel_prefix,
        n_channels=n_channels,
        height=height,
        width=width,
        data_format=data_format,
        dtype=dtype,
    )

    y = None
    image_ids = None

    if labels is not None:
        y = np.asarray(list(labels), dtype=np.int64)
        if y.shape[0] != len(df):
            raise ValueError(f"Tamanho de labels ({y.shape[0]}) difere de len(df) ({len(df)}).")
    elif extracted_codes_path is not None:
        if path_column not in df.columns:
            raise ValueError(f"Coluna '{path_column}' nao encontrada no DataFrame.")
        y, image_ids = labels_from_extracted_codes(df[path_column], extracted_codes_path)

    if y is not None and drop_invalid_labels:
        valid_mask = y != -1
        x = x[valid_mask]
        y = y[valid_mask]
        if image_ids is not None:
            image_ids = image_ids[valid_mask]

    fitted_normalizer = normalizer
    if normalization != "none":
        if normalization not in {"zscore", "minmax"}:
            raise ValueError("normalization deve ser 'none', 'zscore' ou 'minmax'.")
        if fitted_normalizer is None:
            fitted_normalizer = fit_channel_normalizer(
                x,
                method=normalization,
                data_format=data_format,
            )
        x = apply_channel_normalizer(x, fitted_normalizer)

    return {
        "X": x.astype(np.float32, copy=False),
        "y": y,
        "image_ids": image_ids,
        "shape_info": shape_info,
        "normalizer": fitted_normalizer,
    }
