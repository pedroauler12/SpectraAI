from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import rasterio

from .bbox import bbox_with_point_inside
from .multiband import crop_and_stack_multiband, list_band_tifs, summarize_tif


def read_aster_tile(
    tif_path: Path | str,
    *,
    expected_bands: Optional[List[int]] = None,
    require_crs: bool = True,
    allow_masked: bool = True,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Lê um tile ASTER multibanda e aplica validações básicas.

    Args:
        tif_path: Caminho para o GeoTIFF multibanda.
        expected_bands: Lista opcional de contagens de bandas aceitáveis
            (ex.: [9] para VNIR+SWIR, [14] para VNIR+SWIR+TIR).
        require_crs: Se True, exige que o CRS esteja definido.
        allow_masked: Se False, falha quando todos os pixels são NoData.

    Returns:
        (data, meta) onde data tem shape (bands, H, W) e meta é um dict
        com informações básicas (crs, bounds, res, dtype, nodata).

    Raises:
        FileNotFoundError: Se o arquivo não existir.
        ValueError: Se o arquivo falhar em alguma validação básica.
        rasterio.errors.RasterioIOError: Se houver erro de leitura.
    """
    path = Path(tif_path)
    if not path.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {path}")

    if path.suffix.lower() not in {".tif", ".tiff"}:
        raise ValueError(f"Extensão inválida para GeoTIFF: {path.suffix}")

    with rasterio.open(path) as src:
        if require_crs and src.crs is None:
            raise ValueError("CRS ausente no tile.")

        if src.count < 2:
            raise ValueError(f"Tile precisa ser multibanda (count={src.count}).")

        if expected_bands and src.count not in expected_bands:
            raise ValueError(f"Bandas inválidas: {src.count} (esperado {expected_bands}).")

        if src.width <= 0 or src.height <= 0:
            raise ValueError(f"Dimensão inválida: {src.width}x{src.height}.")

        data = src.read(masked=True)
        if not allow_masked and hasattr(data, "mask"):
            if np.all(data.mask):
                raise ValueError("Tile contém apenas pixels NoData.")

        meta = {
            "crs": str(src.crs),
            "bounds": tuple(src.bounds),
            "res": tuple(src.res),
            "bands": src.count,
            "height": src.height,
            "width": src.width,
            "dtype": str(src.dtypes[0]),
            "nodata": src.nodata,
        }

    return np.asarray(data), meta


__all__ = [
    "bbox_with_point_inside",
    "crop_and_stack_multiband",
    "list_band_tifs",
    "summarize_tif",
    "read_aster_tile",
]
