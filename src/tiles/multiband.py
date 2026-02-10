from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.vrt import WarpedVRT
from rasterio.windows import from_bounds


def list_band_tifs(folder: Path) -> List[Path]:
    """
    Retorna apenas bandas espectrais VNIR+SWIR.
    Ignora QA e qualquer arquivo que não seja banda.
    """
    tifs = sorted(folder.glob("**/*.tif")) + sorted(folder.glob("**/*.tiff"))

    keep: List[Path] = []
    for p in tifs:
        name = p.name.upper()

        # ignora outputs já gerados
        if p.name.startswith("chip_"):
            continue

        # ignora QA
        if "QA" in name:
            continue

        # mantém VNIR B01, B02, B03N
        if re.search(r"_SRF_VNIR_B(01|02|03N)\.TIF(F)?$", name):
            keep.append(p)
            continue

        # mantém SWIR B04..B09
        if re.search(r"_SRF_SWIR_B0[4-9]\.TIF(F)?$", name):
            keep.append(p)
            continue

    return keep


def crop_and_stack_multiband(
    tif_paths: List[Path],
    out_tif: Path,
    bbox_wgs84: Tuple[float, float, float, float],
    dst_crs: str = "EPSG:4326",
    resampling: Resampling = Resampling.nearest,
) -> Optional[Path]:
    """
    Recorta todos os tifs no bbox e empilha num GeoTIFF multibanda.
    Reamostra tudo para a grade da 1ª banda.
    """
    if not tif_paths:
        return None

    west, south, east, north = bbox_wgs84
    out_tif.parent.mkdir(parents=True, exist_ok=True)

    # Banda referência
    with rasterio.open(tif_paths[0]) as src0:
        with WarpedVRT(src0, crs=dst_crs, resampling=resampling) as ref:
            win = from_bounds(west, south, east, north, transform=ref.transform)
            win = win.round_offsets().round_lengths()

            first = ref.read(1, window=win)
            h, w = first.shape
            transform = rasterio.windows.transform(win, ref.transform)

            meta = ref.meta.copy()
            meta.update(
                {
                    "driver": "GTiff",
                    "count": len(tif_paths),
                    "height": h,
                    "width": w,
                    "crs": dst_crs,
                    "transform": transform,
                }
            )

            stacked = np.zeros((len(tif_paths), h, w), dtype=first.dtype)
            stacked[0] = first

    # Demais bandas alinhadas para a mesma grade
    for i, p in enumerate(tif_paths[1:], start=1):
        with rasterio.open(p) as src:
            with WarpedVRT(
                src,
                crs=dst_crs,
                transform=transform,
                width=w,
                height=h,
                resampling=resampling,
            ) as vrt:
                win2 = from_bounds(west, south, east, north, transform=vrt.transform)
                win2 = win2.round_offsets().round_lengths()
                stacked[i] = vrt.read(1, window=win2)

    with rasterio.open(out_tif, "w", **meta) as dst:
        dst.write(stacked)

    return out_tif


def summarize_tif(tif_path: Path) -> Dict[str, Any]:
    with rasterio.open(tif_path) as src:
        return {
            "crs": str(src.crs),
            "bounds": tuple(src.bounds),
            "res": tuple(src.res),
            "bands": src.count,
            "height": src.height,
            "width": src.width,
            "dtype": str(src.dtypes[0]),
        }
