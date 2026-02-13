from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import rasterio


def _band_stats(masked_band: np.ma.MaskedArray) -> dict:
    if masked_band.mask.all():
        return {"min": None, "max": None, "mean": None, "std": None}
    return {
        "min": float(masked_band.min()),
        "max": float(masked_band.max()),
        "mean": float(masked_band.mean()),
        "std": float(masked_band.std()),
    }


def inspect_tile(tile_path: str | Path) -> dict:
    tile_path = Path(tile_path)
    with rasterio.open(tile_path) as src:
        shape = (src.count, src.height, src.width)
        dtypes = src.dtypes
        crs = src.crs.to_string() if src.crs else None
        transform = src.transform
        nodata = src.nodata

        data = src.read(masked=True)
        total_elements = data.size
        if total_elements == 0:
            nodata_pct = None
        else:
            nodata_count = np.ma.getmaskarray(data).sum()
            nodata_pct = float(nodata_count / total_elements * 100.0)

        stats = {}
        for band_idx in range(src.count):
            band_stats = _band_stats(data[band_idx])
            for stat_name, stat_value in band_stats.items():
                stats[f"{stat_name}_b{band_idx + 1}"] = stat_value

    info = {
        "path": str(tile_path),
        "filename": tile_path.name,
        "shape": shape,
        "count": shape[0],
        "height": shape[1],
        "width": shape[2],
        "dtype": ",".join(dtypes),
        "crs": crs,
        "transform": (
            transform.a,
            transform.b,
            transform.c,
            transform.d,
            transform.e,
            transform.f,
        ),
        "nodata": nodata,
        "nodata_pct": nodata_pct,
        "valid_pct": None if nodata_pct is None else float(100.0 - nodata_pct),
    }
    info.update(stats)
    return info


def build_tiles_report(
    tiles_dir: str | Path,
    recursive: bool = True,
    save_csv: str | Path | None = None,
) -> pd.DataFrame:
    tiles_dir = Path(tiles_dir)
    if recursive:
        tif_paths = sorted(tiles_dir.rglob("*.tif"))
        tif_paths += sorted(tiles_dir.rglob("*.tiff"))
    else:
        tif_paths = sorted(tiles_dir.glob("*.tif"))
        tif_paths += sorted(tiles_dir.glob("*.tiff"))

    records = [inspect_tile(path) for path in tif_paths]
    df = pd.DataFrame.from_records(records)

    if save_csv is not None:
        save_csv = Path(save_csv)
        save_csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(save_csv, index=False)

    return df
