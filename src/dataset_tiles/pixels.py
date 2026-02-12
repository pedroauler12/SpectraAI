from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import rasterio


def find_chip_images(
    root_dir: str | Path,
    filename: str = "chip_2000m_multiband.tif",
    recursive: bool = True,
) -> list[Path]:
    root_dir = Path(root_dir)
    if recursive:
        return sorted(root_dir.rglob(filename))
    return sorted(root_dir.glob(filename))


def image_to_row(
    image_path: str | Path,
    flatten_order: str = "C",
    dtype: str | None = None,
) -> tuple[np.ndarray, dict]:
    image_path = Path(image_path)
    with rasterio.open(image_path) as src:
        data = src.read()
        if dtype is not None:
            data = data.astype(dtype, copy=False)
        row = data.reshape(-1, order=flatten_order)
        meta = {
            "path": str(image_path),
            "filename": image_path.name,
            "count": src.count,
            "height": src.height,
            "width": src.width,
            "dtype": str(data.dtype),
            "crs": src.crs.to_string() if src.crs else None,
            "transform": (
                src.transform.a,
                src.transform.b,
                src.transform.c,
                src.transform.d,
                src.transform.e,
                src.transform.f,
            ),
        }
    return row, meta


def build_pixel_dataset(
    root_dir: str | Path,
    filename: str = "chip_2000m_multiband.tif",
    recursive: bool = True,
    flatten_order: str = "C",
    dtype: str | None = None,
    include_meta: bool = True,
    save_csv: str | Path | None = None,
) -> pd.DataFrame:
    paths = find_chip_images(root_dir, filename=filename, recursive=recursive)
    if not paths:
        return pd.DataFrame()

    rows = []
    metas = []
    for path in paths:
        row, meta = image_to_row(path, flatten_order=flatten_order, dtype=dtype)
        rows.append(row)
        metas.append(meta)

    matrix = np.stack(rows, axis=0)
    pixel_cols = [f"pixel_{i}" for i in range(matrix.shape[1])]
    df_pixels = pd.DataFrame(matrix, columns=pixel_cols)

    if include_meta:
        df_meta = pd.DataFrame(metas)
        df = pd.concat([df_meta, df_pixels], axis=1)
    else:
        df = df_pixels

    if save_csv is not None:
        save_csv = Path(save_csv)
        save_csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(save_csv, index=False)

    return df
