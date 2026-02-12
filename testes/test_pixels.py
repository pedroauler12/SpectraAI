from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import from_origin

from src.dataset_tiles.pixels import find_chip_images, image_to_row, build_pixel_dataset


def _write_raster(path: Path, count=2, height=4, width=3, dtype="uint16"):
    data = np.arange(count * height * width, dtype=dtype).reshape(count, height, width)
    transform = from_origin(0, 0, 1, 1)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=count,
        dtype=dtype,
        crs="EPSG:4326",
        transform=transform,
    ) as dst:
        dst.write(data)


def test_find_chip_images(tmp_path: Path):
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()

    f1 = tmp_path / "a" / "chip_2000m_multiband.tif"
    f2 = tmp_path / "b" / "chip_2000m_multiband.tif"
    f3 = tmp_path / "b" / "other.tif"
    _write_raster(f1)
    _write_raster(f2)
    _write_raster(f3)

    paths = find_chip_images(tmp_path, filename="chip_2000m_multiband.tif", recursive=True)
    assert paths == sorted([f1, f2])


def test_image_to_row_meta(tmp_path: Path):
    f1 = tmp_path / "chip_2000m_multiband.tif"
    _write_raster(f1, count=3, height=2, width=2, dtype="int16")

    row, meta = image_to_row(f1)
    assert row.shape == (3 * 2 * 2,)
    assert meta["filename"] == f1.name
    assert meta["count"] == 3
    assert meta["height"] == 2
    assert meta["width"] == 2
    assert meta["dtype"] == "int16"


def test_build_pixel_dataset(tmp_path: Path):
    f1 = tmp_path / "chip_2000m_multiband.tif"
    f2 = tmp_path / "sub" / "chip_2000m_multiband.tif"
    f2.parent.mkdir(parents=True, exist_ok=True)
    _write_raster(f1, count=2, height=3, width=4)
    _write_raster(f2, count=2, height=3, width=4)

    df = build_pixel_dataset(tmp_path, recursive=True, include_meta=True)

    pixel_cols = 2 * 3 * 4
    meta_cols = 8  # path, filename, count, height, width, dtype, crs, transform
    assert df.shape == (2, meta_cols + pixel_cols)
    assert "pixel_0" in df.columns
    assert "pixel_23" in df.columns
