from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from src.tiles.aster_tile import read_aster_tile


def _write_multiband(path: Path, count=2, height=4, width=4, dtype="uint16"):
    data = np.arange(count * height * width, dtype=dtype).reshape(count, height, width)
    transform = from_origin(0, 10, 1, 1)
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


def test_read_aster_tile_ok(tmp_path: Path):
    tif = tmp_path / "chip.tif"
    _write_multiband(tif, count=3, height=5, width=6)

    data, meta = read_aster_tile(tif, expected_bands=[3])
    assert data.shape == (3, 5, 6)
    assert meta["bands"] == 3


def test_read_aster_tile_missing_file(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        read_aster_tile(tmp_path / "missing.tif")


def test_read_aster_tile_invalid_ext(tmp_path: Path):
    bad = tmp_path / "chip.png"
    bad.write_text("x")
    with pytest.raises(ValueError):
        read_aster_tile(bad)


def test_read_aster_tile_no_crs(tmp_path: Path):
    tif = tmp_path / "no_crs.tif"
    data = np.ones((2, 4, 4), dtype="uint16")
    transform = from_origin(0, 10, 1, 1)
    with rasterio.open(
        tif,
        "w",
        driver="GTiff",
        height=4,
        width=4,
        count=2,
        dtype="uint16",
        transform=transform,
    ) as dst:
        dst.write(data)

    with pytest.raises(ValueError):
        read_aster_tile(tif, require_crs=True)
