from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_origin

from src.inspecao_tiles.inspect import inspect_tile, build_tiles_report


def _write_multiband(path: Path, count=2, height=4, width=4):
    data = np.arange(count * height * width, dtype="uint16").reshape(count, height, width)
    transform = from_origin(0, 10, 1, 1)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=count,
        dtype="uint16",
        crs="EPSG:4326",
        transform=transform,
    ) as dst:
        dst.write(data)


def test_inspect_tile(tmp_path: Path):
    tif = tmp_path / "tile.tif"
    _write_multiband(tif, count=2, height=3, width=3)

    info = inspect_tile(tif)
    assert info["count"] == 2
    assert info["height"] == 3
    assert info["width"] == 3
    assert "mean_b1" in info


def test_build_tiles_report(tmp_path: Path):
    t1 = tmp_path / "a.tif"
    t2 = tmp_path / "b.tif"
    _write_multiband(t1)
    _write_multiband(t2)

    df = build_tiles_report(tmp_path, recursive=False)
    assert df.shape[0] == 2
    assert "nodata_pct" in df.columns
