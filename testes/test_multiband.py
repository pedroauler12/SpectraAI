from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_origin

from src.tiles.multiband import list_band_tifs, crop_and_stack_multiband, summarize_tif


def _write_band(path: Path, value: int, height=5, width=5):
    data = (np.ones((height, width), dtype="uint16") * value)
    transform = from_origin(0, 10, 1, 1)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=1,
        dtype="uint16",
        crs="EPSG:4326",
        transform=transform,
    ) as dst:
        dst.write(data, 1)


def test_list_band_tifs(tmp_path: Path):
    files = [
        tmp_path / "X_SRF_VNIR_B01.tif",
        tmp_path / "X_SRF_VNIR_B02.tif",
        tmp_path / "X_SRF_VNIR_B03N.tif",
        tmp_path / "X_SRF_SWIR_B04.tif",
        tmp_path / "X_SRF_SWIR_B09.tif",
        tmp_path / "X_QA.tif",
        tmp_path / "chip_2000m_multiband.tif",
    ]
    for i, f in enumerate(files):
        _write_band(f, i + 1)

    kept = list_band_tifs(tmp_path)
    names = {p.name for p in kept}

    assert "X_QA.tif" not in names
    assert "chip_2000m_multiband.tif" not in names
    assert "X_SRF_VNIR_B01.tif" in names
    assert "X_SRF_SWIR_B09.tif" in names


def test_crop_and_stack_multiband(tmp_path: Path):
    b1 = tmp_path / "A_SRF_VNIR_B01.tif"
    b2 = tmp_path / "A_SRF_SWIR_B04.tif"
    _write_band(b1, 10, height=6, width=6)
    _write_band(b2, 20, height=6, width=6)

    out = tmp_path / "chip_2000m_multiband.tif"
    bbox = (1, 5, 4, 8)  # within raster bounds (lon/lat)

    out_path = crop_and_stack_multiband([b1, b2], out, bbox)
    assert out_path is not None
    info = summarize_tif(out_path)
    assert info["bands"] == 2
    assert info["height"] > 0
    assert info["width"] > 0
