from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import from_origin

from src.preprocessamento.recortar_banda import recortar_banda_da_amostra
from src.preprocessamento.extrair_chips_lote import extrair_chips_multiplas_amostras


def _write_multiband(path: Path, count=2, height=10, width=10):
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


def test_recortar_banda_da_amostra(tmp_path: Path):
    sample_id = 123
    sample_dir = tmp_path / str(sample_id)
    sample_dir.mkdir()

    tif_path = sample_dir / "chip_2000m_multiband.tif"
    _write_multiband(tif_path, count=2, height=10, width=10)

    df = pd.DataFrame(
        {
            "numero_amostra": [sample_id],
            "latitude_wgs84_decimal": [5],
            "longitude_wgs84_decimal": [5],
        }
    )

    chip = recortar_banda_da_amostra(sample_id, 0, df, tmp_path)
    assert chip.shape == (128, 128)


def test_extrair_chips_multiplas_amostras_mock(monkeypatch, tmp_path: Path):
    # mock recortar_banda_da_amostra para evitar IO pesado
    def fake_recortar(numero_amostra, band_index, df, aster_source_dir):
        return np.zeros((128, 128), dtype="float32")

    monkeypatch.setattr(
        "src.preprocessamento.extrair_chips_lote.recortar_banda_da_amostra",
        fake_recortar,
    )

    df = pd.DataFrame(
        {
            "numero_amostra": [1, 2],
            "latitude_wgs84_decimal": [0, 0],
            "longitude_wgs84_decimal": [0, 0],
        }
    )

    chips = extrair_chips_multiplas_amostras(
        numeros_amostras=[1, 2],
        band_index=0,
        df=df,
        aster_source_dir=str(tmp_path),
        verbose=False,
    )

    assert set(chips.keys()) == {1, 2}
    assert chips[1].shape == (128, 128)
