import json
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import from_origin

from src.dataset_tiles.pixels import build_pixel_dataset
from src.utils.prepare_dataset import prepare_dataset


def _write_raster(path: Path, count=2, height=3, width=3, dtype="uint16"):
    data = np.ones((count, height, width), dtype=dtype)
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


def test_pipeline_pixels_to_labels(tmp_path: Path):
    # Create fake chips in sample folders
    pos_id = "POS001"
    neg_id = "NEG001"

    pos_dir = tmp_path / pos_id
    neg_dir = tmp_path / neg_id
    pos_dir.mkdir()
    neg_dir.mkdir()

    _write_raster(pos_dir / "chip_128x128_multiband.tif", count=2, height=3, width=3)
    _write_raster(neg_dir / "chip_128x128_multiband.tif", count=2, height=3, width=3)

    # Build pixel dataset and save
    csv_path = tmp_path / "pixels_dataset.csv"
    df = build_pixel_dataset(
        tmp_path,
        filename="chip_128x128_multiband.tif",
        recursive=True,
        include_meta=True,
        save_csv=csv_path,
    )
    assert df.shape[0] == 2

    # Labels
    codes = {"positivos": [pos_id], "negativos": [neg_id]}
    codes_path = tmp_path / "codes.json"
    codes_path.write_text(json.dumps(codes))

    X, y = prepare_dataset(str(csv_path), str(codes_path))

    assert set(y.tolist()) == {0, 1}
    assert X.shape[0] == 2
    assert "latitude" in X.columns
    assert "longitude" in X.columns
