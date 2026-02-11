import json
from pathlib import Path

import pandas as pd

from src.utils.prepare_dataset import prepare_dataset


def test_prepare_dataset_labels(tmp_path: Path):
    # Fake pixel dataset
    df = pd.DataFrame(
        {
            "path": [
                "/data/pos/ABC123/chip_128x128_multiband.tif",
                "/data/neg/XYZ/chip_128x128_multiband.tif",
            ],
            "filename": ["chip_128x128_multiband.tif", "chip_128x128_multiband.tif"],
            "count": [2, 2],
            "height": [2, 2],
            "width": [2, 2],
            "dtype": ["int16", "int16"],
            "crs": ["EPSG:4326", "EPSG:4326"],
            "transform": ["t", "t"],
            "pixel_0": [1, 2],
            "pixel_1": [3, 4],
        }
    )

    csv_path = tmp_path / "pixels.csv"
    df.to_csv(csv_path, index=False)

    codes = {"positivos": ["ABC123"], "negativos": ["XYZ"]}
    codes_path = tmp_path / "codes.json"
    codes_path.write_text(json.dumps(codes))

    X, y = prepare_dataset(str(csv_path), str(codes_path))

    assert list(y) == [1, 0]
    assert list(X.columns) == ["pixel_0", "pixel_1"]
